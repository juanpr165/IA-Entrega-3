from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_groq import ChatGroq

from src.config import settings
from src.ingestion import get_embeddings
from src.schemas import RAGResponse


def _format_context(documents) -> str:
    return "\n\n".join(
        f"[Fuente: {doc.metadata['source']}]\n{doc.page_content}" for doc in documents
    )


def _require_index() -> None:
    if not settings.chroma_dir.exists():
        raise FileNotFoundError(
            "No existe el índice local. Ejecute primero: python -m src.ingestion"
        )


def build_rag_chain():
    """Build the LCEL chain without making a model request."""
    _require_index()
    vectorstore = Chroma(
        collection_name=settings.collection_name,
        persist_directory=str(settings.chroma_dir),
        embedding_function=get_embeddings(),
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.top_k})
    parser = PydanticOutputParser(pydantic_object=RAGResponse)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Eres un asistente de la Biblioteca Horizonte. Responde exclusivamente "
                "con el CONTEXTO recuperado. No uses conocimiento externo ni inventes datos. "
                "Si la respuesta no está explícitamente respaldada por el contexto, responde "
                "exactamente 'No lo sé' y usa una lista vacía de referencias. "
                "Devuelve únicamente JSON válido siguiendo estas instrucciones de formato:\n"
                "{format_instructions}",
            ),
            ("human", "CONTEXTO:\n{context}\n\nPREGUNTA: {question}"),
        ]
    )
    model = ChatGroq(model=settings.groq_model, temperature=0)
    return (
        {
            "context": retriever | RunnableLambda(_format_context),
            "question": RunnablePassthrough(),
            "format_instructions": lambda _: parser.get_format_instructions(),
        }
        | prompt
        | model
        | parser
    )


async def get_rag_response(query: str) -> RAGResponse:
    """Answer asynchronously using only relevant local documents."""
    question = query.strip()
    if not question:
        raise ValueError("La consulta no puede estar vacía.")
    return await build_rag_chain().ainvoke(question)
