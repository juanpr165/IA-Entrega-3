from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings

MANIFEST_NAME = "ingestion_manifest.json"


def get_embeddings() -> HuggingFaceEmbeddings:
    """Create the single embedding configuration used by ingestion and retrieval."""
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def documents_fingerprint(data_dir: Path) -> str:
    """Hash supported input files and index-relevant configuration."""
    digest = hashlib.sha256()
    digest.update(settings.embedding_model.encode())
    digest.update(f"{settings.chunk_size}:{settings.chunk_overlap}".encode())
    for path in sorted(data_dir.glob("**/*")):
        if path.suffix.lower() in {".txt", ".md"} and path.is_file():
            digest.update(path.relative_to(data_dir).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def load_documents(data_dir: Path):
    if not data_dir.exists():
        raise FileNotFoundError(f"No existe el directorio de datos: {data_dir}")
    loaders = [
        DirectoryLoader(str(data_dir), glob="**/*.txt", loader_cls=TextLoader),
        DirectoryLoader(str(data_dir), glob="**/*.md", loader_cls=TextLoader),
    ]
    documents = [document for loader in loaders for document in loader.load()]
    if not documents:
        raise ValueError("No se encontraron archivos .txt o .md para indexar.")
    for document in documents:
        document.metadata["source"] = Path(document.metadata["source"]).name
    return documents


def ingest(force: bool = False) -> bool:
    """Persist documents in Chroma. Returns True only when indexing was performed."""
    fingerprint = documents_fingerprint(settings.data_dir)
    manifest_path = settings.chroma_dir / MANIFEST_NAME
    if not force and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") == fingerprint:
            print("El índice está actualizado; no se reindexó.")
            return False

    if settings.chroma_dir.exists():
        # The manifest differed (or --force was requested): rebuild a coherent collection.
        shutil.rmtree(settings.chroma_dir)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(load_documents(settings.data_dir))
    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=settings.collection_name,
        persist_directory=str(settings.chroma_dir),
    )
    manifest_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "embedding_model": settings.embedding_model,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "documents": len({chunk.metadata["source"] for chunk in chunks}),
                "chunks": len(chunks),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Índice creado: {len(chunks)} fragmentos en {settings.chroma_dir}.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexa data/ en ChromaDB local.")
    parser.add_argument("--force", action="store_true", help="Reconstruye el índice.")
    args = parser.parse_args()
    ingest(force=args.force)
