from __future__ import annotations

import asyncio
import sys

from src.rag import get_rag_response


async def ask(question: str) -> None:
    result = await get_rag_response(question)
    print(f"\nRespuesta: {result.respuesta}")
    print("Referencias:")
    for reference in result.referencias:
        print(f"- {reference}")


def main() -> None:
    if len(sys.argv) > 1:
        asyncio.run(ask(" ".join(sys.argv[1:])))
        return
    print("RAG local — escriba una consulta (o 'salir').")
    while True:
        question = input("> ").strip()
        if question.lower() in {"salir", "exit", "quit"}:
            break
        if question:
            asyncio.run(ask(question))


if __name__ == "__main__":
    main()
