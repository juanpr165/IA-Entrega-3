from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class RAGResponse(BaseModel):
    """Respuesta pública del RAG, restringida al contexto recuperado."""

    respuesta: str = Field(description="Respuesta basada solamente en el contexto.")
    referencias: list[str] = Field(
        default_factory=list,
        description="Archivos fuente usados, por ejemplo '01_prestamos.md'.",
    )

    @field_validator("referencias")
    @classmethod
    def unique_references(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))
