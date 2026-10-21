from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    data_dir: Path = PROJECT_ROOT / os.getenv("DATA_DIR", "data")
    chroma_dir: Path = PROJECT_ROOT / os.getenv("CHROMA_DIR", "vectorstore")
    collection_name: str = "biblioteca_horizonte"
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    chunk_size: int = 2000
    chunk_overlap: int = 200
    top_k: int = 4


settings = Settings()
