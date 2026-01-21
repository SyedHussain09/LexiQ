"""Settings loader.

Loads configuration from .env file with sensible defaults.
API key can be overridden at runtime via UI.

Models used:
- LLM: gpt-4o (OpenAI's latest multimodal model)
- Embeddings: text-embedding-3-small (fast, cost-effective)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment."""

    openai_api_key: str | None
    openai_model: str
    openai_embedding_model: str
    chroma_persist_dir: str
    chroma_collection_docs: str


def get_settings() -> Settings:
    """Load settings from .env file."""
    load_dotenv(override=False)

    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_legal")
    persist_dir = str(Path(persist_dir).resolve())

    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        chroma_persist_dir=persist_dir,
        chroma_collection_docs=os.getenv("CHROMA_COLLECTION_NAME", "uae_legal_docs"),
    )
