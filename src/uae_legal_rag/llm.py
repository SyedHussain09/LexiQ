"""LLM + embeddings factories.

OpenAI is the default, but keep creation centralized for easy swapping later.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from uae_legal_rag.config import Settings


def get_chat_llm(settings: Settings, api_key_override: str | None = None) -> ChatOpenAI:
    api_key = api_key_override or settings.openai_api_key
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY (set in .env or via Streamlit sidebar).")

    return ChatOpenAI(
        model_name=settings.openai_model,  # type: ignore[call-arg]
        temperature=0.1,
        openai_api_key=api_key,  # type: ignore[call-arg]
    )


def get_embeddings(settings: Settings, api_key_override: str | None = None) -> OpenAIEmbeddings:
    api_key = api_key_override or settings.openai_api_key
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY for embeddings.")

    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        openai_api_key=api_key,  # type: ignore[call-arg]
    )
