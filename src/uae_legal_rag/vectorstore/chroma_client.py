"""Chroma vector store (Streamlit Cloud safe)."""

from __future__ import annotations

import gc
import os
import shutil
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


def _is_streamlit() -> bool:
    """
    Reliable detection for Streamlit (local + cloud) for Streamlit <= 1.53
    """
    return "STREAMLIT_SERVER_RUN_ON_SAVE" in os.environ


def get_chroma(
    embeddings: Embeddings,
    persist_dir: str | None,
    collection_name: str,
) -> Chroma:
    """
    RULES:
    - Streamlit (local or cloud): ALWAYS in-memory Chroma
    - Non-Streamlit usage: persistent if persist_dir is provided
    """

    # ✅ Streamlit detected → force in-memory (NO persistence)
    if _is_streamlit():
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
        )

    # ✅ Non-Streamlit usage → persistent allowed
    if persist_dir:
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        return Chroma(
            collection_name=collection_name,
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )

    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
    )


def reset_chroma_collection(vs: Chroma | None) -> None:
    if vs is None:
        return
    try:
        result = vs.get()
        if result and result.get("ids"):
            vs.delete(ids=result["ids"])
    except Exception:
        pass


def reset_chroma_dir(persist_dir: str, vs: Chroma | None = None) -> None:
    reset_chroma_collection(vs)

    gc.collect()
    time.sleep(0.2)

    p = Path(persist_dir)
    if p.exists():
        try:
            shutil.rmtree(p)
        except PermissionError:
            pass
