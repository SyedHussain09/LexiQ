"""Local persistent Chroma vector store (cloud-safe)."""

from __future__ import annotations

import gc
import os
import shutil
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


def _is_streamlit_cloud() -> bool:
    """
    Detect Streamlit Cloud environment safely.
    Streamlit always sets STREAMLIT_RUNTIME in cloud.
    """
    return os.environ.get("STREAMLIT_RUNTIME") is not None


def get_chroma(
    embeddings: Embeddings,
    persist_dir: str | None,
    collection_name: str,
) -> Chroma:
    """
    Create a Chroma vector store.

    - Streamlit Cloud  → ALWAYS in-memory (no persistence)
    - Local machine    → Persistent if persist_dir is provided
    """

    # ✅ STREAMLIT CLOUD: force in-memory (THIS FIXES YOUR ERROR)
    if _is_streamlit_cloud():
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
        )

    # ✅ LOCAL DEVELOPMENT: persistent Chroma
    if persist_dir:
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        return Chroma(
            collection_name=collection_name,
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )

    # ✅ Fallback: in-memory
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
    )


def reset_chroma_collection(vs: Chroma | None) -> None:
    """Delete all documents from the collection (safe for Windows)."""
    if vs is None:
        return
    try:
        result = vs.get()
        if result and result.get("ids"):
            vs.delete(ids=result["ids"])
    except Exception:
        pass


def reset_chroma_dir(persist_dir: str, vs: Chroma | None = None) -> None:
    """Reset Chroma directory (local only)."""
    reset_chroma_collection(vs)

    gc.collect()
    time.sleep(0.2)

    p = Path(persist_dir)
    if p.exists():
        try:
            shutil.rmtree(p)
        except PermissionError:
            pass
