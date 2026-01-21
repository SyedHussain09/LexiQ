"""Local persistent Chroma vector store."""

from __future__ import annotations

import gc
import shutil
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


def get_chroma(embeddings: Embeddings, persist_dir: str | None, collection_name: str) -> Chroma:
    """Create a Chroma vector store.

    - If `persist_dir` is provided, Chroma persists locally.
    - If `persist_dir` is None, Chroma runs in-memory (useful for tests).
    """

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
    """Delete all documents from the collection (safe for Windows)."""
    if vs is None:
        return
    try:
        # Get all document IDs and delete them
        result = vs.get()
        if result and result.get("ids"):
            vs.delete(ids=result["ids"])
    except Exception:
        pass


def reset_chroma_dir(persist_dir: str, vs: Chroma | None = None) -> None:
    """Reset Chroma directory. On Windows, close connections first."""
    # First try to clear the collection if we have a reference
    reset_chroma_collection(vs)

    # Force garbage collection to release file handles
    gc.collect()
    time.sleep(0.2)

    p = Path(persist_dir)
    if p.exists():
        try:
            shutil.rmtree(p)
        except PermissionError:
            # On Windows, files may still be locked - just clear the collection instead
            pass
