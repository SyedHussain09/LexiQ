"""Retriever factory."""

from __future__ import annotations

from langchain_core.vectorstores import VectorStore


def build_retriever(vectorstore: VectorStore, k: int = 4):
    return vectorstore.as_retriever(search_kwargs={"k": k})
