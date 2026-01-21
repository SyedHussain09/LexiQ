from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from uae_legal_rag.ingestion.chunking import chunk_documents
from uae_legal_rag.vectorstore.chroma_client import get_chroma


class DeterministicEmbeddings(Embeddings):
    """Offline embeddings for tests (no network)."""

    def __init__(self):
        # Tiny keyword vector; stable and predictable for this project's smoke tests.
        self.keywords = [
            "penalty",
            "pdpl",
            "data",
            "termination",
            "liability",
            "indemnity",
        ]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text: str):
        t = (text or "").lower()
        # Simple counts; works well enough for predictable retrieval in a toy corpus.
        vec = []
        for kw in self.keywords:
            vec.append(float(t.count(kw)))
        # Add a small constant to avoid all-zeros for unrelated queries
        vec.append(0.1)
        return vec


def test_chunking_smoke():
    docs = [
        Document(
            page_content="Termination without notice and unlimited liability.",
            metadata={"filename": "x", "page": 1},
        )
    ]
    chunks = chunk_documents(docs)
    assert len(chunks) >= 1


def test_chroma_retrieval_smoke():
    emb = DeterministicEmbeddings()
    # In-memory Chroma avoids Windows file-lock issues during temp dir cleanup.
    vs = get_chroma(embeddings=emb, persist_dir=None, collection_name="test_collection")
    vs.add_documents(
        [
            Document(
                page_content="This contract includes a penalty clause.",
                metadata={"filename": "a", "page": 1},
            ),
            Document(
                page_content="Data protection obligations under PDPL are referenced.",
                metadata={"filename": "b", "page": 2},
            ),
        ]
    )
    out = vs.similarity_search("What does it say about penalty clauses?", k=1)
    assert out
    assert "penalty" in out[0].page_content.lower()
