"""Token-aware chunking tuned for legal clauses."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_legal_splitter(
    chunk_size_tokens: int = 1300, chunk_overlap_tokens: int = 200
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size_tokens,
        chunk_overlap=chunk_overlap_tokens,
        separators=["\n\n", "\n", ". ", "; ", ": ", " ", ""],
    )


def infer_section_type(text: str) -> str | None:
    t = text.lower()
    rules = {
        "termination": ["terminate", "termination", "notice period", "without notice"],
        "liability": ["liability", "limitation", "unlimited", "indemn", "hold harmless"],
        "confidentiality": ["confidential", "non-disclosure", "nda"],
        "data_protection": ["pdpl", "personal data", "data protection", "processing"],
        "governing_law": ["governing law", "jurisdiction", "courts", "arbitration"],
        "payment": ["fees", "invoice", "payment", "late fee"],
    }
    for label, pats in rules.items():
        if any(p in t for p in pats):
            return label
    return None


def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = get_legal_splitter()
    chunks = splitter.split_documents(docs)

    for d in chunks:
        section = infer_section_type(d.page_content)
        if section:
            d.metadata["section_type"] = section

    return chunks
