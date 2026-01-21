"""PDF loaders for Streamlit uploads."""

from __future__ import annotations

import io

from langchain_core.documents import Document
from pypdf import PdfReader


def load_pdf_bytes(pdf_bytes: bytes, filename: str) -> list[Document]:
    """Return per-page Documents with filename/page metadata."""

    reader = PdfReader(io.BytesIO(pdf_bytes))
    docs: list[Document] = []

    for idx, page in enumerate(reader.pages):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            # Some PDFs have malformed fonts - skip problematic pages
            text = ""

        if not text:
            continue

        docs.append(
            Document(
                page_content=text,
                metadata={"filename": filename, "page": idx + 1},
            )
        )

    return docs
