"""Formatting helpers."""

from __future__ import annotations

from langchain_core.documents import Document


def docs_to_context(docs: list[Document]) -> str:
    parts: list[str] = []
    for d in docs:
        meta = d.metadata or {}
        src = meta.get("filename", "unknown")
        page = meta.get("page", "?")
        section = meta.get("section_type")
        header = f"[{src} | p.{page}]" + (f" ({section})" if section else "")
        parts.append(header)
        parts.append(d.page_content.strip())
        parts.append("")
    return "\n".join(parts).strip()


def docs_to_snippets(docs: list[Document], max_items: int = 4, max_chars: int = 380) -> list[str]:
    out: list[str] = []
    for d in docs[:max_items]:
        meta = d.metadata or {}
        src = meta.get("filename", "unknown")
        page = meta.get("page", "?")
        text = d.page_content.replace("\n", " ").strip()
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."
        out.append(f"[{src} p.{page}] {text}")
    return out
