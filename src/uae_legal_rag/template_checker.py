"""Optional template checker stub.

Approach:
- Store a few "ideal template" clause docs in Chroma (per template type).
- Retrieve them and compare against uploaded contract excerpts.

This is for portfolio demo only.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from uae_legal_rag.rag.formatting import docs_to_context
from uae_legal_rag.rag.prompts import template_prompt


def seed_templates(vectorstore, template_collection_name: str) -> None:
    """Seed dummy templates (placeholder content)."""

    templates = {
        "Employment": [
            "Termination: notice periods, probation, end-of-service benefits (placeholder).",
            "Working hours, overtime, leave entitlements (placeholder).",
            "Confidentiality and IP assignment (placeholder).",
            "PDPL-style data protection and security measures (placeholder).",
        ],
        "NDA": [
            "Definition of Confidential Information and exclusions (placeholder).",
            "Permitted use and standard of care (placeholder).",
            "Return/destruction and remedies (placeholder).",
        ],
        "Services": [
            "Scope of services, acceptance, SLAs (placeholder).",
            "Payment terms, invoicing, late fees (placeholder).",
            "Limitation of liability and indemnities (placeholder).",
        ],
    }

    docs: list[Document] = []
    for ttype, clauses in templates.items():
        for idx, clause in enumerate(clauses, start=1):
            docs.append(
                Document(page_content=clause, metadata={"template_type": ttype, "clause_no": idx})
            )

    # langchain-chroma persists by collection name; easiest is to use a separate store/collection.
    # This function assumes caller created a vectorstore pointing at template_collection_name.
    vectorstore.add_documents(docs)


def run_template_check(
    llm, template_type: str, ideal_template_docs: list[Document], contract_docs: list[Document]
) -> str:
    ideal = docs_to_context(ideal_template_docs)
    contract = docs_to_context(contract_docs)
    return (template_prompt() | llm | StrOutputParser()).invoke(
        {"template_type": template_type, "ideal_template": ideal, "contract": contract}
    )
