"""LangGraph workflow: retrieve -> analyze risk -> answer.

Designed to be readable and easy to extend.
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from uae_legal_rag.rag.formatting import docs_to_context, docs_to_snippets
from uae_legal_rag.rag.prompts import qa_prompt, risk_prompt

RiskLevel = Literal["Low", "Medium", "High"]


class LegalState(BaseModel):
    question: str

    retrieved_docs: list[Document] = Field(default_factory=list)

    analysis: str = ""
    risk_level: RiskLevel = "Low"
    risk_explanation: str = ""

    answer: str = ""
    clause_snippets: list[str] = Field(default_factory=list)

    # Follow-up cycle controls (optional)
    follow_up_question: str | None = None
    run_follow_up: bool = False
    hops_remaining: int = 0


def deterministic_risk_score(analysis_text: str) -> tuple[RiskLevel, str]:
    """Explainable keyword-based scoring (demo only)."""

    t = (analysis_text or "").lower()

    high = [
        "unlimited liability",
        "termination without notice",
        "liquidated damages",
        "penalty",
        "indemnity",
        "hold harmless",
        "non-compete",
        "waiver of rights",
    ]
    medium = [
        "termination",
        "notice period",
        "liability",
        "arbitration",
        "jurisdiction",
        "governing law",
        "pdpl",
        "data protection",
        "personal data",
    ]

    hits_high = sorted({m for m in high if m in t})
    hits_med = sorted({m for m in medium if m in t})

    if hits_high:
        return "High", f"High-risk markers found: {', '.join(hits_high)}."
    if len(hits_med) >= 2:
        return "Medium", f"Multiple moderate-risk topics: {', '.join(hits_med)}."
    if hits_med:
        return "Medium", f"Moderate-risk topic detected: {', '.join(hits_med)}."

    return "Low", "No explicit high-risk markers detected by the rule layer."


def node_retrieve(state: LegalState, retriever) -> dict[str, Any]:
    docs = retriever.invoke(state.question)
    return {"retrieved_docs": docs}


def node_analyze_risk(state: LegalState, llm) -> dict[str, Any]:
    context = docs_to_context(state.retrieved_docs)
    analysis = (risk_prompt() | llm | StrOutputParser()).invoke(
        {"question": state.question, "context": context}
    )
    risk_level, explanation = deterministic_risk_score(analysis)
    return {"analysis": analysis, "risk_level": risk_level, "risk_explanation": explanation}


def node_answer(state: LegalState, llm) -> dict[str, Any]:
    context = docs_to_context(state.retrieved_docs)
    ai_response = (qa_prompt() | llm | StrOutputParser()).invoke(
        {"question": state.question, "context": context}
    )

    snippets = docs_to_snippets(state.retrieved_docs, max_items=4)

    # Build clean, modern response without redundant headers
    answer = ai_response.strip()

    # Add risk insight if relevant (not Low)
    if state.risk_level != "Low":
        answer += f"\n\n---\n\n**⚠️ Risk Assessment: {state.risk_level}**\n{state.risk_explanation}"

    # Add disclaimer subtly at the end
    answer += "\n\n---\n*This analysis is for educational purposes only and does not constitute legal advice.*"

    follow_up = None  # Let the conversation flow naturally

    return {"answer": answer, "clause_snippets": snippets, "follow_up_question": follow_up}


def node_maybe_follow_up(state: LegalState) -> dict[str, Any]:
    if not state.run_follow_up:
        return {}
    if not state.follow_up_question:
        return {}
    if state.hops_remaining <= 0:
        return {}
    return {"question": state.follow_up_question, "hops_remaining": state.hops_remaining - 1}


def build_graph(retriever, llm):
    g = StateGraph(LegalState)

    def retrieve_action(state: LegalState) -> dict[str, Any]:
        return node_retrieve(state, retriever)

    def analyze_risk_action(state: LegalState) -> dict[str, Any]:
        return node_analyze_risk(state, llm)

    def answer_action(state: LegalState) -> dict[str, Any]:
        return node_answer(state, llm)

    g.add_node("retrieve", retrieve_action)
    g.add_node("analyze_risk", analyze_risk_action)
    g.add_node("answer", answer_action)
    g.add_node("maybe_follow_up", node_maybe_follow_up)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "analyze_risk")
    g.add_edge("analyze_risk", "answer")
    g.add_edge("answer", "maybe_follow_up")

    def route(state: LegalState) -> str:
        if state.run_follow_up and state.follow_up_question and state.hops_remaining > 0:
            return "retrieve"
        return END

    g.add_conditional_edges("maybe_follow_up", route)

    return g.compile()
