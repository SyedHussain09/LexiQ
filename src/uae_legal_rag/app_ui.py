"""LexiQ - AI Legal Intelligence Platform UI.

Professional dual-theme design system with proper accessibility.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import streamlit as st

from uae_legal_rag.config import get_settings
from uae_legal_rag.graph.legal_graph import LegalState, build_graph
from uae_legal_rag.ingestion.chunking import chunk_documents
from uae_legal_rag.ingestion.loaders import load_pdf_bytes
from uae_legal_rag.llm import get_chat_llm, get_embeddings
from uae_legal_rag.rag.retriever import build_retriever
from uae_legal_rag.vectorstore.chroma_client import get_chroma, reset_chroma_dir


def _get_logo_base64() -> str:
    """Get logo as base64 string for embedding in HTML."""
    logo_path = Path(__file__).parent.parent.parent / "assets" / "logo.png"
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""


def _inject_styles(dark_mode: bool = True) -> None:
    """Inject professional CSS design system."""

    if dark_mode:
        # ═══════════════════════════════════════════════════════════════
        # DARK THEME - Deep slate with violet accents
        # ═══════════════════════════════════════════════════════════════
        css_vars = """
        :root {
            --primary: #8b5cf6;
            --primary-hover: #a78bfa;
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #8b5cf6 50%, #a855f7 100%);
            --bg-page: #0f172a;
            --bg-sidebar: #0f172a;
            --bg-card: #1e293b;
            --bg-card-hover: #334155;
            --bg-input: #1e293b;
            --text-primary: #f1f5f9;
            --text-secondary: #cbd5e1;
            --text-muted: #64748b;
            --text-input: #f1f5f9;
            --text-placeholder: #64748b;
            --border: #334155;
            --border-light: #1e293b;
            --border-input: #475569;
            --cursor-color: #f1f5f9;
            --btn-bg: #1e293b;
            --btn-text: #f1f5f9;
            --btn-border: #475569;
            --btn-hover-bg: #334155;
            --success-bg: rgba(34, 197, 94, 0.15);
            --success-text: #4ade80;
            --success-border: rgba(34, 197, 94, 0.4);
            --warning-bg: rgba(251, 191, 36, 0.15);
            --warning-text: #fbbf24;
            --warning-border: rgba(251, 191, 36, 0.4);
            --danger-bg: rgba(239, 68, 68, 0.15);
            --danger-text: #f87171;
            --danger-border: rgba(239, 68, 68, 0.4);
            --hero-bg: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(168, 85, 247, 0.05) 100%);
            --stats-bg: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            --card-icon-bg: linear-gradient(135deg, rgba(139, 92, 246, 0.25), rgba(168, 85, 247, 0.25));
        }
        """
    else:
        # ═══════════════════════════════════════════════════════════════
        # LIGHT THEME - Clean white with indigo accents
        # ═══════════════════════════════════════════════════════════════
        css_vars = """
        :root {
            --primary: #6366f1;
            --primary-hover: #4f46e5;
            --primary-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            --bg-page: #f8fafc;
            --bg-sidebar: #ffffff;
            --bg-card: #ffffff;
            --bg-card-hover: #f1f5f9;
            --bg-input: #ffffff;
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --text-muted: #94a3b8;
            --text-input: #0f172a;
            --text-placeholder: #94a3b8;
            --border: #e2e8f0;
            --border-light: #f1f5f9;
            --border-input: #cbd5e1;
            --cursor-color: #0f172a;
            --btn-bg: #ffffff;
            --btn-text: #0f172a;
            --btn-border: #e2e8f0;
            --btn-hover-bg: #f1f5f9;
            --success-bg: #ecfdf5;
            --success-text: #059669;
            --success-border: #6ee7b7;
            --warning-bg: #fffbeb;
            --warning-text: #d97706;
            --warning-border: #fcd34d;
            --danger-bg: #fef2f2;
            --danger-text: #dc2626;
            --danger-border: #fca5a5;
            --hero-bg: linear-gradient(135deg, #eef2ff 0%, #faf5ff 100%);
            --stats-bg: linear-gradient(135deg, #e0e7ff 0%, #f3e8ff 100%);
            --card-icon-bg: linear-gradient(135deg, #c7d2fe, #ddd6fe);
        }
        """

    st.markdown(
        f"""
<style>
/* ═══════════════════════════════════════════════════════════════════════════
   LEXIQ DESIGN SYSTEM v2.0 - Professional Legal AI Interface
   ═══════════════════════════════════════════════════════════════════════════ */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

{css_vars}

/* ─────────────────────────────────────────────────────────────────────────────
   BASE & RESETS
   ───────────────────────────────────────────────────────────────────────────── */
body, p, span, div, h1, h2, h3, h4, h5, h6, label, input, textarea, button {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}}

* {{
    box-sizing: border-box;
}}

.stApp,
.main,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"] {{
    background: var(--bg-page) !important;
}}

.main .block-container {{
    padding: 1rem;
    max-width: 1100px;
    margin: 0 auto;
}}

@media (min-width: 768px) {{
    .main .block-container {{
        padding: 1.5rem 2rem;
    }}
}}

#MainMenu, footer{{
    visibility: hidden;
}}

/* ─────────────────────────────────────────────────────────────────────────────
   SIDEBAR
   ───────────────────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
}}

section[data-testid="stSidebar"] > div {{
    background: var(--bg-sidebar) !important;
    padding-top: 0 !important;
}}

section[data-testid="stSidebar"] * {{
    color: var(--text-primary) !important;
}}

/* Hide Material Icon text fallback for sidebar collapse button */
[data-testid="stSidebarCollapseButton"] span {{
    font-size: 0 !important;
    visibility: hidden !important;
}}

[data-testid="stSidebarCollapseButton"] svg {{
    width: 24px !important;
    height: 24px !important;
    visibility: visible !important;
}}

/* Fix expander - hide arrow icon and its text fallback completely */
[data-testid="stExpander"] summary svg {{
    display: none !important;
}}

[data-testid="stExpander"] [data-testid="stExpanderToggleIcon"] {{
    display: none !important;
}}

/* Hide the Material Icon text fallback (e.g., "arrow_right") */
[data-testid="stExpander"] summary {{
    gap: 0 !important;
}}

[data-testid="stExpander"] summary > span:not([data-testid="stMarkdownContainer"]) {{
    display: none !important;
}}

[data-testid="stExpander"] summary span[data-testid="stMarkdownContainer"] {{
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}}

.streamlit-expanderHeader p {{
    overflow: hidden !important;
    white-space: nowrap !important;
    text-overflow: ellipsis !important;
    margin: 0 !important;
}}

.sidebar-logo {{
    text-align: center;
    padding: 1.5rem 1rem;
    border-bottom: 1px solid var(--border-light);
    margin-bottom: 1.5rem;
}}

.sidebar-logo img {{
    max-width: 70px;
    height: auto;
    margin-bottom: 0.5rem;
}}

.sidebar-logo-icon {{
    font-size: 2rem;
    margin-bottom: 0.25rem;
}}

.sidebar-logo-text {{
    font-weight: 700;
    font-size: 1.25rem;
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}

.sidebar-logo-tag {{
    font-size: 0.65rem;
    color: var(--text-muted) !important;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 0.25rem;
}}

/* ─────────────────────────────────────────────────────────────────────────────
   TYPOGRAPHY
   ───────────────────────────────────────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6 {{
    color: var(--text-primary) !important;
}}

p, span, div, label, li {{
    color: var(--text-primary);
}}

.section-header {{
    font-weight: 600;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--text-muted) !important;
    margin-bottom: 0.75rem;
    padding-left: 2px;
}}

/* ─────────────────────────────────────────────────────────────────────────────
   HERO
   ───────────────────────────────────────────────────────────────────────────── */
.hero-section {{
    background: var(--hero-bg);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem 1.5rem;
    margin-bottom: 1.5rem;
    text-align: center;
}}

.hero-logo {{
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}}

.hero-title {{
    font-size: 1.75rem;
    font-weight: 800;
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.25rem 0;
}}

.hero-tagline {{
    color: var(--text-secondary) !important;
    font-size: 0.9rem;
    margin: 0;
}}

/* ─────────────────────────────────────────────────────────────────────────────
   CARDS
   ───────────────────────────────────────────────────────────────────────────── */
.pro-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
}}

.pro-card-header {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 0.75rem;
}}

.pro-card-icon {{
    width: 40px;
    height: 40px;
    background: var(--card-icon-bg);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
}}

.pro-card-title {{
    font-weight: 600;
    font-size: 1rem;
    color: var(--text-primary) !important;
}}

.pro-card-desc {{
    color: var(--text-secondary) !important;
    font-size: 0.875rem;
    line-height: 1.7;
}}

.stats-panel {{
    background: var(--stats-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
    margin-bottom: 1rem;
}}

.stats-number {{
    font-size: 2.5rem;
    font-weight: 800;
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}}

.stats-label {{
    color: var(--text-secondary) !important;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.5rem;
    font-weight: 500;
}}

/* ─────────────────────────────────────────────────────────────────────────────
   BUTTONS
   ───────────────────────────────────────────────────────────────────────────── */
.stButton > button {{
    background: var(--btn-bg) !important;
    color: var(--btn-text) !important;
    border: 1px solid var(--btn-border) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}}

.stButton > button:hover {{
    background: var(--btn-hover-bg) !important;
    border-color: var(--primary) !important;
    color: var(--primary) !important;
}}

.stButton > button:disabled {{
    background: var(--bg-card) !important;
    color: var(--text-muted) !important;
    border-color: var(--border-light) !important;
    opacity: 0.6 !important;
    cursor: not-allowed !important;
}}

.stButton > button[kind="primary"] {{
    background: var(--primary) !important;
    color: #ffffff !important;
    border: none !important;
}}

.stButton > button[kind="primary"]:hover {{
    background: var(--primary-hover) !important;
}}

/* ─────────────────────────────────────────────────────────────────────────────
   CHAT INPUT - Critical for cursor & text visibility
   ───────────────────────────────────────────────────────────────────────────── */
.stChatInput > div {{
    background: var(--bg-input) !important;
    border: 2px solid var(--border-input) !important;
    border-radius: 12px !important;
    overflow: hidden;
}}

.stChatInput > div:focus-within {{
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
}}

.stChatInput textarea {{
    background: var(--bg-input) !important;
    color: var(--text-input) !important;
    caret-color: var(--cursor-color) !important;
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    padding: 0.75rem 1rem !important;
}}

.stChatInput textarea::placeholder {{
    color: var(--text-placeholder) !important;
    opacity: 1 !important;
}}

.stChatInput button {{
    background: var(--primary) !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    margin: 4px !important;
}}

.stChatInput button:hover {{
    background: var(--primary-hover) !important;
}}

[data-testid="stChatInputContainer"],
[data-testid="stChatInput"] > div {{
    background: var(--bg-input) !important;
}}

/* ─────────────────────────────────────────────────────────────────────────────
   CHAT MESSAGES
   ───────────────────────────────────────────────────────────────────────────── */
.stChatMessage {{
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    margin-bottom: 0.75rem !important;
}}

.stChatMessage p,
.stChatMessage span,
.stChatMessage div,
.stChatMessage li,
.stChatMessage code,
.stChatMessage strong,
.stChatMessage em {{
    color: var(--text-primary) !important;
}}

.stChatMessage [data-testid="stMarkdownContainer"] * {{
    color: var(--text-primary) !important;
}}

/* ─────────────────────────────────────────────────────────────────────────────
   TEXT INPUTS
   ───────────────────────────────────────────────────────────────────────────── */
.stTextInput input,
.stTextArea textarea {{
    background: var(--bg-input) !important;
    color: var(--text-input) !important;
    caret-color: var(--cursor-color) !important;
    border: 1px solid var(--border-input) !important;
    border-radius: 8px !important;
}}

.stTextInput input:focus,
.stTextArea textarea:focus {{
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.15) !important;
}}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {{
    color: var(--text-placeholder) !important;
}}

/* ─────────────────────────────────────────────────────────────────────────────
   TABS
   ───────────────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    background: var(--bg-card);
    padding: 0.4rem;
    border-radius: 12px;
    border: 1px solid var(--border);
}}

.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.875rem;
    padding: 0.6rem 1.25rem;
    color: var(--text-secondary) !important;
}}

.stTabs [data-baseweb="tab"]:hover {{
    color: var(--text-primary) !important;
}}

.stTabs [aria-selected="true"] {{
    background: var(--card-icon-bg) !important;
    color: var(--primary) !important;
}}

/* ─────────────────────────────────────────────────────────────────────────────
   RISK BADGES
   ───────────────────────────────────────────────────────────────────────────── */
.risk-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

.risk-low {{
    background: var(--success-bg);
    color: var(--success-text) !important;
    border: 1px solid var(--success-border);
}}

.risk-medium {{
    background: var(--warning-bg);
    color: var(--warning-text) !important;
    border: 1px solid var(--warning-border);
}}

.risk-high {{
    background: var(--danger-bg);
    color: var(--danger-text) !important;
    border: 1px solid var(--danger-border);
}}

/* ─────────────────────────────────────────────────────────────────────────────
   UTILITY COMPONENTS
   ───────────────────────────────────────────────────────────────────────────── */
.source-snippet {{
    background: var(--bg-card);
    border-left: 3px solid var(--primary);
    padding: 12px 16px;
    margin: 8px 0;
    border-radius: 0 8px 8px 0;
    font-size: 0.85rem;
    color: var(--text-primary) !important;
    line-height: 1.6;
    word-wrap: break-word;
    overflow-wrap: break-word;
    white-space: pre-wrap;
    max-width: 100%;
}}

.step-item {{
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 0.75rem 0;
    border-bottom: 1px solid var(--border-light);
}}

.step-item:last-child {{
    border-bottom: none;
}}

.step-number {{
    width: 26px;
    height: 26px;
    background: var(--primary);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 0.75rem;
    color: white !important;
    flex-shrink: 0;
}}

.step-content {{
    flex: 1;
}}

.step-title {{
    font-weight: 600;
    font-size: 0.875rem;
    color: var(--text-primary) !important;
    margin-bottom: 2px;
}}

.step-desc {{
    color: var(--text-secondary) !important;
    font-size: 0.8rem;
    line-height: 1.5;
}}

.feature-grid {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.5rem;
    margin-top: 0.75rem;
}}

.feature-item {{
    background: var(--bg-card);
    border: 1px solid var(--border-light);
    border-radius: 8px;
    padding: 0.6rem 0.75rem;
    display: flex;
    align-items: center;
    gap: 8px;
}}

.feature-icon {{
    font-size: 1rem;
}}

.feature-text {{
    font-size: 0.75rem;
    color: var(--text-secondary) !important;
    font-weight: 500;
}}

.empty-state {{
    text-align: center;
    padding: 3rem 2rem;
    background: var(--bg-card);
    border: 2px dashed var(--border);
    border-radius: 16px;
    margin: 1rem 0;
}}

.empty-icon {{
    font-size: 3rem;
    margin-bottom: 1rem;
    opacity: 0.5;
}}

.empty-title {{
    font-weight: 600;
    font-size: 1.1rem;
    color: var(--text-primary) !important;
    margin-bottom: 0.5rem;
}}

.empty-desc {{
    color: var(--text-muted) !important;
    font-size: 0.9rem;
}}

.divider {{
    height: 1px;
    background: var(--border-light);
    margin: 1.25rem 0;
}}

.app-footer {{
    text-align: center;
    padding: 1.5rem 1rem;
    margin-top: 2rem;
    border-top: 1px solid var(--border);
}}

.footer-brand {{
    font-weight: 600;
    font-size: 0.875rem;
    color: var(--text-secondary) !important;
    margin-bottom: 0.25rem;
}}

.footer-tagline {{
    font-size: 0.75rem;
    color: var(--text-muted) !important;
}}

/* ─────────────────────────────────────────────────────────────────────────────
   FILE UPLOADER
   ───────────────────────────────────────────────────────────────────────────── */
.stFileUploader > div {{
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    background: var(--bg-card) !important;
}}

.stFileUploader label {{
    color: var(--text-primary) !important;
}}

.stProgress > div > div {{
    background: var(--primary-gradient) !important;
    border-radius: 10px;
}}

.streamlit-expanderHeader {{
    font-size: 0.875rem;
    font-weight: 500;
    color: var(--text-primary) !important;
    background: var(--bg-card) !important;
}}

/* ─────────────────────────────────────────────────────────────────────────────
   MARKDOWN OVERRIDES
   ───────────────────────────────────────────────────────────────────────────── */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {{
    color: var(--text-primary) !important;
}}

.stMarkdown p,
.stMarkdown li,
.stMarkdown span {{
    color: var(--text-primary) !important;
}}

/* ─────────────────────────────────────────────────────────────────────────────
   RESPONSIVE
   ───────────────────────────────────────────────────────────────────────────── */
@media (max-width: 767px) {{
    [data-testid="column"] {{
        width: 100% !important;
        flex: 1 1 100% !important;
    }}

    .hero-section {{
        padding: 1.25rem 1rem;
    }}

    .hero-title {{
        font-size: 1.5rem;
    }}
}}
</style>
""",
        unsafe_allow_html=True,
    )


def _risk_badge(level: str) -> str:
    """Generate HTML for risk badge."""
    lvl = (level or "Low").capitalize()
    icons = {"Low": "✓", "Medium": "⚡", "High": "⚠"}
    return f'<span class="risk-badge risk-{lvl.lower()}">{icons.get(lvl, "●")} {lvl} Risk</span>'


def _doc_count(vs) -> int:
    """Get document count from vectorstore."""
    if not vs:
        return 0
    try:
        return int(vs._collection.count())
    except Exception:
        return 0


def render_app() -> None:
    """Main app renderer."""
    settings = get_settings()

    # Initialize theme in session state
    if "dark_mode" not in st.session_state:
        st.session_state["dark_mode"] = True  # Default to dark

    _inject_styles(st.session_state["dark_mode"])

    if not settings.openai_api_key:
        _show_setup_required()
        return

    vs = _init_vectorstore(settings)
    count = _doc_count(vs)

    # Sidebar
    logo_b64 = _get_logo_base64()
    with st.sidebar:
        if logo_b64:
            st.markdown(
                f"""
                <div class="sidebar-logo">
                    <img src="data:image/png;base64,{logo_b64}" alt="LexiQ">
                    <div class="sidebar-logo-tag">Legal AI Platform</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="sidebar-logo">
                    <div class="sidebar-logo-icon">⚖️</div>
                    <div class="sidebar-logo-text">LexiQ</div>
                    <div class="sidebar-logo-tag">Legal AI Platform</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Theme Toggle
        st.markdown('<div class="section-header">Appearance</div>', unsafe_allow_html=True)
        theme_col1, theme_col2 = st.columns([1, 1])
        with theme_col1:
            if st.button(
                "🌙 Dark" if not st.session_state["dark_mode"] else "🌙 Dark ✓",
                use_container_width=True,
                disabled=st.session_state["dark_mode"],
                key="btn_theme_dark",
            ):
                st.session_state["dark_mode"] = True
                st.rerun()
        with theme_col2:
            if st.button(
                "☀️ Light" if st.session_state["dark_mode"] else "☀️ Light ✓",
                use_container_width=True,
                disabled=not st.session_state["dark_mode"],
                key="btn_theme_light",
            ):
                st.session_state["dark_mode"] = False
                st.rerun()

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Stats
        st.markdown(
            f"""
            <div class="stats-panel">
                <div class="stats-number">{count}</div>
                <div class="stats-label">Documents Indexed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Actions
        st.markdown('<div class="section-header">Quick Actions</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        if c1.button(
            "🗑️ Reset", use_container_width=True, help="Clear all documents", key="btn_reset"
        ):
            reset_chroma_dir(settings.chroma_persist_dir, st.session_state.get("vs"))
            dark_mode = st.session_state.get("dark_mode", True)
            st.session_state.clear()
            st.session_state["dark_mode"] = dark_mode
            st.rerun()
        if c2.button(
            "💬 New Chat",
            use_container_width=True,
            help="Start fresh conversation",
            key="btn_new_chat",
        ):
            st.session_state["messages"] = []
            st.rerun()

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # How to Use
        st.markdown('<div class="section-header">How to Use</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="step-item">
                <div class="step-number">1</div>
                <div class="step-content">
                    <div class="step-title">Upload Documents</div>
                    <div class="step-desc">Go to Documents tab and upload your legal PDFs</div>
                </div>
            </div>
            <div class="step-item">
                <div class="step-number">2</div>
                <div class="step-content">
                    <div class="step-title">Process Files</div>
                    <div class="step-desc">Click Process to index your documents</div>
                </div>
            </div>
            <div class="step-item">
                <div class="step-number">3</div>
                <div class="step-content">
                    <div class="step-title">Ask Questions</div>
                    <div class="step-desc">Use AI Chat to analyze contracts</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Features
        st.markdown('<div class="section-header">Capabilities</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="feature-grid">
                <div class="feature-item">
                    <span class="feature-icon">🔍</span>
                    <span class="feature-text">Risk Analysis</span>
                </div>
                <div class="feature-item">
                    <span class="feature-icon">📋</span>
                    <span class="feature-text">Clause Detection</span>
                </div>
                <div class="feature-item">
                    <span class="feature-icon">⚡</span>
                    <span class="feature-text">Quick Insights</span>
                </div>
                <div class="feature-item">
                    <span class="feature-icon">🎯</span>
                    <span class="feature-text">Smart Search</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Main Content - Hero
    if logo_b64:
        st.markdown(
            f"""
            <div class="hero-section">
                <img src="data:image/png;base64,{logo_b64}" style="width: 100px; height: auto; margin-bottom: 0.5rem;" alt="LexiQ">
                <p class="hero-tagline">AI-Powered Legal Document Intelligence</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="hero-section">
                <div class="hero-logo">⚖️</div>
                <h1 class="hero-title">LexiQ</h1>
                <p class="hero-tagline">AI-Powered Legal Document Intelligence</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["💬 AI Chat", "📄 Documents", "ℹ️ About"])

    with tab1:
        _chat_ui(settings, vs, count)

    with tab2:
        _upload_ui(settings)

    with tab3:
        _about_ui()

    # Footer
    st.markdown(
        """
        <div class="app-footer">
            <div class="footer-brand">LexiQ • Educational AI System</div>
            <div class="footer-tagline">Local-first • Privacy-conscious • Risk-aware</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _show_setup_required() -> None:
    """Show setup required message."""
    if "dark_mode" not in st.session_state:
        st.session_state["dark_mode"] = True
    _inject_styles(st.session_state["dark_mode"])

    st.markdown(
        """
        <div class="hero-section">
            <div class="hero-logo">⚖️</div>
            <h1 class="hero-title">LexiQ</h1>
            <p class="hero-tagline">AI-Powered Legal Document Intelligence</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="pro-card" style="text-align: center; max-width: 500px; margin: 2rem auto;">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">🔐</div>
            <div style="font-weight: 700; font-size: 1.1rem; margin-bottom: 0.75rem; color: var(--text-primary);">Configuration Required</div>
            <div style="color: var(--text-secondary); font-size: 0.9rem; line-height: 1.6; margin-bottom: 1rem;">
                To get started, add your OpenAI API key to the environment file
            </div>
            <div style="background: var(--stats-bg); padding: 12px 20px; border-radius: 8px; border: 1px solid var(--border);">
                <code style="color: var(--primary); font-size: 0.85rem;">OPENAI_API_KEY=sk-...</code>
            </div>
            <div style="color: var(--text-muted); font-size: 0.8rem; margin-top: 1rem;">
                Add this to your <code>.env</code> file in the project root
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _init_vectorstore(settings):
    """Initialize vectorstore."""
    if "vs" not in st.session_state:
        emb = get_embeddings(settings)
        st.session_state["vs"] = get_chroma(
            embeddings=emb,
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_docs,
        )
    return st.session_state["vs"]


def _chat_ui(settings, vs, count: int) -> None:
    """Render chat interface."""
    if count == 0:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-icon">📄</div>
                <div class="empty-title">No Documents Indexed</div>
                <div class="empty-desc">Upload legal documents in the Documents tab to start analyzing</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.chat_input(
            "Upload documents first to start chatting...", disabled=True, key="chat_disabled"
        )
        return

    # Quick prompts
    st.markdown('<div class="section-header">Quick Analysis</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    prompts = ["🔍 Key risks?", "📋 Termination clauses?", "⚖️ Liability limits?", "📅 Key dates?"]
    for i, (col, p) in enumerate(zip(cols, prompts, strict=False)):
        if col.button(p, use_container_width=True, key=f"quick_{i}"):
            st.session_state["_q"] = p.split(" ", 1)[1]
            st.rerun()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Chat messages
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for idx, m in enumerate(st.session_state["messages"]):
        avatar = "👤" if m["role"] == "user" else "⚖️"
        with st.chat_message(m["role"], avatar=avatar):
            st.markdown(m["content"], unsafe_allow_html=True)

    # Chat input
    q = st.session_state.pop("_q", None) or st.chat_input(
        "Ask anything about your legal documents...", key="chat_main"
    )

    if q:
        st.session_state["messages"].append({"role": "user", "content": q})
        with st.chat_message("user", avatar="👤"):
            st.markdown(q)

        with st.chat_message("assistant", avatar="⚖️"):
            with st.spinner("Thinking..."):
                try:
                    llm = get_chat_llm(settings)

                    # Smart intent classification using LLM
                    intent_prompt = f"""Classify this user message into one of these categories:
- "greeting": Simple greetings like hi, hello, hey, good morning, etc.
- "about": Questions about you (the AI), your capabilities, who made you, etc.
- "general": General questions NOT about documents (weather, math, coding, etc.)
- "document": Questions about legal documents, contracts, clauses, analysis, etc.

User message: "{q}"

Reply with ONLY one word: greeting, about, general, or document"""

                    intent_response = llm.invoke(intent_prompt)
                    # Extract content safely
                    if hasattr(intent_response, "content"):
                        raw_content = intent_response.content
                        if isinstance(raw_content, list):
                            intent = (
                                str(raw_content[0]).strip().lower() if raw_content else "document"
                            )
                        else:
                            intent = str(raw_content).strip().lower()
                    else:
                        intent = str(intent_response).strip().lower()

                    # Handle based on intent
                    if "greeting" in intent:
                        response = """Hello! 👋 I'm **LexiQ**, your AI legal document assistant.

I can help you with:
- 📄 **Analyzing contracts** and legal documents
- ⚠️ **Identifying risks** and potential issues  
- 🔍 **Finding specific clauses** or terms
- 📝 **Summarizing** key points

**How can I assist you today?** Try asking something like:
- *"Summarize this contract"*
- *"What are the payment terms?"*
- *"Are there any risky clauses?"*"""
                        st.markdown(response)
                        st.session_state["messages"].append(
                            {"role": "assistant", "content": response}
                        )

                    elif "about" in intent:
                        response = """I'm **LexiQ** ⚖️ — an AI-powered legal document analysis assistant.

**What I do:**
- Analyze contracts, agreements, and legal documents
- Identify potential risks and red flags
- Extract key terms, clauses, and obligations
- Provide educational summaries (not legal advice)

**Built with:**
- 🧠 Advanced language models for understanding
- 🔍 RAG (Retrieval-Augmented Generation) for accuracy
- 📊 Risk assessment algorithms

I'm here to help you understand your legal documents better. What would you like to analyze?"""
                        st.markdown(response)
                        st.session_state["messages"].append(
                            {"role": "assistant", "content": response}
                        )

                    elif "general" in intent:
                        # Use LLM for general conversation but remind about capabilities
                        general_resp = llm.invoke(f"""You are LexiQ, a friendly AI legal document assistant. 
The user asked a general question (not about documents). Give a brief, helpful response, 
then gently remind them you specialize in legal document analysis.

User: {q}""")
                        general_response = (
                            general_resp.content
                            if hasattr(general_resp, "content")
                            else str(general_resp)
                        )
                        st.markdown(general_response)
                        st.session_state["messages"].append(
                            {"role": "assistant", "content": general_response}
                        )

                    else:
                        # Document-related query - use full RAG pipeline
                        if "retriever" not in st.session_state:
                            st.session_state["retriever"] = build_retriever(vs, k=4)

                        graph = build_graph(st.session_state["retriever"], llm)
                        raw: Any = graph.invoke(LegalState(question=q))
                        out = LegalState.model_validate(raw) if isinstance(raw, dict) else raw

                        st.markdown(_risk_badge(out.risk_level), unsafe_allow_html=True)
                        st.markdown("")
                        st.markdown(out.answer)

                        if out.clause_snippets:
                            # Custom styled source references section
                            snippets_html = "".join(
                                f'<div class="source-snippet">{s}</div>' for s in out.clause_snippets
                            )
                            st.markdown(
                                f"""
                                <details class="source-references-box">
                                    <summary class="source-references-header">📋 Source References</summary>
                                    <div class="source-references-content">
                                        {snippets_html}
                                    </div>
                                </details>
                                """,
                                unsafe_allow_html=True,
                            )

                        st.session_state["messages"].append(
                            {
                                "role": "assistant",
                                "content": f"{_risk_badge(out.risk_level)}<br><br>{out.answer}",
                            }
                        )
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")


def _upload_ui(settings) -> None:
    """Render upload interface."""
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
            <div class="pro-card">
                <div class="pro-card-header">
                    <div class="pro-card-icon">📄</div>
                    <div class="pro-card-title">Upload Legal Documents</div>
                </div>
                <div class="pro-card-desc">
                    Upload PDF contracts, agreements, and legal documents for AI-powered analysis.
                    Supported formats: PDF files up to 200MB each.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        files = st.file_uploader(
            "Drop PDF files here or click to browse",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            help="Upload one or more PDF documents for analysis",
            key="file_uploader",
        )

        if files:
            st.markdown(f"**{len(files)} file(s) selected**")
            for f in files:
                st.markdown(f"- 📄 {f.name} ({f.size / 1024:.1f} KB)")

        if files and st.button(
            "🚀 Process Documents", type="primary", use_container_width=True, key="btn_process"
        ):
            vs = _init_vectorstore(settings)
            prog = st.progress(0, text="Initializing...")

            pages = []
            for i, f in enumerate(files):
                prog.progress((i + 1) / (len(files) + 2), f"Loading: {f.name}")
                pages.extend(load_pdf_bytes(f.read(), f.name))

            if not pages:
                st.error("⚠️ No readable text found in the uploaded PDFs")
                return

            prog.progress(0.7, "Processing and chunking documents...")
            chunks = chunk_documents(pages)

            prog.progress(0.9, "Building search index...")
            vs.add_documents(chunks)
            st.session_state["retriever"] = build_retriever(vs, k=4)

            prog.progress(1.0, "Complete!")
            st.success(
                f"✅ Successfully indexed {len(chunks)} document sections from {len(files)} file(s)"
            )

    with col2:
        st.markdown(
            """
            <div class="pro-card">
                <div class="pro-card-header">
                    <div class="pro-card-icon">💡</div>
                    <div class="pro-card-title">Tips</div>
                </div>
                <div class="pro-card-desc">
                    <strong>Best Results:</strong><br>
                    • Use searchable PDFs<br>
                    • Ensure clear text quality<br>
                    • Multiple docs supported<br><br>
                    <strong>After Upload:</strong><br>
                    Go to AI Chat tab to analyze your documents
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _about_ui() -> None:
    """Render about section."""
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown(
            """
            <div class="pro-card">
                <div class="pro-card-header">
                    <div class="pro-card-icon">⚖️</div>
                    <div class="pro-card-title">What is LexiQ?</div>
                </div>
                <div class="pro-card-desc" style="line-height: 1.8;">
                    LexiQ helps you understand legal documents faster. Upload your contracts or agreements,
                    and ask questions in plain English. It reads through your documents and gives you
                    clear answers with risk assessments.
                    <br><br>
                    Built for learning and exploration — perfect if you want to quickly review contracts
                    without reading every page yourself.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="pro-card">
                <div class="pro-card-header">
                    <div class="pro-card-icon">📖</div>
                    <div class="pro-card-title">Quick Start</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="step-item">
                <div class="step-number">1</div>
                <div class="step-content">
                    <div class="step-title">Upload PDFs</div>
                    <div class="step-desc">Go to Documents tab, drop your files</div>
                </div>
            </div>
            <div class="step-item">
                <div class="step-number">2</div>
                <div class="step-content">
                    <div class="step-title">Process</div>
                    <div class="step-desc">Hit the Process button and wait a few seconds</div>
                </div>
            </div>
            <div class="step-item">
                <div class="step-number">3</div>
                <div class="step-content">
                    <div class="step-title">Ask Away</div>
                    <div class="step-desc">Go to AI Chat and ask anything about your docs</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="pro-card">
                <div class="pro-card-header">
                    <div class="pro-card-icon">✨</div>
                    <div class="pro-card-title">What You Can Do</div>
                </div>
                <div class="pro-card-desc" style="line-height: 1.8;">
                    🔍 Find risks and red flags<br>
                    📋 Spot important clauses<br>
                    💬 Ask questions naturally<br>
                    📄 Analyze multiple docs<br>
                    ⚡ Get quick summaries<br>
                    🔒 Everything stays local
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="pro-card">
                <div class="pro-card-header">
                    <div class="pro-card-icon">💬</div>
                    <div class="pro-card-title">Try Asking</div>
                </div>
                <div class="pro-card-desc" style="line-height: 1.8;">
                    "What are the main risks here?"<br>
                    "How can I terminate this contract?"<br>
                    "Any liability caps I should know?"<br>
                    "Summarize the payment terms"<br>
                    "What happens if they breach?"
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
