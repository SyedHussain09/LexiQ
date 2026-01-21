"""LexiQ - AI Legal Intelligence Platform."""

from __future__ import annotations

import streamlit as st

from uae_legal_rag.app_ui import render_app


def main() -> None:
    st.set_page_config(
        page_title="LexiQ",
        page_icon="⚖️",
        layout="wide",
    )
    render_app()


if __name__ == "__main__":
    main()
