# Project Structure

```
LexiQ/
├── app.py                    # Application entry point
├── src/
│   └── uae_legal_rag/
│       ├── app_ui.py         # Streamlit UI with dual themes
│       ├── config.py         # Settings and configuration
│       ├── llm.py            # OpenAI LLM setup
│       ├── graph/
│       │   └── legal_graph.py    # LangGraph workflow
│       ├── ingestion/
│       │   ├── chunking.py       # Document chunking
│       │   └── loaders.py        # PDF loading
│       ├── rag/
│       │   ├── prompts.py        # LLM prompts
│       │   ├── retriever.py      # Vector retrieval
│       │   └── formatting.py     # Output formatting
│       └── vectorstore/
│           └── chroma_client.py  # ChromaDB client
├── scripts/
│   ├── dev.ps1               # Windows dev script
│   └── dev.sh                # Unix dev script
├── tests/
│   └── test_smoke.py         # Smoke tests
└── assets/
    └── logo.png              # Application logo
```

## Key Components

| Directory | Purpose |
|-----------|---------|
| `src/uae_legal_rag/` | Core application code |
| `graph/` | LangGraph workflow orchestration |
| `ingestion/` | PDF loading and document chunking |
| `rag/` | Retrieval and prompt templates |
| `vectorstore/` | ChromaDB vector storage |
| `scripts/` | Development automation |
| `tests/` | Test suite |
