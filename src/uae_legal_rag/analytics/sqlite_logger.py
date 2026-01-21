"""Optional local SQLite analytics (anonymized)."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def init_db(db_path: str = "./analytics.sqlite") -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS turns (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              ts_utc TEXT NOT NULL,
              question_hash TEXT NOT NULL,
              risk_level TEXT NOT NULL
            );
            """
        )
        conn.commit()


def log_turn(question: str, risk_level: str, db_path: str = "./analytics.sqlite") -> None:
    init_db(db_path)
    ts = datetime.now(timezone.utc).isoformat()
    qh = _hash(question)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO turns (ts_utc, question_hash, risk_level) VALUES (?, ?, ?)",
            (ts, qh, risk_level),
        )
        conn.commit()
