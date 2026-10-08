"""
Advice log: drafts the Content Advisor produced and what the professor decided about them.

Nothing the advisor writes reaches students on its own. A draft clarification or a draft quiz sits here
with status "draft" until a professor approves it. The decision time is kept, so that later (see
`rag_core/confusion.py`) the change in a concept's confusion before and after an approved action can be
measured.

Kinds:
  clarification   a short note for students on a concept; once approved it is added to the course index
  remedial_quiz   a quiz on a concept; `ref` is the quiz id, and it is hidden from students until approved
  action          something done outside the app (re-taught in class, updated a slide), logged by hand
"""

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rag_core.config import DB_DIR, QUERY_LOG_FILENAME

KIND_CLARIFICATION = "clarification"
KIND_REMEDIAL_QUIZ = "remedial_quiz"
KIND_ACTION = "action"      # something the professor did outside the app, logged by hand

STATUS_DRAFT = "draft"
STATUS_APPROVED = "approved"
STATUS_DISMISSED = "dismissed"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS advice (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    concept_key TEXT NOT NULL,
    concept     TEXT NOT NULL,
    kind        TEXT NOT NULL,
    title       TEXT NOT NULL,
    body        TEXT NOT NULL,
    status      TEXT NOT NULL,
    decided_at  TEXT,
    ref         INTEGER
)
"""


def _connect(db_dir: str | Path = DB_DIR) -> sqlite3.Connection:
    conn = sqlite3.connect(Path(db_dir) / QUERY_LOG_FILENAME)
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def add_advice(
    concept_key: str,
    concept: str,
    kind: str,
    title: str,
    body: str = "",
    ref: Optional[int] = None,
    db_dir: str | Path = DB_DIR,
) -> int:
    """Save a new draft. Returns its id."""
    with closing(_connect(db_dir)) as conn, conn:
        cursor = conn.execute(
            "INSERT INTO advice (created_at, concept_key, concept, kind, title, body, status, ref) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (_now(), concept_key, concept, kind, title, body, STATUS_DRAFT, ref),
        )
        return int(cursor.lastrowid)


def get_advice(advice_id: int, db_dir: str | Path = DB_DIR) -> Optional[Dict[str, Any]]:
    with closing(_connect(db_dir)) as conn:
        row = conn.execute("SELECT * FROM advice WHERE id = ?", (advice_id,)).fetchone()
    return dict(row) if row else None


def list_advice(
    concept_key: Optional[str] = None,
    kind: Optional[str] = None,
    status: Optional[str] = None,
    db_dir: str | Path = DB_DIR,
) -> List[Dict[str, Any]]:
    """Saved items, newest first, optionally filtered."""
    clauses, params = [], []
    for column, value in (("concept_key", concept_key), ("kind", kind), ("status", status)):
        if value is not None:
            clauses.append(f"{column} = ?")
            params.append(value)
    sql = "SELECT * FROM advice" + (" WHERE " + " AND ".join(clauses) if clauses else "") + " ORDER BY id DESC"
    with closing(_connect(db_dir)) as conn:
        return [dict(r) for r in conn.execute(sql, tuple(params)).fetchall()]


def update_body(advice_id: int, body: str, db_dir: str | Path = DB_DIR) -> None:
    """Save the professor's edits to a draft."""
    with closing(_connect(db_dir)) as conn, conn:
        conn.execute("UPDATE advice SET body = ? WHERE id = ?", (body, advice_id))


def set_status(advice_id: int, status: str, db_dir: str | Path = DB_DIR) -> None:
    """Approve or dismiss an item. The time of the decision is recorded."""
    decided = _now() if status in (STATUS_APPROVED, STATUS_DISMISSED) else None
    with closing(_connect(db_dir)) as conn, conn:
        conn.execute("UPDATE advice SET status = ?, decided_at = ? WHERE id = ?", (status, decided, advice_id))


def log_action(
    concept_key: str,
    concept: str,
    note: str,
    when: Optional[str] = None,
    db_dir: str | Path = DB_DIR,
) -> int:
    """
    Record something the professor did outside the app (re-taught it in class, updated a slide) so its effect can
    be measured like any other action. `when` is an ISO UTC time (default: now). It is saved as already approved.
    """
    item = add_advice(concept_key, concept, KIND_ACTION, note, body=note, db_dir=db_dir)
    with closing(_connect(db_dir)) as conn, conn:
        conn.execute(
            "UPDATE advice SET status = ?, decided_at = ? WHERE id = ?", (STATUS_APPROVED, when or _now(), item)
        )
    return item


def delete_by_title_prefix(prefix: str, db_dir: str | Path = DB_DIR) -> int:
    """Delete items whose title starts with `prefix` (used to clear simulated demo actions)."""
    escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    with closing(_connect(db_dir)) as conn, conn:
        return conn.execute("DELETE FROM advice WHERE title LIKE ? ESCAPE '\\'", (escaped + "%",)).rowcount


def approved_clarifications(db_dir: str | Path = DB_DIR) -> List[Tuple[str, str]]:
    """(concept, text) for every approved clarification, oldest first. Re-added to the index on a rebuild."""
    with closing(_connect(db_dir)) as conn:
        rows = conn.execute(
            "SELECT concept, body FROM advice WHERE kind = ? AND status = ? ORDER BY id",
            (KIND_CLARIFICATION, STATUS_APPROVED),
        ).fetchall()
    return [(r["concept"], r["body"]) for r in rows]
