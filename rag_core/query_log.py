"""
Question log: a small SQLite database recording every student question.

This is the data source for the faculty-facing features. Each row stores
the question, whether the notes covered it, how close the best match was,
and which slides/pages were retrieved. Questions are anonymous by design:
no student identifier is stored.

A question the notes could not answer is a "gap". A professor can resolve
a gap by writing an answer, which is then added to the index (see
`vectorstore.add_faculty_answer`) so future students get it.
"""

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_core.documents import Document

from rag_core.config import DB_DIR, QUERY_LOG_FILENAME

STATUS_ANSWERED = "answered"
STATUS_GAP = "gap"
STATUS_RESOLVED = "resolved"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    asked_at          TEXT NOT NULL,
    question          TEXT NOT NULL,
    grounded          INTEGER NOT NULL,
    top_score         REAL,
    retrieved         TEXT NOT NULL,
    scope             TEXT NOT NULL,
    status            TEXT NOT NULL,
    faculty_answer    TEXT,
    resolved_question TEXT,
    resolved_at       TEXT,
    student_id        TEXT
)
"""


def _connect(db_dir: str | Path = DB_DIR) -> sqlite3.Connection:
    conn = sqlite3.connect(Path(db_dir) / QUERY_LOG_FILENAME)
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(queries)")}
    if "student_id" not in columns:
        conn.execute("ALTER TABLE queries ADD COLUMN student_id TEXT")
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    data = dict(row)
    data["grounded"] = bool(data["grounded"])
    data["retrieved"] = json.loads(data["retrieved"])
    data["scope"] = json.loads(data["scope"])
    return data


def _describe_chunk(doc: Document) -> Dict[str, Any]:
    return {
        "source": Path(doc.metadata.get("source", "unknown")).name,
        "page": doc.metadata.get("page"),
        "section": doc.metadata.get("section"),
    }


def log_query(
    question: str,
    grounded: bool,
    top_score: Optional[float],
    source_documents: Iterable[Document],
    scope: Optional[Iterable[str]] = None,
    db_dir: str | Path = DB_DIR,
    student_id: Optional[str] = None,
) -> int:
    """Record one question and its outcome. Returns the new row id."""
    retrieved = [_describe_chunk(d) for d in source_documents]
    status = STATUS_ANSWERED if grounded else STATUS_GAP
    with closing(_connect(db_dir)) as conn, conn:
        cursor = conn.execute(
            "INSERT INTO queries (asked_at, question, grounded, top_score, "
            "retrieved, scope, status, student_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _now(),
                question,
                int(grounded),
                top_score,
                json.dumps(retrieved),
                json.dumps(sorted(scope) if scope else []),
                status,
                student_id,
            ),
        )
        return int(cursor.lastrowid)


def list_queries(
    status: Optional[str] = None,
    db_dir: str | Path = DB_DIR,
    student_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """All logged questions, oldest first, optionally filtered by status or student."""
    clauses: List[str] = []
    params: List[Any] = []
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    if student_id is not None:
        clauses.append("student_id = ?")
        params.append(student_id)
    sql = "SELECT * FROM queries"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY id"
    with closing(_connect(db_dir)) as conn:
        return [_row_to_dict(r) for r in conn.execute(sql, tuple(params)).fetchall()]


def delete_by_scope_marker(marker: str, db_dir: str | Path = DB_DIR) -> int:
    """Delete rows whose scope contains `marker` (used to clear simulated demo data)."""
    with closing(_connect(db_dir)) as conn, conn:
        cursor = conn.execute(
            "DELETE FROM queries WHERE scope LIKE ?", (f'%"{marker}"%',)
        )
        return cursor.rowcount


def resolve_gaps(
    ids: Iterable[int],
    representative_question: str,
    answer: str,
    db_dir: str | Path = DB_DIR,
) -> None:
    """Mark gap rows as resolved and store the professor's answer on them."""
    ids = list(ids)
    if not ids:
        return
    placeholders = ",".join("?" for _ in ids)
    with closing(_connect(db_dir)) as conn, conn:
        conn.execute(
            f"UPDATE queries SET status = ?, faculty_answer = ?, "
            f"resolved_question = ?, resolved_at = ? "
            f"WHERE status = ? AND id IN ({placeholders})",
            (STATUS_RESOLVED, answer, representative_question, _now(), STATUS_GAP, *ids),
        )


def resolved_answers(db_dir: str | Path = DB_DIR) -> List[Tuple[str, str]]:
    """Distinct (question, answer) pairs a professor has written so far."""
    with closing(_connect(db_dir)) as conn:
        rows = conn.execute(
            "SELECT DISTINCT resolved_question, faculty_answer FROM queries "
            "WHERE status = ? ORDER BY resolved_at",
            (STATUS_RESOLVED,),
        ).fetchall()
    return [(r["resolved_question"], r["faculty_answer"]) for r in rows]
