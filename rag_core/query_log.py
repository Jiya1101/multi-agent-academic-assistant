"""
Question log: a small SQLite database recording every student question.

This is the data source for the faculty-facing features. Each row stores
the question, whether the notes covered it, how close the best match was,
and which slides/pages were retrieved, which agent handled it, and a random
session token (so a return to the same topic can be recorded as a follow-up).
A student identifier is stored only when the student typed one.

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

from rag_core.config import DB_DIR, FOLLOW_UP_WINDOW_MINUTES, QUERY_LOG_FILENAME

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
    student_id        TEXT,
    agent             TEXT,
    session_id        TEXT,
    follow_up_of      INTEGER
)
"""

# Columns added after the first version; older databases get them on first open.
_ADDED_COLUMNS = {
    "student_id": "TEXT",
    "agent": "TEXT",           # which agent handled it (Doubt Resolver, Concept Explainer, ...)
    "session_id": "TEXT",      # random per-browser-session token, not tied to a person
    "follow_up_of": "INTEGER", # id of the earlier question this one came back to
}


def _connect(db_dir: str | Path = DB_DIR) -> sqlite3.Connection:
    conn = sqlite3.connect(Path(db_dir) / QUERY_LOG_FILENAME)
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(queries)")}
    for name, kind in _ADDED_COLUMNS.items():
        if name not in columns:
            conn.execute(f"ALTER TABLE queries ADD COLUMN {name} {kind}")
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


def _first_topic(retrieved: List[Dict[str, Any]]) -> Optional[str]:
    """The slide topic a question landed on (its best-matching chunk), if any."""
    from rag_core.insights import topic_title  # local import: insights pulls in sklearn

    return topic_title(retrieved[0]) if retrieved else None


def _find_follow_up(
    conn: sqlite3.Connection,
    session_id: Optional[str],
    retrieved: List[Dict[str, Any]],
    asked_at: str,
    window_minutes: float,
) -> Optional[int]:
    """
    The id of the most recent earlier question from the same session that landed
    on the same slide topic within the time window, or None.

    Needs a session id and a retrieved slide: an anonymous or unanswerable
    question has nothing to compare, so it is never counted as a follow-up.
    """
    topic = _first_topic(retrieved)
    if not session_id or topic is None:
        return None
    now = datetime.fromisoformat(asked_at)
    best: Optional[int] = None
    for row in conn.execute(
        "SELECT id, asked_at, retrieved FROM queries WHERE session_id = ? ORDER BY id", (session_id,)
    ):
        try:
            gap = (now - datetime.fromisoformat(row["asked_at"])).total_seconds()
        except (ValueError, TypeError):  # an unreadable timestamp cannot be compared
            continue
        if not 0 <= gap <= window_minutes * 60:
            continue
        if _first_topic(json.loads(row["retrieved"])) == topic:
            best = int(row["id"])  # ORDER BY id, so the last match is the most recent
    return best


def log_query(
    question: str,
    grounded: bool,
    top_score: Optional[float],
    source_documents: Iterable[Document],
    scope: Optional[Iterable[str]] = None,
    db_dir: str | Path = DB_DIR,
    student_id: Optional[str] = None,
    agent: Optional[str] = None,
    session_id: Optional[str] = None,
    asked_at: Optional[str] = None,
    window_minutes: float = FOLLOW_UP_WINDOW_MINUTES,
) -> int:
    """
    Record one question and its outcome. Returns the new row id.

    `agent` says who handled it (so explanation requests can be told apart from
    plain questions). `session_id` groups one browsing session, which lets a
    return to the same topic be recorded as a follow-up even for an anonymous
    student. `asked_at` (ISO, UTC) is only for tests and demo seeding.
    """
    retrieved = [_describe_chunk(d) for d in source_documents]
    status = STATUS_ANSWERED if grounded else STATUS_GAP
    asked_at = asked_at or _now()
    with closing(_connect(db_dir)) as conn, conn:
        follow_up_of = _find_follow_up(conn, session_id, retrieved, asked_at, window_minutes)
        cursor = conn.execute(
            "INSERT INTO queries (asked_at, question, grounded, top_score, "
            "retrieved, scope, status, student_id, agent, session_id, follow_up_of) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                asked_at,
                question,
                int(grounded),
                top_score,
                json.dumps(retrieved),
                json.dumps(sorted(scope) if scope else []),
                status,
                student_id,
                agent,
                session_id,
                follow_up_of,
            ),
        )
        return int(cursor.lastrowid)


def interaction_summary(db_dir: str | Path = DB_DIR) -> Dict[str, Any]:
    """
    How much behaviour data the log holds, for the professor dashboard: totals,
    questions per agent, distinct sessions, and how many were follow-ups.
    Older rows have no agent or session, and are counted as "unknown"/untracked.
    """
    rows = list_queries(db_dir=db_dir)
    by_agent: Dict[str, int] = {}
    for row in rows:
        name = row.get("agent") or "unknown"
        by_agent[name] = by_agent.get(name, 0) + 1
    sessions = {r["session_id"] for r in rows if r.get("session_id")}
    return {
        "questions": len(rows),
        "by_agent": by_agent,
        "sessions": len(sessions),
        "untracked": sum(1 for r in rows if not r.get("session_id")),
        "follow_ups": sum(1 for r in rows if r.get("follow_up_of")),
    }


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
