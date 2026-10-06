"""
Quizzes and quiz attempts, stored in the same SQLite file as the question log.

A quiz belongs to one `topic` (a slide title from the notes), so quiz results
and a student's questions can be compared on the same unit. `scope` is
"class" for quizzes a professor published to everyone, or "personal" for one
a student asked for.
"""

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag_core.config import DB_DIR, QUERY_LOG_FILENAME

SCOPE_CLASS = "class"
SCOPE_PERSONAL = "personal"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS quizzes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    topic      TEXT NOT NULL,
    prompt     TEXT NOT NULL,
    scope      TEXT NOT NULL,
    owner      TEXT,
    questions  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attempts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    taken_at   TEXT NOT NULL,
    student_id TEXT NOT NULL,
    quiz_id    INTEGER NOT NULL,
    topic      TEXT NOT NULL,
    correct    INTEGER NOT NULL,
    total      INTEGER NOT NULL,
    answers    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS route_feedback (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    at        TEXT NOT NULL,
    message   TEXT NOT NULL,
    predicted TEXT NOT NULL,
    correct   TEXT NOT NULL
);
"""


def _connect(db_dir: str | Path = DB_DIR) -> sqlite3.Connection:
    conn = sqlite3.connect(Path(db_dir) / QUERY_LOG_FILENAME)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _quiz_row(row: sqlite3.Row) -> Dict[str, Any]:
    data = dict(row)
    data["questions"] = json.loads(data["questions"])
    return data


def create_quiz(
    topic: str,
    prompt: str,
    questions: List[Dict[str, Any]],
    scope: str = SCOPE_CLASS,
    owner: Optional[str] = None,
    db_dir: str | Path = DB_DIR,
) -> int:
    with closing(_connect(db_dir)) as conn, conn:
        cursor = conn.execute(
            "INSERT INTO quizzes (created_at, topic, prompt, scope, owner, questions) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (_now(), topic, prompt, scope, owner, json.dumps(questions)),
        )
        return int(cursor.lastrowid)


def get_quiz(quiz_id: int, db_dir: str | Path = DB_DIR) -> Optional[Dict[str, Any]]:
    with closing(_connect(db_dir)) as conn:
        row = conn.execute("SELECT * FROM quizzes WHERE id = ?", (quiz_id,)).fetchone()
    return _quiz_row(row) if row else None


def list_quizzes(
    scope: Optional[str] = None, db_dir: str | Path = DB_DIR
) -> List[Dict[str, Any]]:
    sql, params = "SELECT * FROM quizzes", ()
    if scope is not None:
        sql, params = sql + " WHERE scope = ?", (scope,)
    with closing(_connect(db_dir)) as conn:
        rows = conn.execute(sql + " ORDER BY id DESC", params).fetchall()
    return [_quiz_row(r) for r in rows]


def record_attempt(
    student_id: str,
    quiz_id: int,
    topic: str,
    correct: int,
    total: int,
    answers: List[Optional[int]],
    db_dir: str | Path = DB_DIR,
) -> int:
    with closing(_connect(db_dir)) as conn, conn:
        cursor = conn.execute(
            "INSERT INTO attempts (taken_at, student_id, quiz_id, topic, correct, total, answers) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (_now(), student_id, quiz_id, topic, correct, total, json.dumps(answers)),
        )
        return int(cursor.lastrowid)


def list_attempts(
    student_id: Optional[str] = None, db_dir: str | Path = DB_DIR
) -> List[Dict[str, Any]]:
    sql, params = "SELECT * FROM attempts", ()
    if student_id is not None:
        sql, params = sql + " WHERE student_id = ?", (student_id,)
    with closing(_connect(db_dir)) as conn:
        rows = conn.execute(sql + " ORDER BY id", params).fetchall()
    out = []
    for row in rows:
        data = dict(row)
        data["answers"] = json.loads(data["answers"])
        out.append(data)
    return out


def record_route_feedback(
    message: str, predicted: str, correct: str, db_dir: str | Path = DB_DIR
) -> int:
    """Store a correction: the router chose `predicted`, the student wanted `correct`."""
    with closing(_connect(db_dir)) as conn, conn:
        cursor = conn.execute(
            "INSERT INTO route_feedback (at, message, predicted, correct) VALUES (?, ?, ?, ?)",
            (_now(), message, predicted, correct),
        )
        return int(cursor.lastrowid)


def list_route_feedback(db_dir: str | Path = DB_DIR) -> List[Dict[str, Any]]:
    with closing(_connect(db_dir)) as conn:
        rows = conn.execute("SELECT * FROM route_feedback ORDER BY id").fetchall()
    return [dict(r) for r in rows]