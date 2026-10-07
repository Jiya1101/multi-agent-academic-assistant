"""
Persistence schema for oral understanding assessments.

The assessment itself combines two families of evidence:
  - content evidence: how correct/complete the spoken answer was;
  - speech evidence: pause, hesitation and fluency metrics from the audio.

This module only stores and retrieves those results. The next layer can decide
how to generate questions and score answer correctness.
"""

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag_core.audio_metrics import AudioMetrics
from rag_core.config import DB_DIR, QUERY_LOG_FILENAME

LEVEL_STRONG = "Strong understanding"
LEVEL_DEVELOPING = "Developing understanding"
LEVEL_NEEDS_PRACTICE = "Needs practice"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS oral_assessments (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    taken_at            TEXT NOT NULL,
    student_id          TEXT,
    topic               TEXT NOT NULL,
    prompt              TEXT NOT NULL,
    transcript          TEXT NOT NULL,
    content_score       REAL NOT NULL,
    understanding_level TEXT NOT NULL,
    feedback            TEXT NOT NULL,
    metrics             TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class OralAssessment:
    id: int
    taken_at: str
    student_id: Optional[str]
    topic: str
    prompt: str
    transcript: str
    content_score: float
    understanding_level: str
    feedback: str
    metrics: Dict[str, Any]


def _connect(db_dir: str | Path = DB_DIR) -> sqlite3.Connection:
    conn = sqlite3.connect(Path(db_dir) / QUERY_LOG_FILENAME)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def infer_understanding_level(content_score: float, metrics: AudioMetrics | Dict[str, Any]) -> str:
    """
    Conservative rubric for combining answer quality with hesitation signals.

    `content_score` is expected to be 0.0-1.0 from a future semantic evaluator.
    Speech metrics can only lower confidence a little; they should not mark a
    correct but careful speaker as weak by themselves.
    """
    values = metrics.to_dict() if isinstance(metrics, AudioMetrics) else dict(metrics)
    score = max(0.0, min(1.0, float(content_score)))

    hesitation_penalty = 0.0
    if values.get("long_pause_count", 0) >= 3:
        hesitation_penalty += 0.10
    if values.get("pause_ratio", 0.0) >= 0.45:
        hesitation_penalty += 0.10
    if values.get("words_per_minute", 0.0) and values.get("words_per_minute", 0.0) < 45:
        hesitation_penalty += 0.05
    if values.get("filler_rate_per_minute", 0.0) >= 10:
        hesitation_penalty += 0.05

    adjusted = score - hesitation_penalty
    if adjusted >= 0.75:
        return LEVEL_STRONG
    if adjusted >= 0.45:
        return LEVEL_DEVELOPING
    return LEVEL_NEEDS_PRACTICE


def record_oral_assessment(
    topic: str,
    prompt: str,
    transcript: str,
    content_score: float,
    metrics: AudioMetrics | Dict[str, Any],
    feedback: str = "",
    student_id: Optional[str] = None,
    understanding_level: Optional[str] = None,
    db_dir: str | Path = DB_DIR,
) -> int:
    """Store one oral assessment and return its id."""
    metric_values = metrics.to_dict() if isinstance(metrics, AudioMetrics) else dict(metrics)
    level = understanding_level or infer_understanding_level(content_score, metric_values)
    with closing(_connect(db_dir)) as conn, conn:
        cursor = conn.execute(
            "INSERT INTO oral_assessments "
            "(taken_at, student_id, topic, prompt, transcript, content_score, "
            "understanding_level, feedback, metrics) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _now(),
                student_id,
                topic,
                prompt,
                transcript,
                float(content_score),
                level,
                feedback,
                json.dumps(metric_values, sort_keys=True),
            ),
        )
        return int(cursor.lastrowid)


def _row_to_assessment(row: sqlite3.Row) -> OralAssessment:
    data = dict(row)
    data["metrics"] = json.loads(data["metrics"])
    return OralAssessment(**data)


def list_oral_assessments(
    student_id: Optional[str] = None,
    topic: Optional[str] = None,
    db_dir: str | Path = DB_DIR,
) -> List[OralAssessment]:
    """List oral assessments oldest first, optionally filtered by student/topic."""
    clauses: List[str] = []
    params: List[Any] = []
    if student_id is not None:
        clauses.append("student_id = ?")
        params.append(student_id)
    if topic is not None:
        clauses.append("topic = ?")
        params.append(topic)

    sql = "SELECT * FROM oral_assessments"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY id"
    with closing(_connect(db_dir)) as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()
    return [_row_to_assessment(row) for row in rows]
