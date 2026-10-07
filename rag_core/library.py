"""
Removing a study file from the library: the PDF, its search index entries and everything learned about it.

"Everything" means the student's questions that were answered from it, quizzes written from it (and the
scores on them) and oral checks on its headings, so no progress for it is left behind.
"""

import json
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Dict

from rag_core.config import DATA_DIR, DB_DIR, FAISS_INDEX_NAME, QUERY_LOG_FILENAME
from rag_core.normalize import human_title
from rag_core.vectorstore import save_vectorstore


def _file_of(path: str) -> str:
    return Path(str(path or "")).name


def delete_material(filename: str, vectorstore, db_dir: str | Path = DB_DIR, data_dir: str | Path = DATA_DIR) -> Dict[str, Any]:
    """Delete `filename` from the library. Returns how much was removed."""
    from agents.base import AgentContext          # local import: rag_core must not depend on agents at import time
    from agents.note_generator import group_material

    summary = {"pdf": False, "chunks": 0, "questions": 0, "quizzes": 0, "oral_checks": 0}

    # 1. Oral checks are stored by heading. Keep only those that belong to a file that stays in the library;
    #    that also clears checks left over from earlier deletions whose headings no longer match anything.
    keep_headings = set()
    if vectorstore is not None:
        for group in group_material(AgentContext(vectorstore=vectorstore, db_dir=Path(db_dir))):
            if group["file"] != filename and group["title"]:
                keep_headings.add(human_title(group["title"]))

    # 2. Progress records.
    db_file = Path(db_dir) / QUERY_LOG_FILENAME
    if db_file.exists():
        with closing(sqlite3.connect(db_file)) as conn, conn:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "queries" in tables:
                ids = [
                    row[0] for row in conn.execute("SELECT id, retrieved FROM queries")
                    if any(_file_of(chunk.get("source")) == filename for chunk in json.loads(row[1] or "[]"))
                ]
                conn.executemany("DELETE FROM queries WHERE id = ?", [(i,) for i in ids])
                summary["questions"] = len(ids)
            if "quizzes" in tables:
                quiz_ids = [
                    row[0] for row in conn.execute("SELECT id, questions FROM quizzes")
                    if any(_file_of(q.get("source", {}).get("file")) == filename for q in json.loads(row[1] or "[]"))
                ]
                if "attempts" in tables:
                    conn.executemany("DELETE FROM attempts WHERE quiz_id = ?", [(i,) for i in quiz_ids])
                conn.executemany("DELETE FROM quizzes WHERE id = ?", [(i,) for i in quiz_ids])
                summary["quizzes"] = len(quiz_ids)
            if "oral_assessments" in tables and vectorstore is not None:
                marks = ",".join("?" * len(keep_headings)) or "''"
                summary["oral_checks"] = conn.execute(
                    f"DELETE FROM oral_assessments WHERE topic NOT IN ({marks})", tuple(keep_headings)
                ).rowcount

    # 3. Search index: drop this file's chunks (or the whole index if nothing else is left).
    if vectorstore is not None:
        ids = [
            doc_id for doc_id in vectorstore.index_to_docstore_id.values()
            if _file_of(vectorstore.docstore._dict[doc_id].metadata.get("source")) == filename
        ]
        summary["chunks"] = len(ids)
        if ids:
            if len(ids) == len(vectorstore.index_to_docstore_id):
                shutil.rmtree(Path(db_dir) / FAISS_INDEX_NAME, ignore_errors=True)
            else:
                vectorstore.delete(ids)
                save_vectorstore(vectorstore, db_dir)

    # 4. The PDF itself.
    pdf = Path(data_dir) / filename
    if pdf.exists():
        pdf.unlink()
        summary["pdf"] = True
    return summary
