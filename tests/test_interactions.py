"""
Tests for the interaction log: which agent handled a question, which browsing
session it came from, and whether it was a follow-up to an earlier question.

Run:  python -m unittest discover -s tests -v
"""

import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

from rag_core.config import QUERY_LOG_FILENAME
from rag_core.query_log import interaction_summary, list_queries, log_query
from test_agents import AgentTestCase, fake


def doc(section: str) -> Document:
    return Document(page_content="text", metadata={"source": "notes.pdf", "page": 1, "section": section})


ACID = doc("ACID DATABASE CONSISTENCY MODEL")
RDS = doc("AMAZON RDS BENEFITS")


class FollowUpTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def log(self, question, docs, minute, session="s1", **kw):
        return log_query(
            question, grounded=bool(docs), top_score=0.5, source_documents=docs, db_dir=self.db,
            session_id=session, asked_at=f"2026-10-08T10:{minute:02d}:00+00:00", **kw,
        )

    def follow_up_of(self, row_id):
        return next(r for r in list_queries(db_dir=self.db) if r["id"] == row_id)["follow_up_of"]

    def test_first_question_is_never_a_follow_up(self):
        self.assertIsNone(self.follow_up_of(self.log("What is ACID?", [ACID], 0)))

    def test_same_topic_in_same_session_links_to_the_earlier_question(self):
        first = self.log("What is ACID?", [ACID], 0)
        second = self.log("Give an example of atomicity", [ACID], 5)
        self.assertEqual(self.follow_up_of(second), first)

    def test_links_to_the_most_recent_matching_question(self):
        self.log("What is ACID?", [ACID], 0)
        middle = self.log("And isolation?", [ACID], 4)
        last = self.log("And durability?", [ACID], 8)
        self.assertEqual(self.follow_up_of(last), middle)

    def test_different_topic_is_not_a_follow_up(self):
        self.log("What is ACID?", [ACID], 0)
        self.assertIsNone(self.follow_up_of(self.log("What is a read replica?", [RDS], 3)))

    def test_other_session_is_not_a_follow_up(self):
        self.log("What is ACID?", [ACID], 0, session="s1")
        self.assertIsNone(self.follow_up_of(self.log("What is ACID?", [ACID], 2, session="s2")))

    def test_outside_the_time_window_is_not_a_follow_up(self):
        self.log("What is ACID?", [ACID], 0)
        self.assertIsNone(self.follow_up_of(self.log("What is ACID again?", [ACID], 45)))

    def test_window_is_inclusive_at_the_edge(self):
        first = self.log("What is ACID?", [ACID], 0)
        self.assertEqual(self.follow_up_of(self.log("ACID again", [ACID], 30)), first)

    def test_no_session_means_no_follow_up_detection(self):
        self.log("What is ACID?", [ACID], 0, session=None)
        self.assertIsNone(self.follow_up_of(self.log("What is ACID?", [ACID], 1, session=None)))

    def test_unanswerable_question_is_never_a_follow_up(self):
        self.log("What is ACID?", [ACID], 0)
        self.assertIsNone(self.follow_up_of(self.log("What is a Kubernetes pod?", [], 2)))

    def test_interleaved_sessions_are_kept_apart(self):
        a1 = self.log("What is ACID?", [ACID], 0, session="a")
        b1 = self.log("What is a read replica?", [RDS], 1, session="b")
        a2 = self.log("ACID example?", [ACID], 2, session="a")
        b2 = self.log("Replica lag?", [RDS], 3, session="b")
        self.assertEqual((self.follow_up_of(a2), self.follow_up_of(b2)), (a1, b1))

    def test_agent_and_session_are_stored(self):
        row_id = self.log("What is ACID?", [ACID], 0, agent="Doubt Resolver")
        row = next(r for r in list_queries(db_dir=self.db) if r["id"] == row_id)
        self.assertEqual((row["agent"], row["session_id"]), ("Doubt Resolver", "s1"))

    def test_summary_counts_agents_sessions_and_follow_ups(self):
        self.log("What is ACID?", [ACID], 0, session="a", agent="Doubt Resolver")
        self.log("Explain ACID", [ACID], 2, session="a", agent="Concept Explainer")
        self.log("What is a replica?", [RDS], 3, session="b", agent="Doubt Resolver")
        self.log("old style row", [], 4, session=None)
        summary = interaction_summary(self.db)
        self.assertEqual(summary["questions"], 4)
        self.assertEqual(summary["by_agent"], {"Doubt Resolver": 2, "Concept Explainer": 1, "unknown": 1})
        self.assertEqual((summary["sessions"], summary["untracked"], summary["follow_ups"]), (2, 1, 1))


class MigrationTests(unittest.TestCase):
    def test_database_from_before_this_change_gains_the_new_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = sqlite3.connect(Path(tmp) / QUERY_LOG_FILENAME)
            conn.execute(
                "CREATE TABLE queries (id INTEGER PRIMARY KEY AUTOINCREMENT, asked_at TEXT NOT NULL, "
                "question TEXT NOT NULL, grounded INTEGER NOT NULL, top_score REAL, retrieved TEXT NOT NULL, "
                "scope TEXT NOT NULL, status TEXT NOT NULL, faculty_answer TEXT, resolved_question TEXT, "
                "resolved_at TEXT, student_id TEXT)"
            )
            conn.execute("INSERT INTO queries (asked_at, question, grounded, retrieved, scope, status) "
                         "VALUES ('not a date', 'old question', 1, '[]', '[]', 'answered')")
            conn.commit()
            conn.close()

            old = list_queries(db_dir=tmp)[0]
            self.assertEqual((old["question"], old["agent"], old["session_id"], old["follow_up_of"]),
                             ("old question", None, None, None))
            new_id = log_query("new", True, 0.5, [ACID], db_dir=tmp, agent="Doubt Resolver", session_id="s")
            self.assertEqual(len(list_queries(db_dir=tmp)), 2)
            self.assertTrue(new_id)

    def test_unreadable_timestamp_on_an_earlier_row_does_not_break_logging(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = log_query("a", True, 0.5, [ACID], db_dir=tmp, session_id="s")
            with closing(sqlite3.connect(Path(tmp) / QUERY_LOG_FILENAME)) as conn, conn:
                conn.execute("UPDATE queries SET asked_at = 'garbage' WHERE id = ?", (first,))
            second = log_query("b", True, 0.5, [ACID], db_dir=tmp, session_id="s",
                               asked_at="2026-10-08T10:00:00+00:00")
            row = next(r for r in list_queries(db_dir=tmp) if r["id"] == second)
            self.assertIsNone(row["follow_up_of"])


class AgentLoggingTests(AgentTestCase):
    """The agents pass their name and the session through to the log."""

    def ctx_in_session(self, session="sess-1", llm=None):
        context = self.ctx(llm=llm)
        context.session_id = session
        return context

    def test_doubt_resolver_logs_its_name_and_session(self):
        self.orch.handle("What is the difference between ACID and BASE?", self.ctx_in_session(llm=fake("Answer [Chunk 1].")))
        row = list_queries(db_dir=self.db)[0]
        self.assertEqual((row["agent"], row["session_id"]), ("Doubt Resolver", "sess-1"))

    def test_explanation_requests_are_told_apart_from_questions(self):
        self.orch.handle("What is the difference between ACID and BASE?", self.ctx_in_session(llm=fake("Answer.")))
        self.orch.handle("explain ACID simply", self.ctx_in_session(llm=fake("Explanation.")))
        agents = [r["agent"] for r in list_queries(db_dir=self.db)]
        self.assertEqual(agents, ["Doubt Resolver", "Concept Explainer"])

    def test_second_question_on_the_same_topic_is_marked_as_a_follow_up(self):
        self.orch.handle("What is ACID?", self.ctx_in_session(llm=fake("Answer.")))
        self.orch.handle("Explain the ACID properties", self.ctx_in_session(llm=fake("Answer.")))
        rows = list_queries(db_dir=self.db)
        self.assertEqual(rows[1]["follow_up_of"], rows[0]["id"])

    def test_anonymous_session_without_id_still_gets_agent_but_no_follow_up(self):
        self.orch.handle("What is ACID?", self.ctx(llm=fake("Answer.")))
        self.orch.handle("What is ACID?", self.ctx(llm=fake("Answer.")))
        rows = list_queries(db_dir=self.db)
        self.assertEqual([r["agent"] for r in rows], ["Doubt Resolver", "Doubt Resolver"])
        self.assertEqual([r["follow_up_of"] for r in rows], [None, None])


if __name__ == "__main__":
    unittest.main()
