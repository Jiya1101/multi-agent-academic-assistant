"""
Tests for the multi-agent layer.

They use a small synthetic index (real MiniLM embeddings, so retrieval and
the relevance gate behave as in the app) and a fake chat model, so the whole
suite runs in seconds with no Ollama and no PDFs.

Run:  python -m unittest discover -s tests -v
"""

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from agents import AgentContext, Orchestrator, RuleRouter, route_message
from agents.progress_tracker import ProgressTracker
from agents.quiz_generator import grade_answers
from agents.quiz_generator import parse_quiz_json, shuffle_options
from rag_core.embeddings import get_embeddings
from rag_core.learning_log import SCOPE_CLASS, SCOPE_PERSONAL, create_quiz, get_quiz, list_quizzes, record_attempt
from rag_core.query_log import list_queries
from rag_core.vectorstore import build_vectorstore

import random

_DOCS = [
    ("ACID DATABASE CONSISTENCY MODEL",
     "The ACID model guarantees atomicity, consistency, isolation and durability for database transactions."),
    ("BASE DATABASE CONSISTENCY MODEL",
     "BASE stands for basically available, soft state and eventual consistency. BASE relaxes the ACID guarantees to scale."),
    ("AMAZON RDS BENEFITS",
     "Read replicas let Amazon RDS serve read traffic from copies of the database. Multi-AZ keeps a standby copy for failover."),
    ("OLTP VS OLAP SYSTEMS",
     "OLTP systems process many small transactions. OLAP systems run complex analytical queries over large amounts of data."),
]


def quiz_json(chunk: int = 1, n: int = 1) -> str:
    return json.dumps({"questions": [
        {"question": f"Question {i}?", "options": ["alpha", "beta", "gamma", "delta"],
         "answer_index": 0, "chunk": chunk, "explanation": "Because the notes say so."}
        for i in range(n)
    ]})


class CountingFake(FakeListChatModel):
    """Fake chat model that counts calls. (FakeListChatModel.i wraps back to 0 after its
    last response, so it cannot be used to prove a model was never called.)"""

    calls: int = 0

    def _call(self, *args, **kwargs):
        self.calls += 1
        return super()._call(*args, **kwargs)


def fake(*responses: str) -> CountingFake:
    return CountingFake(responses=list(responses))


class AgentTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        docs = [
            Document(page_content=text, metadata={"source": "notes.pdf", "page": i, "section": title})
            for i, (title, text) in enumerate(_DOCS)
        ]
        cls.vectorstore = build_vectorstore(docs, get_embeddings())

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name)
        self.orch = Orchestrator(router=RuleRouter())  # agent tests should not depend on the ML router

    def tearDown(self):
        self._tmp.cleanup()

    def ctx(self, llm=None, student_id=None) -> AgentContext:
        return AgentContext(vectorstore=self.vectorstore, db_dir=self.db, student_id=student_id, llm=llm)


class RouterTests(unittest.TestCase):
    def test_routes(self):
        cases = {
            "What is the difference between ACID and BASE?": "doubt",
            "Explain the ACID properties": "doubt",
            "What is a Kubernetes pod?": "doubt",
            "explain ACID simply": "explain",
            "Explain BASE step by step": "explain",
            "give me a detailed explanation of OLAP": "explain",
            "quiz me on RDS read replicas": "quiz",
            "test me": "quiz",
            "give me some practice questions on OLTP": "quiz",
            "how am I doing": "progress",
            "show my weak topics": "progress",
            "what should I study next": "study_next",
        }
        for message, expected in cases.items():
            with self.subTest(message=message):
                self.assertEqual(route_message(message).name, expected)

    def test_quiz_topic_extraction(self):
        self.assertEqual(route_message("quiz me on RDS read replicas").params["topic"], "RDS read replicas")
        self.assertEqual(route_message("Quiz me about ACID?").params["topic"], "ACID")
        self.assertEqual(route_message("quiz me").params["topic"], "")

    def test_explain_level(self):
        self.assertEqual(route_message("explain BASE in detail").params["level"], "detailed")
        self.assertEqual(route_message("explain BASE simply").params["level"], "beginner")


class KeywordRescueTests(AgentTestCase):
    """score_threshold is set near zero so the embedding gate always rejects,
    isolating the keyword rescue."""

    def retrieve(self, question, vectorstore=None, **kwargs):
        from rag_core.chain import retrieve
        return retrieve(question, vectorstore or self.vectorstore, score_threshold=0.01, **kwargs)

    def test_content_terms(self):
        from rag_core.chain import _content_terms
        self.assertEqual(_content_terms("What is ACID?"), ["acid"])
        self.assertEqual(_content_terms("quiz me on RDS read replicas"), ["rds", "read", "replicas"])
        self.assertEqual(_content_terms("When is assignment 2 due"), ["assignment", "due"])

    def test_rescues_short_keyword_query_the_distance_gate_rejects(self):
        result = self.retrieve("ACID")
        self.assertTrue(result.relevant)
        self.assertEqual(result.via, "keyword")
        self.assertTrue(all("acid" in c.page_content.lower() for c in result.chunks))

    def test_absent_words_are_not_rescued(self):
        self.assertFalse(self.retrieve("Kubernetes pod").relevant)

    def test_every_term_must_appear(self):
        self.assertFalse(self.retrieve("ACID kubernetes").relevant)

    def test_long_queries_are_left_to_the_embedding_gate(self):
        self.assertFalse(self.retrieve("tell me everything about acid transactions atomicity isolation").relevant)

    def test_respects_the_file_filter(self):
        self.assertFalse(self.retrieve("ACID", source_filenames=["other.pdf"]).relevant)
        self.assertTrue(self.retrieve("ACID", source_filenames=["notes.pdf"]).relevant)

    def test_a_word_in_most_chunks_is_not_distinctive(self):
        docs = [Document(page_content=f"data item {i} about topic {i}", metadata={"source": "big.pdf", "page": i})
                for i in range(30)]
        store = build_vectorstore(docs, get_embeddings())
        self.assertFalse(self.retrieve("data", vectorstore=store).relevant)


class QuizParsingTests(unittest.TestCase):
    def good(self, **overrides):
        item = {"question": "Q?", "options": ["a", "b", "c", "d"], "answer_index": 1, "chunk": 1, "explanation": "e"}
        item.update(overrides)
        return item

    def parse(self, *items, n_chunks=2):
        return parse_quiz_json(json.dumps({"questions": list(items)}), n_chunks)

    def test_accepts_valid(self):
        self.assertEqual(len(self.parse(self.good())), 1)

    def test_accepts_bare_list(self):
        self.assertEqual(len(parse_quiz_json(json.dumps([self.good()]), 2)), 1)

    def test_rejects_malformed_json_and_wrong_shapes(self):
        for raw in ["not json", "", "null", json.dumps({"questions": "x"}), json.dumps(42)]:
            with self.subTest(raw=raw):
                self.assertEqual(parse_quiz_json(raw, 2), [])

    def test_rejects_bad_questions(self):
        bad = [
            self.good(question=""),
            self.good(options=["a", "b", "c"]),
            self.good(options=["a", "a", "c", "d"]),
            self.good(options=["a", "b", "c", ""]),
            self.good(answer_index=4),
            self.good(answer_index=-1),
            self.good(answer_index=True),
            self.good(answer_index="1"),
            self.good(chunk=0),
            self.good(chunk=3),
            self.good(chunk=None),
        ]
        for item in bad:
            with self.subTest(item=item):
                self.assertEqual(self.parse(item), [])

    def test_keeps_good_drops_bad_in_same_batch(self):
        self.assertEqual(len(self.parse(self.good(), self.good(answer_index=9), self.good())), 2)

    def test_shuffle_keeps_correct_answer(self):
        question = self.good(options=["right", "w1", "w2", "w3"], answer_index=0)
        for seed in range(25):
            shuffled = shuffle_options(question, random.Random(seed))
            self.assertEqual(shuffled["options"][shuffled["answer_index"]], "right")
            self.assertCountEqual(shuffled["options"], question["options"])

    def test_grade_answers(self):
        questions = [{"answer_index": 0}, {"answer_index": 2}, {"answer_index": 3}]
        graded = grade_answers(questions, [0, 1, None])
        self.assertEqual((graded["correct"], graded["total"]), (1, 3))
        self.assertEqual(graded["per_question"], [True, False, False])


class DoubtAndGapTests(AgentTestCase):
    def test_grounded_question_is_answered_and_logged_with_student(self):
        result = self.orch.handle("What is ACID?", self.ctx(fake("ACID means ... [Chunk 1]"), "s1"))
        self.assertEqual(result.route.name, "doubt")
        self.assertTrue(result.primary.grounded)
        rows = list_queries(db_dir=self.db)
        self.assertEqual([(r["status"], r["student_id"]) for r in rows], [("answered", "s1")])
        self.assertEqual(len(result.results), 1)  # no Gap Handler hand-off

    def test_off_topic_question_hands_off_to_gap_handler_without_calling_llm(self):
        llm = fake("SHOULD NOT BE USED")
        result = self.orch.handle("What is a Kubernetes pod?", self.ctx(llm))
        self.assertFalse(result.primary.grounded)
        self.assertEqual([r.agent for r in result.results], ["Doubt Resolver", "Gap Handler"])
        self.assertTrue(any("Doubt Resolver -> Gap Handler" in line for line in result.trace))
        self.assertEqual(list_queries(db_dir=self.db)[0]["status"], "gap")
        self.assertEqual(llm.calls, 0, "relevance gate must stop the LLM from being called")

    def test_gap_handler_counts_similar_open_gaps(self):
        self.orch.handle("What is a Kubernetes pod?", self.ctx())
        second = self.orch.handle("Explain pods in Kubernetes", self.ctx())
        self.assertEqual(second.results[-1].data["similar_open_gaps"], 1)
        self.assertIn("1 other", second.results[-1].text)

    def test_explainer_not_covered_also_reaches_gap_handler(self):
        result = self.orch.handle("explain Kubernetes pods simply", self.ctx())
        self.assertEqual(result.route.name, "explain")
        self.assertEqual(result.results[-1].agent, "Gap Handler")


class ExplainerTests(AgentTestCase):
    def test_explanation_uses_level_and_is_logged(self):
        result = self.orch.handle("explain BASE simply", self.ctx(fake("BASE is ... [Chunk 1]"), "s1"))
        self.assertEqual(result.primary.kind, "explanation")
        self.assertEqual(result.primary.data["level"], "beginner")
        self.assertEqual(len(list_queries(db_dir=self.db)), 1)


class QuizGeneratorTests(AgentTestCase):
    def test_quiz_is_saved_with_source_slide(self):
        result = self.orch.quiz_generator.run("What is ACID?", self.ctx(fake(quiz_json(chunk=1, n=2))), n_questions=2)
        self.assertEqual(result.kind, "quiz")
        quiz = get_quiz(result.data["quiz_id"], self.db)
        self.assertEqual(len(quiz["questions"]), 2)
        self.assertEqual(quiz["scope"], SCOPE_PERSONAL)
        for question in quiz["questions"]:
            self.assertEqual(question["source"]["file"], "notes.pdf")
            self.assertIn("DATABASE", question["source"]["slide"] + "RDS OLTP")
            self.assertEqual(question["options"][question["answer_index"]], "alpha")

    def test_retries_once_on_invalid_output(self):
        llm = fake("not json at all", quiz_json())
        result = self.orch.quiz_generator.run("What is ACID?", self.ctx(llm))
        self.assertNotIn("error", result.data)
        self.assertEqual(llm.calls, 2)

    def test_gives_up_after_two_invalid_outputs_and_saves_nothing(self):
        result = self.orch.quiz_generator.run("What is ACID?", self.ctx(fake("bad", "worse")))
        self.assertEqual(result.data["error"], "invalid_output")
        self.assertEqual(list_quizzes(db_dir=self.db), [])

    def test_refuses_topic_not_in_notes_without_calling_llm(self):
        llm = fake(quiz_json())
        result = self.orch.quiz_generator.run("Kubernetes pods", self.ctx(llm))
        self.assertEqual(result.data["error"], "not_covered")
        self.assertEqual(llm.calls, 0)


class ProgressTests(AgentTestCase):
    def quiz(self, topic):
        quiz_id = create_quiz(topic, "p", [{"answer_index": 0}] * 5, SCOPE_CLASS, db_dir=self.db)
        return {"id": quiz_id, "topic": topic, "questions": [{"answer_index": 0}] * 5}

    def test_anonymous_student_gets_prompt_not_data(self):
        result = ProgressTracker().run("", self.ctx())
        self.assertEqual(result.data["topics"], [])
        self.assertIn("Student ID", result.text)

    def test_status_from_quiz_scores(self):
        tracker = ProgressTracker()
        ctx = self.ctx(student_id="s1")
        tracker.record_attempt(self.quiz("A"), [0, 0, 0, 0, 0], ctx)   # 5/5
        tracker.record_attempt(self.quiz("B"), [0, 0, 0, 1, 1], ctx)   # 3/5
        tracker.record_attempt(self.quiz("C"), [1, 1, 1, 1, 0], ctx)   # 1/5
        status = {t.topic: t.status for t in tracker.run("", ctx).data["topics"]}
        self.assertEqual(status, {"A": "Strong", "B": "Developing", "C": "Weak"})
        self.assertEqual(tracker.run("", ctx).data["recommended"].topic, "C")

    def test_attempts_are_per_student(self):
        tracker = ProgressTracker()
        tracker.record_attempt(self.quiz("A"), [1, 1, 1, 1, 1], self.ctx(student_id="s1"))
        self.assertEqual(tracker.run("", self.ctx(student_id="s2")).data["topics"], [])

    def test_anonymous_attempt_is_graded_but_not_saved(self):
        graded = ProgressTracker().record_attempt(self.quiz("A"), [0, 0, 1, 1, 1], self.ctx())
        self.assertEqual((graded["correct"], graded["saved"]), (2, False))

    def test_repeated_questions_mark_topic_as_needs_practice(self):
        ctx = self.ctx(fake("a", "a", "a"), "s1")
        for _ in range(3):
            self.orch.handle("What is ACID?", ctx)
        topics = ProgressTracker().run("", ctx).data["topics"]
        self.assertEqual((topics[0].status, topics[0].questions), ("Needs practice", 3))


class HandoffTests(AgentTestCase):
    def test_quiz_me_without_topic_uses_weakest_topic(self):
        ctx = self.ctx(fake("a", "a"), "s1")
        for _ in range(2):
            self.orch.handle("What is ACID?", ctx)
        result = self.orch.handle("quiz me", self.ctx(fake(quiz_json()), "s1"))
        self.assertTrue(any("Progress Tracker -> Quiz Generator" in line for line in result.trace))
        self.assertEqual(result.primary.kind, "quiz")

    def test_quiz_me_with_nothing_to_go_on_asks_for_topic(self):
        result = self.orch.handle("quiz me", self.ctx(student_id="new"))
        self.assertEqual(result.primary.kind, "clarify")

    def test_study_next_chains_progress_to_explainer_without_logging_a_question(self):
        ctx = self.ctx(fake("a", "a"), "s1")
        for _ in range(2):
            self.orch.handle("What is ACID?", ctx)
        before = len(list_queries(db_dir=self.db))
        result = self.orch.handle("what should I study next", self.ctx(fake("Explanation [Chunk 1]"), "s1"))
        self.assertEqual([r.agent for r in result.results], ["Progress Tracker", "Concept Explainer"])
        self.assertEqual(len(list_queries(db_dir=self.db)), before, "system-generated request must not pollute the log")

    def test_professor_pipeline_faculty_insight_to_quiz_generator(self):
        for question in ["What is ACID?", "Explain the ACID properties", "ACID guarantees atomicity?"]:
            self.orch.handle(question, self.ctx(fake("answer")))
        result = self.orch.publish_class_quizzes(self.ctx(fake(quiz_json())), top_n=3, n_questions=1)

        topics = result.results[0].data["confusion_topics"]
        self.assertTrue(topics, "three ACID questions should form a confusion topic")
        published = list_quizzes(SCOPE_CLASS, self.db)
        self.assertEqual([q["topic"] for q in published], [t.topic for t in topics][: len(published)])
        self.assertTrue(any("Faculty Insight -> Quiz Generator" in line for line in result.trace))

        again = self.orch.publish_class_quizzes(self.ctx(fake(quiz_json())), top_n=3, n_questions=1)
        self.assertEqual(len(list_quizzes(SCOPE_CLASS, self.db)), len(published), "must not republish")
        self.assertTrue(any("already published" in line for line in again.trace))

    def test_unanswered_topics_are_not_turned_into_quizzes(self):
        for question in ["What is a Kubernetes pod?", "Explain pods in Kubernetes"]:
            self.orch.handle(question, self.ctx())
        result = self.orch.publish_class_quizzes(self.ctx(fake(quiz_json())))
        self.assertEqual(result.results[0].data["confusion_topics"], [])
        self.assertEqual(list_quizzes(db_dir=self.db), [])


class StorageTests(unittest.TestCase):
    def test_old_database_without_student_id_is_migrated(self):
        with tempfile.TemporaryDirectory() as tmp:
            from rag_core.config import QUERY_LOG_FILENAME
            conn = sqlite3.connect(Path(tmp) / QUERY_LOG_FILENAME)
            conn.execute(
                "CREATE TABLE queries (id INTEGER PRIMARY KEY AUTOINCREMENT, asked_at TEXT NOT NULL, "
                "question TEXT NOT NULL, grounded INTEGER NOT NULL, top_score REAL, retrieved TEXT NOT NULL, "
                "scope TEXT NOT NULL, status TEXT NOT NULL, faculty_answer TEXT, resolved_question TEXT, resolved_at TEXT)"
            )
            conn.execute("INSERT INTO queries (asked_at, question, grounded, retrieved, scope, status) "
                         "VALUES ('t', 'old question', 1, '[]', '[]', 'answered')")
            conn.commit()
            conn.close()
            rows = list_queries(db_dir=tmp)
            self.assertEqual((rows[0]["question"], rows[0]["student_id"]), ("old question", None))

    def test_attempts_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            record_attempt("s1", 1, "T", 2, 3, [0, 1, None], tmp)
            from rag_core.learning_log import list_attempts
            attempt = list_attempts("s1", tmp)[0]
            self.assertEqual((attempt["correct"], attempt["total"], attempt["answers"]), (2, 3, [0, 1, None]))


if __name__ == "__main__":
    unittest.main()
