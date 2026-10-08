"""
Tests for the Content Advisor: the rule-based diagnosis, the advice log, clarifications in the index, the
draft clarification (with a fake chat model) and the professor hand-offs. The key promises checked here:

  - every finding comes from evidence and says which;
  - nothing drafted reaches students before approval;
  - an approved clarification never moves students' questions to a different concept.

Run:  python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from agents import ContentAdvisor
from rag_core.advice import Advice, diagnose
from rag_core.advice_log import (
    KIND_CLARIFICATION, KIND_REMEDIAL_QUIZ, STATUS_APPROVED, STATUS_DISMISSED, STATUS_DRAFT,
    add_advice, approved_clarifications, get_advice, list_advice, set_status, update_body,
)
from rag_core.confusion import CONCEPT_EXPLAINER, compute_confusion, concept_key
from rag_core.embeddings import get_embeddings
from rag_core.insights import topic_title
from rag_core.learning_log import (
    SCOPE_CLASS, SCOPE_DRAFT, delete_draft_quiz, get_quiz, list_quizzes, set_quiz_scope,
)
from rag_core.normalize import human_title
from rag_core.oral_assessment import LEVEL_DEVELOPING
from rag_core.query_log import log_query
from rag_core.vectorstore import (
    add_clarification, build_vectorstore, clarification_documents, faculty_documents, remove_clarification,
)
from test_agents import AgentTestCase, CountingFake, fake, quiz_json
from test_confusion import Builder, students
from test_item_analysis import Attempts, make_quiz

HARD = "Hard Topic"
EARLIER = "Earlier Topic"


def advise(b: Builder, **kw):
    report = b.report()
    return {a.concept: a for a in diagnose(report, b.rows, b.quizzes, b.attempts, **kw)}, report


def kinds(advice: Advice):
    return [f.kind for f in advice.findings]


def quiz_with_questions(b: Builder, concept: str, n_questions: int, picks_for):
    """A class quiz for `concept` with real questions; picks_for(student_index, question_index) -> option picked."""
    quiz = make_quiz(len(b.quizzes) + 1, n_questions, topic=concept)
    quiz["scope"] = "class"
    b.quizzes.append(quiz)
    log = Attempts()
    log.rows = b.attempts
    for k in range(24):
        log.add(f"s{k}", quiz, [picks_for(k, i) for i in range(n_questions)])
    return quiz


class DiagnosisTests(unittest.TestCase):
    def hard_topic_with_one_shared_mistake(self):
        b = Builder()
        for s in students(24):
            b.ask(s, HARD, day=1, question="What does the term mean?")
        quiz_with_questions(b, HARD, 1, lambda k, i: 2 if k < 17 else 0)       # 17 of 24 chose wrong-b
        return b

    def test_a_shared_wrong_answer_becomes_a_misconception_finding_that_names_the_evidence(self):
        advice, _ = advise(self.hard_topic_with_one_shared_mistake())
        a = advice[HARD]
        self.assertIn("misconception", kinds(a))
        finding = next(f for f in a.findings if f.kind == "misconception")
        self.assertIn("71%", finding.detail)
        self.assertIn("wrong-b", finding.detail)
        self.assertIn("right", finding.detail)
        self.assertEqual((a.misconceptions[0]["wrong"], a.misconceptions[0]["right"]), ("wrong-b", "right"))

    def test_findings_are_ordered_most_urgent_first(self):
        b = self.hard_topic_with_one_shared_mistake()
        advice, _ = advise(b)
        priorities = [f.priority for f in advice[HARD].findings]
        self.assertEqual(priorities, sorted(priorities))

    def test_a_question_that_strong_students_miss_is_flagged_for_checking_first(self):
        b = Builder()
        for s in students(24):
            b.ask(s, HARD, day=1)
        # Two 4-question quizzes. Ability rises with the student number; question 4 of quiz 2 has a reversed key.
        for quiz_no in range(2):
            def picks(k, i, quiz_no=quiz_no):
                ability = k / 24
                right = ability >= 0.2 + 0.15 * i + 0.1 * quiz_no
                if quiz_no == 1 and i == 3:
                    right = ability < 0.5
                return 0 if right else 1
            quiz_with_questions(b, HARD, 4, picks)
        advice, _ = advise(b)
        a = advice[HARD]
        self.assertEqual(kinds(a)[0], "check_question")
        self.assertEqual(sum(k == "check_question" for k in kinds(a)), 1)
        self.assertIn("not just a low score", a.findings[0].detail)

    def test_an_earlier_topic_that_is_also_confusing_is_offered_as_a_possible_prerequisite(self):
        b = Builder()
        for i, s in enumerate(students(10)):
            for concept, page in ((EARLIER, 2), (HARD, 7)):
                b.ask(s, concept, day=1, page=page)
                b.ask(s, concept, day=2 + i % 3, page=page, follow_up=True, agent=CONCEPT_EXPLAINER)
        advice, _ = advise(b)
        self.assertIn("prerequisite", kinds(advice[HARD]))
        self.assertIn(EARLIER, next(f for f in advice[HARD].findings if f.kind == "prerequisite").detail)
        self.assertIn("guess", next(f for f in advice[HARD].findings if f.kind == "prerequisite").detail)
        self.assertNotIn("prerequisite", kinds(advice[EARLIER]))          # nothing comes before it

    def test_a_topic_in_a_different_file_is_not_a_prerequisite(self):
        b = Builder()
        for s in students(10):
            b.ask(s, EARLIER, page=2)
            b.ask(s, HARD, page=7)
        for row in b.rows:
            if row["retrieved"][0]["section"] == EARLIER:
                row["retrieved"][0]["source"] = "other.pdf"
        advice, _ = advise(b)
        self.assertNotIn("prerequisite", kinds(advice[HARD]))

    def test_students_returning_with_follow_ups_gives_a_coming_back_finding_with_the_slide(self):
        b = Builder()
        for s in students(10):
            b.ask(s, HARD, day=1, page=4)
            b.ask(s, HARD, day=1, page=4, follow_up=True)
        a, _ = advise(b)
        a = a[HARD]
        self.assertIn("coming_back", kinds(a))
        self.assertIn("notes.pdf", a.landing[0][0])
        self.assertEqual(a.landing[0][1], 20)

    def test_quiz_fine_but_oral_weak_is_reported_as_recognise_not_explain(self):
        b = Builder()
        for s in students(8):
            b.ask(s, HARD, day=1)
            b.ask(s, HARD, day=2)
            b.speak(s, HARD, LEVEL_DEVELOPING)
        b.quiz(HARD, {s: 4 for s in students(8)})
        advice, _ = advise(b)
        self.assertIn("recognise_not_explain", kinds(advice[HARD]))

    def test_questions_phrased_far_from_the_slide_give_a_wording_finding(self):
        b = Builder()
        for s in students(6):
            b.ask(s, HARD, top_score=1.05, question="why does this thing break")
        advice, _ = advise(b)
        self.assertIn("wording", kinds(advice[HARD]))
        self.assertIn("why does this thing break", next(f for f in advice[HARD].findings if f.kind == "wording").detail)

    def test_well_matched_questions_give_no_wording_finding(self):
        b = Builder()
        for s in students(6):
            b.ask(s, HARD, top_score=0.3)
        advice, _ = advise(b)
        self.assertNotIn("wording", kinds(advice[HARD]))

    def test_thin_evidence_is_called_out_and_advises_against_acting(self):
        b = Builder()
        for s in students(6):
            b.ask(s, HARD, day=1)
            b.ask(s, HARD, day=2)
        advice, _ = advise(b)
        thin = next(f for f in advice[HARD].findings if f.kind == "thin_evidence")
        self.assertIn("before changing the material", thin.action)

    def test_a_medium_confidence_topic_with_no_pattern_says_so_instead_of_inventing_a_cause(self):
        b = Builder()
        for s in students(6):
            b.ask(s, HARD, day=1, question=f"q from {s}")
        b.quiz(HARD, {s: 3 for s in students(6)})
        advice, report = advise(b)
        self.assertEqual(report.concepts[0].confidence, "Medium")
        self.assertEqual(kinds(advice[HARD]), ["no_cause"])

    def test_repeated_questions_are_listed_once_with_their_count(self):
        b = Builder()
        for s in students(6):
            b.ask(s, HARD, question="What is it?")
        b.ask("s0", HARD, question="Why?")
        advice, _ = advise(b)
        self.assertEqual(advice[HARD].sub_questions[0], ("What is it?", 6))

    def test_only_the_top_concepts_get_advice(self):
        b = Builder()
        for j in range(4):
            for s in students(6):
                b.ask(s, f"Topic {j}", day=1 + j)
        report = b.report()
        self.assertEqual(len(diagnose(report, b.rows, top_n=2)), 2)
        self.assertEqual(diagnose(report, b.rows, top_n=2)[0].concept, report.concepts[0].concept)

    def test_no_concepts_means_no_advice(self):
        self.assertEqual(diagnose(Builder().report(), []), [])


class AdviceLogTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.db = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_new_items_are_drafts_and_nothing_is_approved_until_decided(self):
        item = add_advice("acid", "ACID", KIND_CLARIFICATION, "Clarify ACID", "ACID means ...", db_dir=self.db)
        row = get_advice(item, self.db)
        self.assertEqual((row["status"], row["decided_at"]), (STATUS_DRAFT, None))
        self.assertEqual(approved_clarifications(self.db), [])

    def test_approval_records_the_time_and_makes_the_text_available_for_rebuilds(self):
        item = add_advice("acid", "ACID", KIND_CLARIFICATION, "Clarify ACID", "first draft", db_dir=self.db)
        update_body(item, "edited by the professor", self.db)
        set_status(item, STATUS_APPROVED, self.db)
        self.assertIsNotNone(get_advice(item, self.db)["decided_at"])
        self.assertEqual(approved_clarifications(self.db), [("ACID", "edited by the professor")])

    def test_dismissed_and_quiz_items_are_not_clarifications(self):
        a = add_advice("a", "A", KIND_CLARIFICATION, "t", "text", db_dir=self.db)
        add_advice("b", "B", KIND_REMEDIAL_QUIZ, "t", ref=7, db_dir=self.db)
        set_status(a, STATUS_DISMISSED, self.db)
        self.assertEqual(approved_clarifications(self.db), [])

    def test_listing_filters(self):
        add_advice("a", "A", KIND_CLARIFICATION, "t", "x", db_dir=self.db)
        add_advice("b", "B", KIND_REMEDIAL_QUIZ, "t", ref=1, db_dir=self.db)
        self.assertEqual(len(list_advice(db_dir=self.db)), 2)
        self.assertEqual([r["concept"] for r in list_advice(kind=KIND_REMEDIAL_QUIZ, db_dir=self.db)], ["B"])
        self.assertEqual(list_advice(concept_key="zzz", db_dir=self.db), [])


class ClarificationIndexTests(unittest.TestCase):
    TITLES = [
        "ACID DATABASE CONSISTENCY MODEL", "OLTP VS OLAP", "Spark Architecture", "Amazon RDS Benefits",
        "Fault Tolerant: If a partition is lost, Spark...", "M U L T I M E D I A S Y S T E M S Introduction",
    ]

    def test_a_clarification_stays_under_the_concept_it_explains(self):
        """If it moved to a new heading, students' later questions would drop out of the concept and its
        confusion score would fall for the wrong reason."""
        for raw in self.TITLES:
            concept = human_title(topic_title({"section": raw}))
            doc = clarification_documents([(concept, "An explanation.")])[0]
            self.assertEqual(concept_key(topic_title(doc.metadata)), concept_key(concept), raw)

    def test_add_find_and_remove(self):
        import tempfile
        docs = [Document(page_content="ACID guarantees atomicity and isolation for transactions.",
                         metadata={"source": "notes.pdf", "page": 1, "section": "ACID Model"})]
        with tempfile.TemporaryDirectory() as tmp:
            store = build_vectorstore(docs, get_embeddings())
            add_clarification(store, "ACID Model", "Atomicity means all or nothing.", tmp)
            found = [d for d in store.docstore._dict.values() if d.metadata.get("kind") == "clarification"]
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0].metadata["section"], "ACID Model")
            self.assertEqual(remove_clarification(store, "ACID Model", "Atomicity means all or nothing.", tmp), 1)
            self.assertEqual(len(store.docstore._dict), 1)
            self.assertEqual(remove_clarification(store, "ACID Model", "never added", tmp), 0)

    def test_a_rebuild_brings_approved_clarifications_back_beside_gap_answers(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            item = add_advice("acid", "ACID Model", KIND_CLARIFICATION, "t", "Atomicity means all or nothing.", db_dir=tmp)
            set_status(item, STATUS_APPROVED, tmp)
            rebuilt = faculty_documents([("A gap question", "A gap answer")]) + clarification_documents(
                approved_clarifications(tmp)
            )
            self.assertEqual(len(rebuilt), 2)
            self.assertTrue(any("Atomicity means all or nothing." in d.page_content for d in rebuilt))


class DraftQuizGateTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from rag_core.learning_log import create_quiz
        self._tmp = tempfile.TemporaryDirectory()
        self.db = self._tmp.name
        self.create = create_quiz

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_draft_is_invisible_to_students_until_published(self):
        quiz_id = self.create("T", "p", [], scope=SCOPE_DRAFT, db_dir=self.db)
        self.assertEqual(list_quizzes(SCOPE_CLASS, self.db), [])
        set_quiz_scope(quiz_id, SCOPE_CLASS, self.db)
        self.assertEqual([q["id"] for q in list_quizzes(SCOPE_CLASS, self.db)], [quiz_id])

    def test_only_drafts_can_be_deleted_this_way(self):
        draft = self.create("T", "p", [], scope=SCOPE_DRAFT, db_dir=self.db)
        published = self.create("T", "p", [], scope=SCOPE_CLASS, db_dir=self.db)
        self.assertTrue(delete_draft_quiz(draft, self.db))
        self.assertFalse(delete_draft_quiz(published, self.db))
        self.assertIsNotNone(get_quiz(published, self.db))

    def test_a_draft_quiz_does_not_count_towards_the_confusion_score(self):
        b = Builder()
        for s in students(6):
            b.ask(s, HARD)
        b.quiz(HARD, {s: 0 for s in students(6)}, scope=SCOPE_DRAFT)
        concept = b.report().concepts[0]
        self.assertIsNone(next(s for s in concept.signals if s.name == "quiz").value)


class ClarificationDraftTests(AgentTestCase):
    ACID = "ACID DATABASE CONSISTENCY MODEL"

    def advice(self, concept=None):
        return Advice(key=concept_key(concept or self.ACID), concept=concept or self.ACID, score=80, confidence="Medium",
                      misconceptions=[{"question": "What is BASE?", "wrong": "Always consistent", "right": "Eventually consistent",
                                       "share": 0.7, "students": 20, "count": 14, "source": {}}],
                      sub_questions=[("What does atomicity mean?", 4)])

    def draft(self, response, concept=None):
        llm = fake(response) if isinstance(response, str) else response
        return ContentAdvisor().draft_clarification(self.advice(concept), self.ctx(llm=llm)), llm

    def test_a_supported_draft_is_returned_and_marked_a_draft_for_review(self):
        result, _ = self.draft(
            "ACID guarantees atomicity, consistency, isolation and durability for database transactions. "
            "BASE relaxes the ACID guarantees to scale."
        )
        self.assertEqual(result.kind, "clarification")
        self.assertIn("atomicity", result.text)
        self.assertEqual(result.data["dropped"], 0)
        self.assertTrue(result.sources)

    def test_a_sentence_the_material_does_not_support_is_dropped(self):
        result, _ = self.draft(
            "ACID guarantees atomicity, consistency, isolation and durability for database transactions. "
            "BASE relaxes the ACID guarantees to scale. "
            "Blockchain ledgers reward miners with cryptocurrency for solving hashing puzzles."
        )
        self.assertNotIn("Blockchain", result.text)
        self.assertEqual(result.data["dropped"], 1)

    def test_a_draft_with_too_little_support_is_refused_not_saved_as_text(self):
        result, _ = self.draft("Blockchain ledgers reward miners with cryptocurrency for solving hashing puzzles. "
                               "Quantum computers factor large integers using superposition and entanglement.")
        self.assertFalse(result.grounded)
        self.assertEqual(result.data["error"], "unsupported")

    def test_citation_markers_and_bullets_are_removed(self):
        result, _ = self.draft(
            "- ACID guarantees atomicity, consistency, isolation and durability for database transactions [Chunk 1]. "
            "BASE relaxes the ACID guarantees to scale [Chunk 2]."
        )
        self.assertNotIn("[Chunk", result.text)
        self.assertFalse(result.text.startswith("-"))

    def test_a_topic_the_notes_do_not_cover_never_calls_the_model(self):
        llm = CountingFake(responses=["unused"])
        bare = Advice(key="k", concept="Kubernetes pod scheduling", score=80, confidence="Medium")
        result = ContentAdvisor().draft_clarification(bare, self.ctx(llm=llm))
        self.assertEqual(result.data["error"], "not_covered")
        self.assertEqual(llm.calls, 0)

    def test_the_confusing_quiz_question_is_searched_too_not_only_the_concept_title(self):
        """A concept the notes cover under another heading is still found through its confusing question."""
        # The concept title is not in the notes, but its confusing quiz question (about BASE) is.
        result, llm = self.draft(CountingFake(responses=[
            "BASE stands for basically available, soft state and eventual consistency. "
            "BASE relaxes the ACID guarantees to scale."
        ]), concept="Kubernetes pod scheduling")
        self.assertEqual(llm.calls, 1)
        self.assertIn("eventual consistency", result.text)

    def test_the_task_names_the_mistake_and_the_correct_answer(self):
        from agents.content_advisor import _task
        task = _task(self.advice())
        self.assertIn("Eventually consistent", task)
        self.assertIn("Always consistent", task)

    def test_chatty_introductions_are_removed(self):
        result, _ = self.draft(
            "Sure, here is a clarification about the ACID model: "
            "ACID guarantees atomicity, consistency, isolation and durability for database transactions. "
            "BASE relaxes the ACID guarantees to scale."
        )
        self.assertTrue(result.text.startswith("ACID guarantees"))
        self.assertEqual(result.data["dropped"], 0)

    def test_an_unavailable_model_gives_an_error_not_invented_text(self):
        class Broken(FakeListChatModel):
            def _call(self, *a, **k):
                raise ConnectionError("ollama is not running")
        result, _ = self.draft(Broken(responses=["x"]))
        self.assertEqual(result.data["error"], "llm_unavailable")


class ProfessorPipelineTests(AgentTestCase):
    def ask_as_class(self):
        """Eight identified students who keep returning to the ACID topic."""
        from agents.orchestrator import Orchestrator  # noqa: F401
        doc = Document(page_content="x", metadata={"source": "notes.pdf", "page": 0, "section": "ACID DATABASE CONSISTENCY MODEL"})
        for i in range(8):
            for minute in (0, 3):
                log_query(f"Explain ACID {i}", True, 0.4, [doc], db_dir=self.db, student_id=f"s{i}", agent="Concept Explainer",
                          session_id=f"sess{i}", asked_at=f"2026-10-0{1 + i % 3}T10:0{minute}:00+00:00")

    def test_with_no_data_the_advisor_says_so(self):
        result = self.orch.advise_class(self.ctx())
        self.assertFalse(result.primary.grounded)
        self.assertEqual(result.primary.data["advice"], [])

    def test_hand_off_trace_shows_insight_then_advisor(self):
        self.ask_as_class()
        result = self.orch.advise_class(self.ctx())
        joined = " | ".join(result.trace)
        self.assertIn("Faculty Insight: scored", joined)
        self.assertIn("Faculty Insight -> Content Advisor", joined)
        advice = result.primary.data["advice"]
        self.assertEqual(advice[0].concept, "ACID Database Consistency MODEL")
        self.assertIn("coming_back", kinds(advice[0]))

    def test_a_remedial_quiz_is_saved_as_a_hidden_draft(self):
        advice = Advice(key="acid", concept="ACID Database Consistency MODEL", score=70, confidence="Medium",
                        misconceptions=[{"question": "What does ACID guarantee?", "wrong": "w", "right": "r",
                                         "share": 0.6, "students": 10, "count": 6, "source": {}}])
        result = self.orch.remedial_quiz(advice, self.ctx(llm=fake(quiz_json(chunk=1, n=2))), n_questions=2)
        self.assertIn("Content Advisor -> Quiz Generator", result.trace[0])
        quiz = get_quiz(result.primary.data["quiz_id"], self.db)
        self.assertEqual(quiz["scope"], SCOPE_DRAFT)
        self.assertEqual(list_quizzes(SCOPE_CLASS, self.db), [])

    def test_the_advisor_is_one_of_the_orchestrators_agents(self):
        self.assertEqual(self.orch.content_advisor.name, "Content Advisor")


if __name__ == "__main__":
    unittest.main()
