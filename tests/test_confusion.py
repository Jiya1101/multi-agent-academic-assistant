"""
Tests for the Concept Confusion engine. Classes are built by hand with KNOWN patterns (a topic that is merely
popular, a topic that is really confusing, missing data) and the ranking has to treat them correctly.
No model, index or database is needed.

Run:  python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_core.confusion import (
    CONCEPT_EXPLAINER,
    DEFAULT_WEIGHTS,
    compute_confusion,
    concept_key,
    rank_stability,
)
from rag_core.oral_assessment import LEVEL_DEVELOPING, LEVEL_STRONG
from test_agents import AgentTestCase, fake

POPULAR = "Popular Topic"
CONFUSING = "Confusing Topic"
QUIET = "Quiet Topic"


class Builder:
    """Hand-builds the data a class would leave behind."""

    def __init__(self):
        self.rows, self.quizzes, self.attempts, self.oral = [], [], [], []

    def ask(self, student, concept, day=1, agent="Doubt Resolver", follow_up=False, session=True,
            page=1, top_score=0.5, question=None):
        row_id = len(self.rows) + 1
        self.rows.append({
            "id": row_id, "student_id": student, "session_id": f"sess-{student}-{day}" if session else None,
            "asked_at": f"2026-10-{day:02d}T10:00:00+00:00", "agent": agent,
            "follow_up_of": 1000 + row_id if follow_up else None,   # only "is it set" matters to the engine
            "question": question or f"question {row_id} about {concept}", "top_score": top_score,
            "retrieved": [{"source": "notes.pdf", "page": page, "section": concept}], "grounded": True,
        })

    def gap(self, student):
        self.rows.append({"id": len(self.rows) + 1, "student_id": student, "session_id": None, "agent": "Doubt Resolver",
                          "asked_at": "2026-10-01T10:00:00+00:00", "follow_up_of": None, "retrieved": [], "grounded": False})

    def quiz(self, concept, accuracy_by_student, scope="class"):
        quiz_id = len(self.quizzes) + 1
        self.quizzes.append({"id": quiz_id, "topic": concept, "scope": scope, "questions": []})
        for student, correct in accuracy_by_student.items():
            self.attempts.append({"id": len(self.attempts) + 1, "student_id": student, "quiz_id": quiz_id,
                                  "topic": concept, "correct": correct, "total": 4, "answers": []})
        return quiz_id

    def speak(self, student, concept, level):
        self.oral.append(SimpleNamespace(id=len(self.oral) + 1, student_id=student, topic=concept, understanding_level=level))

    def report(self, **kw):
        return compute_confusion(self.rows, self.quizzes, self.attempts, self.oral, **kw)


def students(n, prefix="s"):
    return [f"{prefix}{i}" for i in range(n)]


def popular_but_not_confusing(b: Builder, n=20):
    """Many students ask once, never need a follow-up or an explanation, and do well on the quiz."""
    for s in students(n):
        b.ask(s, POPULAR)
    b.quiz(POPULAR, {s: 4 for s in students(n)})


def really_confusing(b: Builder, n=8):
    """Fewer students, but they keep coming back, ask to have it explained, on several days, and miss the quiz."""
    for i, s in enumerate(students(n)):
        b.ask(s, CONFUSING, day=1)
        b.ask(s, CONFUSING, day=2 + i % 3, agent=CONCEPT_EXPLAINER, follow_up=True)
    b.quiz(CONFUSING, {s: 1 for s in students(n)})


def get(report, concept):
    return next(c for c in report.concepts if c.concept == concept)


def signal(concept_result, name):
    return next(s for s in concept_result.signals if s.name == name)


class RankingTests(unittest.TestCase):
    def test_a_merely_popular_topic_does_not_outrank_a_confusing_one(self):
        b = Builder()
        popular_but_not_confusing(b, 20)
        really_confusing(b, 8)
        report = b.report()
        self.assertEqual([c.concept for c in report.concepts], [CONFUSING, POPULAR])
        self.assertGreater(get(report, CONFUSING).score, get(report, POPULAR).score + 20)

    def test_counting_questions_alone_would_have_ranked_them_the_wrong_way(self):
        b = Builder()
        popular_but_not_confusing(b, 20)
        really_confusing(b, 8)
        counts = {c.concept: c.questions for c in b.report().concepts}
        self.assertGreater(counts[POPULAR], 0)
        self.assertGreater(counts[POPULAR], counts[CONFUSING] - 1)   # 20 vs 16: volume points at the popular topic

    def test_empty_input_gives_an_empty_report(self):
        report = Builder().report()
        self.assertEqual((report.concepts, report.unscored, report.active_students), ([], [], 0))

    def test_questions_the_notes_could_not_answer_have_no_concept(self):
        b = Builder()
        for s in students(6):
            b.gap(s)
        self.assertEqual(b.report().concepts, [])


class MissingDataTests(unittest.TestCase):
    def test_small_group_hides_signals_about_people(self):
        b = Builder()
        for s in students(4):
            for day in range(1, 4):
                b.ask(s, CONFUSING, day=day)
        c = get(b.report(), CONFUSING)
        self.assertIsNone(signal(c, "breadth").value)
        self.assertIsNone(signal(c, "repeat").value)
        self.assertEqual(signal(c, "breadth").evidence, "")
        self.assertIsNotNone(signal(c, "persistence").value)      # built from questions, not from people

    def test_anonymous_questions_count_as_questions_but_never_as_students(self):
        b = Builder()
        for day in range(1, 8):
            b.ask(None, CONFUSING, day=day, session=False)         # no Student ID and no session
        report = b.report()
        c = get(report, CONFUSING)
        self.assertEqual((c.questions, report.active_students), (7, 0))
        self.assertIsNone(signal(c, "breadth").value)
        self.assertIsNone(signal(c, "repeat").value)
        self.assertIsNotNone(signal(c, "persistence").value)       # days still count

    def test_signal_with_too_few_questions_is_missing(self):
        b = Builder()
        for s in students(2):
            b.ask(s, QUIET)
        report = b.report()
        self.assertEqual(report.concepts, [])
        self.assertEqual(report.unscored, [(QUIET, 2)])

    def test_missing_quiz_is_filled_with_the_class_average_not_with_zero(self):
        b = Builder()
        really_confusing(b, 8)                # has a quiz: miss rate 75 percent
        for i, s in enumerate(students(8, "t")):   # exactly the question pattern of CONFUSING, but no quiz at all
            b.ask(s, QUIET, day=1)
            b.ask(s, QUIET, day=2 + i % 3, agent=CONCEPT_EXPLAINER, follow_up=True)
        report = b.report()
        quiet = get(report, QUIET)
        self.assertIsNone(signal(quiet, "quiz").value)
        self.assertAlmostEqual(report.means["quiz"], 0.75)
        # The only quiz result is the class average, so the topic without one scores exactly like the one with it.
        self.assertAlmostEqual(quiet.score, get(report, CONFUSING).score, delta=0.01)
        # Counting the missing quiz as "nobody missed it" would have given 70 instead.
        self.assertGreater(quiet.score, 85)

    def test_confidence_counts_the_kinds_of_evidence(self):
        b = Builder()
        really_confusing(b, 8)                                  # questions + quiz
        for s in students(6, "t"):
            b.ask(s, QUIET, day=1)
            b.ask(s, QUIET, day=2)
        for s in students(6, "u"):
            b.ask(s, POPULAR, day=1)
            b.ask(s, POPULAR, day=2)
            b.speak(s, POPULAR, LEVEL_DEVELOPING)
        b.quiz(POPULAR, {s: 3 for s in students(6, "u")})
        report = b.report()
        self.assertEqual(get(report, QUIET).confidence, "Low")           # questions only
        self.assertEqual(get(report, CONFUSING).confidence, "Medium")    # questions + quiz
        self.assertEqual(get(report, POPULAR).confidence, "High")        # questions + quiz + oral


class SignalTests(unittest.TestCase):
    def test_quiz_uses_first_attempts_and_class_quizzes_only(self):
        b = Builder()
        for s in students(6):
            b.ask(s, CONFUSING, day=1)
        quiz_id = b.quiz(CONFUSING, {s: 0 for s in students(6)})          # everyone fails first
        for s in students(6):                                              # and passes on the retake
            b.attempts.append({"id": len(b.attempts) + 1, "student_id": s, "quiz_id": quiz_id,
                               "topic": CONFUSING, "correct": 4, "total": 4, "answers": []})
        b.quiz(CONFUSING, {s: 4 for s in students(6)}, scope="personal")   # a personal quiz is ignored
        c = get(b.report(), CONFUSING)
        self.assertAlmostEqual(signal(c, "quiz").value, 1.0)

    def test_quiz_needs_enough_students(self):
        b = Builder()
        for s in students(8):
            b.ask(s, CONFUSING)
        b.quiz(CONFUSING, {s: 0 for s in students(4)})
        self.assertIsNone(signal(get(b.report(), CONFUSING), "quiz").value)

    def test_oral_share_not_yet_strong(self):
        b = Builder()
        for s in students(6):
            b.speak(s, CONFUSING, LEVEL_STRONG if s == "s0" else LEVEL_DEVELOPING)
        c = get(b.report(), CONFUSING)
        self.assertAlmostEqual(signal(c, "oral").value, 5 / 6)
        self.assertIn("5 of 6", signal(c, "oral").evidence)

    def test_oral_check_without_a_student_id_is_not_counted(self):
        b = Builder()
        for _ in range(6):
            b.speak(None, CONFUSING, LEVEL_DEVELOPING)
        self.assertEqual(b.report().concepts, [])

    def test_follow_up_and_explain_rates(self):
        b = Builder()
        for i, s in enumerate(students(10)):
            b.ask(s, CONFUSING, follow_up=i < 5, agent=CONCEPT_EXPLAINER if i < 2 else "Doubt Resolver")
        c = get(b.report(), CONFUSING)
        self.assertAlmostEqual(signal(c, "follow_up").value, 1.0)             # 50 percent follow-ups is the maximum
        self.assertAlmostEqual(signal(c, "explain").value, 0.2 / 0.5)         # 20 percent explain requests
        self.assertIn("5 of 10", signal(c, "follow_up").evidence)

    def test_persistence_counts_different_days(self):
        b = Builder()
        for day in (1, 1, 1, 2, 2, 5):
            b.ask("s0", CONFUSING, day=day)
        self.assertAlmostEqual(signal(get(b.report(), CONFUSING), "persistence").value, 1.0)
        b2 = Builder()
        for _ in range(6):
            b2.ask("s0", CONFUSING, day=3)
        self.assertAlmostEqual(signal(get(b2.report(), CONFUSING), "persistence").value, 1 / 3)

    def test_one_concept_under_different_spellings_is_one_concept(self):
        self.assertEqual(concept_key("ACID DATABASE CONSISTENCY MODEL"), concept_key("ACID Database Consistency MODEL"))
        b = Builder()
        for i, s in enumerate(students(6)):
            b.ask(s, "ACID DATABASE CONSISTENCY MODEL" if i % 2 else "ACID Database Consistency MODEL", day=1 + i % 3)
        self.assertEqual(len(b.report().concepts), 1)

    def test_a_concept_known_only_from_a_quiz_is_still_scored(self):
        b = Builder()
        b.quiz(CONFUSING, {s: 1 for s in students(6)})
        c = get(b.report(), CONFUSING)
        self.assertEqual((c.questions, c.confidence), (0, "Low"))


class WeightTests(unittest.TestCase):
    def build(self):
        b = Builder()
        popular_but_not_confusing(b, 20)
        really_confusing(b, 8)
        return b

    def test_weights_are_normalised_and_bad_weights_fall_back_to_the_defaults(self):
        b = self.build()
        report = b.report(weights={name: 7 * w for name, w in DEFAULT_WEIGHTS.items()})
        self.assertAlmostEqual(sum(report.weights.values()), 1.0)
        self.assertEqual(b.report(weights={name: 0 for name in DEFAULT_WEIGHTS}).weights, DEFAULT_WEIGHTS)

    def test_changing_the_weights_can_change_the_order(self):
        b = Builder()
        for s in students(10):
            b.ask(s, "Comes back often", day=1)
            b.ask(s, "Comes back often", day=1, follow_up=True)       # every student follows up
            b.ask(s, "Fails the quiz", day=1)
            b.ask(s, "Fails the quiz", day=1)
        b.quiz("Comes back often", {s: 3 for s in students(10)})
        b.quiz("Fails the quiz", {s: 1 for s in students(10)})
        only = lambda name: {n: 0.0 for n in DEFAULT_WEIGHTS} | {name: 1.0}
        self.assertEqual(b.report(weights=only("quiz")).concepts[0].concept, "Fails the quiz")
        self.assertEqual(b.report(weights=only("follow_up")).concepts[0].concept, "Comes back often")

    def test_scores_stay_between_zero_and_a_hundred(self):
        for c in self.build().report().concepts:
            self.assertTrue(0 <= c.score <= 100)


class StabilityTests(unittest.TestCase):
    def test_a_clear_winner_stays_on_top_under_any_weighting(self):
        b = Builder()
        popular_but_not_confusing(b, 20)
        really_confusing(b, 8)
        stability = rank_stability(b.report(), k=1, draws=100)
        self.assertEqual(stability[concept_key(CONFUSING)].top_share, 1.0)

    def test_a_close_call_is_reported_as_unstable(self):
        b = Builder()
        for s in students(10):                 # two topics, each strong on a different kind of evidence
            b.ask(s, "Topic A", day=1)
            b.ask(s, "Topic A", day=2, follow_up=True)
            b.ask(s, "Topic B", day=1)
            b.ask(s, "Topic B", day=2, agent=CONCEPT_EXPLAINER)
        b.quiz("Topic A", {s: 2 for s in students(10)})
        b.quiz("Topic B", {s: 2 for s in students(10)})
        stability = rank_stability(b.report(), k=1, draws=200)
        shares = sorted(v.top_share for v in stability.values())
        self.assertLess(shares[0], 1.0)
        self.assertGreater(shares[-1], 0.0)
        self.assertAlmostEqual(sum(shares), 1.0)       # exactly one concept is first in each re-weighting

    def test_empty_report_has_no_stability(self):
        self.assertEqual(rank_stability(Builder().report()), {})


class FacultyInsightAgentTests(AgentTestCase):
    """The Faculty Insight agent exposes the report, and its quiz hand-off is unchanged."""

    def test_agent_returns_the_concept_report(self):
        for question in ["What is ACID?", "Explain the ACID properties", "What is the difference between ACID and BASE?"]:
            self.orch.handle(question, self.ctx(llm=fake("Answer.")))
        result = self.orch.faculty_insight.run("", self.ctx())
        self.assertIn("concept_report", result.data)
        self.assertEqual(len(result.data["confusion_topics"]), 1)          # still the cluster-size rule
        self.assertEqual(result.data["concept_report"].concepts, [])       # one student-less topic: too little evidence

    def test_agent_with_an_empty_log_has_no_report(self):
        result = self.orch.faculty_insight.run("", self.ctx())
        self.assertIsNone(result.data["concept_report"])


if __name__ == "__main__":
    unittest.main()
