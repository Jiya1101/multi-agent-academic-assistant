"""
Tests for intervention tracking. Classes are built by hand so the truth is known:

  - a topic that really improved after the action while the others stayed the same;
  - a class where EVERY topic improved (the action must not get the credit);
  - too little data, no yardstick topics, signals that appear on one side only, retakes.

Run:  python -m unittest discover -s tests -v
"""

import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from rag_core.advice_log import (
    KIND_ACTION, KIND_CLARIFICATION, STATUS_APPROVED, add_advice, log_action, set_status,
)
from rag_core.confusion import CONCEPT_EXPLAINER, SIGNALS, concept_key
from rag_core.intervention import (
    Intervention, _resample, interventions_from_log, measure, measure_all, parse_time, window_days,
)
from test_confusion import Builder

UTC = timezone.utc
T = datetime(2026, 10, 8, tzinfo=UTC)              # when the action happened
AS_OF = datetime(2026, 10, 15, tzinfo=UTC)
TREATED = "Treated Topic"
OTHERS = ["Topic A", "Topic B", "Topic C"]
EXPLAINER = CONCEPT_EXPLAINER


class World(Builder):
    """A class over two weeks: days 1-7 are before the action, days 8-14 after."""

    def quiz_for(self, concept):
        found = next((q for q in self.quizzes if q["topic"] == concept), None)
        if found:
            return found["id"]
        self.quizzes.append({"id": len(self.quizzes) + 1, "topic": concept, "scope": "class", "questions": []})
        return len(self.quizzes)

    def attempt(self, concept, student, correct, day, quiz_id=None):
        self.attempts.append({
            "id": len(self.attempts) + 1, "student_id": student, "quiz_id": quiz_id or self.quiz_for(concept),
            "topic": concept, "correct": correct, "total": 4, "answers": [],
            "taken_at": f"2026-10-{day:02d}T12:00:00+00:00",
        })


def intervention(concept=TREATED, at=T, id_=1, kind=KIND_ACTION):
    return Intervention(id_, concept_key(concept), concept, kind, "re-taught", at)


def build(better, n=24, concepts=(TREATED, *OTHERS), quiz=True, flat_low=()):
    """
    `better` is the set of topics that improve after the action (fewer follow-ups and explanations, better quiz).
    `flat_low` topics are low in confusion the whole time (one plain question, a good quiz, in both windows).
    """
    w = World()
    for k in range(n):
        student = f"s{k}"
        for concept in flat_low:
            d0, d1 = 1 + k % 6, 8 + k % 6
            w.ask(student, concept, day=d0)
            w.ask(student, concept, day=d1)
            if quiz:
                w.attempt(concept, student, 3, d0 if k < n // 2 else d1)
        for concept in concepts:
            d0, d1 = 1 + k % 6, 8 + k % 6
            w.ask(student, concept, day=d0, agent=EXPLAINER)
            w.ask(student, concept, day=d0, follow_up=True)
            if concept in better:
                w.ask(student, concept, day=d1)
            else:
                w.ask(student, concept, day=d1, agent=EXPLAINER)
                w.ask(student, concept, day=d1, follow_up=True)
            if quiz:                                                    # first half take it before, second half after
                if k < n // 2:
                    w.attempt(concept, student, 1, d0)
                else:
                    w.attempt(concept, student, 3 if concept in better else 1, d1)
    return w


def run(w, iv=None, as_of=AS_OF, draws=60, **kw):
    return measure(iv or intervention(), w.rows, w.quizzes, w.attempts, w.oral, [iv or intervention()], as_of,
                   draws=draws, **kw)


class MeasuringAChangeTests(unittest.TestCase):
    def test_a_real_improvement_is_found_and_reported_relative_to_other_topics(self):
        effect = run(build(better={TREATED}))
        self.assertEqual(effect.status, "measured")
        self.assertLess(effect.change, -10)
        self.assertGreater(effect.control_change, -3)
        self.assertEqual(effect.n_controls, 3)
        self.assertLess(effect.interval[1], 0)
        self.assertEqual(effect.verdict, "Fell more than other topics")
        self.assertIn("does not prove", effect.message)

    def test_a_change_that_every_topic_shares_is_not_credited_to_the_action(self):
        effect = run(build(better={TREATED, *OTHERS}))
        self.assertLess(effect.change, -10)                    # the topic did get better ...
        self.assertLess(effect.control_change, -10)            # ... but so did all the others
        self.assertLess(abs(effect.relative), 5)
        self.assertEqual(effect.verdict, "No clear difference from other topics")

    def test_nothing_changing_anywhere_is_no_clear_difference(self):
        effect = run(build(better=set()))
        self.assertLess(abs(effect.change), 3)
        self.assertEqual(effect.verdict, "No clear difference from other topics")

    def test_a_small_difference_that_chance_could_produce_is_not_called_a_change(self):
        """One of 24 students stops following up on the treated topic only. The point estimate is below zero, but
        in about a third of the re-checks that student is not drawn at all, so the range reaches zero and the
        verdict follows the range, not the point estimate."""
        w = build(better=set())
        gone = {"s0"}
        w.rows = [r for r in w.rows
                  if not (r["student_id"] in gone and r["follow_up_of"] and parse_time(r["asked_at"]) >= T
                          and r["retrieved"][0]["section"] == TREATED)]
        effect = run(w)
        self.assertLess(effect.relative, 0)                      # a real, small difference ...
        self.assertLess(effect.interval[0], 0)
        self.assertGreaterEqual(effect.interval[1], 0)           # ... whose range reaches zero
        self.assertEqual(effect.verdict, "No clear difference from other topics")

    def test_a_topic_that_got_worse_than_the_others_is_reported_as_rising(self):
        effect = run(build(better=set(OTHERS)))               # the others improved, the treated topic did not
        self.assertEqual(effect.verdict, "Rose more than other topics")
        self.assertIn("did not help", effect.message)

    def test_the_range_is_repeatable_for_a_given_seed(self):
        w = build(better={TREATED})
        self.assertEqual(run(w, seed=3).interval, run(w, seed=3).interval)

    def test_the_signals_that_were_compared_are_named(self):
        effect = run(build(better={TREATED}))
        self.assertIn(SIGNALS["quiz"][0], effect.signals_used)
        self.assertEqual([s.label for s in effect.signal_changes], [v[0] for v in SIGNALS.values()])


class RegressionToTheMeanTests(unittest.TestCase):
    """Professors act on the highest-scoring topics. High scores tend to fall back by themselves, low ones do not,
    so comparing with EVERY untouched topic credits the action with that fall."""

    LOWS = ["Low A", "Low B", "Low C"]

    def world(self, peers_better):
        better = {TREATED, *peers_better}
        return build(better=better, concepts=(TREATED, "Peer 1", "Peer 2"), flat_low=self.LOWS)

    def test_only_topics_that_started_at_a_similar_level_are_the_yardstick(self):
        effect = run(self.world(peers_better={"Peer 1", "Peer 2"}))
        self.assertEqual(effect.n_controls, 2)                    # the two peers, not the three low topics

    def test_a_fall_shared_by_the_high_topics_is_not_credited_to_the_action(self):
        effect = run(self.world(peers_better={"Peer 1", "Peer 2"}))
        self.assertLess(effect.change, -10)                       # the treated topic fell ...
        self.assertLess(effect.control_change, -10)               # ... and so did the equally high untouched ones
        self.assertEqual(effect.verdict, "No clear difference from other topics")

    def test_comparing_with_every_untouched_topic_would_have_credited_the_action(self):
        """The low topics never move, so their median change is about zero: the trap this design avoids."""
        w = self.world(peers_better={"Peer 1", "Peer 2"})
        from rag_core import intervention as module
        original = module.CONTROL_BAND
        module.CONTROL_BAND = 1000.0                              # accept every untouched topic
        try:
            effect = run(w)
        finally:
            module.CONTROL_BAND = original
        self.assertEqual(effect.verdict, "Fell more than other topics")

    def test_one_comparable_topic_is_not_enough(self):
        w = build(better={TREATED, "Peer 1"}, concepts=(TREATED, "Peer 1"), flat_low=self.LOWS)
        effect = run(w)
        self.assertEqual((effect.n_controls, effect.verdict), (1, "Cannot tell"))
        self.assertIn("similar level", effect.message)


class WaitingTests(unittest.TestCase):
    def test_hours_after_the_action_it_is_too_early(self):
        effect = run(build(better={TREATED}), as_of=T + timedelta(hours=3))
        self.assertEqual(effect.status, "waiting")
        self.assertIn("Too early", effect.message)

    def test_an_action_before_any_data_cannot_be_judged(self):
        effect = run(build(better={TREATED}), iv=intervention(at=datetime(2026, 9, 1, tzinfo=UTC)))
        self.assertEqual(effect.status, "waiting")
        self.assertIn("before the action", effect.message)

    def test_too_little_data_after_gives_a_waiting_message_with_counts(self):
        w = World()
        for k in range(6):
            w.ask(f"s{k}", TREATED, day=3)
            w.ask(f"s{k}", TREATED, day=4)
        w.ask("s0", TREATED, day=9)                              # one question after the action
        effect = run(w)
        self.assertEqual(effect.status, "waiting")
        self.assertIn("1 question(s)", effect.message)

    def test_window_length_is_limited_by_the_shorter_side(self):
        w = build(better=set())
        self.assertAlmostEqual(window_days(T, w.rows, w.attempts, w.oral, T + timedelta(days=3)), 3.0)
        self.assertLess(window_days(T, w.rows, w.attempts, w.oral, T + timedelta(days=30)), 7)   # data before is the limit
        self.assertEqual(window_days(T, [], [], [], AS_OF), 0.0)


class YardstickTests(unittest.TestCase):
    def test_a_single_topic_cannot_be_compared_with_anything(self):
        effect = run(build(better={TREATED}, concepts=(TREATED,)))
        self.assertEqual(effect.status, "measured")
        self.assertEqual((effect.n_controls, effect.verdict, effect.interval), (0, "Cannot tell", None))
        self.assertIn("cannot be separated", effect.message)

    def test_a_topic_that_was_itself_acted_on_is_not_used_as_a_yardstick(self):
        w = build(better={TREATED})
        other = Intervention(2, concept_key("Topic A"), "Topic A", KIND_ACTION, "also re-taught", T + timedelta(days=1))
        effect = measure(intervention(), w.rows, w.quizzes, w.attempts, w.oral, [intervention(), other], AS_OF, draws=40)
        self.assertEqual(effect.n_controls, 2)

    def test_too_many_acted_on_topics_leaves_no_yardstick(self):
        w = build(better={TREATED})
        others = [Intervention(10 + i, concept_key(c), c, KIND_ACTION, "x", T) for i, c in enumerate(OTHERS[:2])]
        effect = measure(intervention(), w.rows, w.quizzes, w.attempts, w.oral, [intervention(), *others], AS_OF, draws=40)
        self.assertEqual((effect.n_controls, effect.verdict), (1, "Cannot tell"))


class LikeWithLikeTests(unittest.TestCase):
    def test_a_signal_that_exists_on_one_side_only_is_not_counted_as_a_change(self):
        w = build(better={TREATED}, quiz=False)
        for k in range(12, 24):                                  # a quiz taken only AFTER the action, by some students
            for concept in (TREATED, *OTHERS):
                w.attempt(concept, f"s{k}", 4, 9)
        effect = run(w)
        self.assertEqual(effect.status, "measured")
        self.assertNotIn(SIGNALS["quiz"][0], effect.signals_used)

    def test_a_signal_missing_for_this_topic_but_present_for_others_is_not_filled_in(self):
        """The quiz exists for the other topics on both sides, but for THIS topic only after the action.
        Filling the missing 'before' with the class average would invent a change."""
        w = build(better={TREATED})
        w.attempts = [a for a in w.attempts
                      if not (a["topic"] == TREATED and parse_time(a["taken_at"]) < T)]
        effect = run(w)
        self.assertEqual(effect.status, "measured")
        self.assertNotIn(SIGNALS["quiz"][0], effect.signals_used)

    def test_too_few_signals_in_common_means_no_claim(self):
        w = World()
        for k in range(3):                                       # 3 students: only "days it came up" can be computed
            for day in (2, 3, 4, 9, 10, 11):
                w.ask(f"s{k}", TREATED, day=day, agent=None, session=False)
        effect = run(w, draws=20)
        self.assertEqual(effect.status, "waiting")
        self.assertIn("Not enough evidence", effect.message)

    def test_a_comparison_from_questions_alone_says_so(self):
        effect = run(build(better={TREATED}, quiz=False))
        self.assertEqual(effect.evidence_kinds, ["questions"])
        self.assertIn("Only question behaviour could be compared", effect.message)

    def test_a_comparison_that_includes_the_quiz_does_not_carry_that_warning(self):
        effect = run(build(better={TREATED}))
        self.assertEqual(effect.evidence_kinds, ["questions", "quiz"])
        self.assertNotIn("Only question behaviour", effect.message)

    def test_retaking_an_old_quiz_after_the_action_is_not_new_evidence(self):
        w = build(better=set())
        base = run(w, draws=30)
        for k in range(12):                                      # students who took it BEFORE retake it after, perfectly
            w.attempt(TREATED, f"s{k}", 4, 9)
        retaken = run(w, draws=30)
        self.assertAlmostEqual(base.change, retaken.change, places=6)

    def test_a_different_quiz_on_the_same_topic_after_the_action_does_count(self):
        w = build(better={TREATED}, quiz=False)
        for k in range(24):
            w.attempt(TREATED, f"s{k}", 1, 2, quiz_id=w.quiz_for(TREATED))             # the original quiz, before
            w.attempt(TREATED, f"s{k}", 3, 9, quiz_id=None or self._remedial(w))       # a remedial quiz, after
        for concept in OTHERS:
            for k in range(24):
                w.attempt(concept, f"s{k}", 1, 2)
                w.attempt(concept, f"s{k}", 1, 9, quiz_id=self._remedial_for(w, concept))
        effect = run(w, draws=30)
        self.assertIn(SIGNALS["quiz"][0], effect.signals_used)

    @staticmethod
    def _remedial(w):
        return LikeWithLikeTests._remedial_for(w, TREATED)

    @staticmethod
    def _remedial_for(w, concept):
        existing = [q for q in w.quizzes if q["topic"] == concept]
        if len(existing) > 1:
            return existing[1]["id"]
        w.quizzes.append({"id": len(w.quizzes) + 1, "topic": concept, "scope": "class", "questions": []})
        return len(w.quizzes)


class ResamplingTests(unittest.TestCase):
    def test_every_drawn_student_is_kept_distinct_even_when_drawn_twice(self):
        w = build(better=set(), n=10, concepts=(TREATED,))
        rng = np.random.default_rng(1)
        rows, attempts, _ = _resample(w.rows, w.attempts, [], rng)
        students = {r["student_id"] for r in rows}
        self.assertEqual(len(students), 10)                       # 10 draws -> 10 distinct relabelled students
        self.assertTrue(all("#" in s for s in students))
        self.assertEqual({a["student_id"] for a in attempts} - students, set())

    def test_anonymous_questions_stay_anonymous(self):
        w = World()
        for day in range(1, 5):
            w.ask(None, TREATED, day=day, session=False)
        rows, _, _ = _resample(w.rows, [], [], np.random.default_rng(0))
        self.assertTrue(all(r["student_id"] is None for r in rows))


class LogTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_only_approved_actions_are_interventions(self):
        draft = add_advice("a", "A", KIND_CLARIFICATION, "draft", "text", db_dir=self.db)
        approved = add_advice("b", "B", KIND_CLARIFICATION, "ok", "text", db_dir=self.db)
        set_status(approved, STATUS_APPROVED, self.db)
        log_action("c", "C", "re-taught it in class", "2026-10-05T09:00:00+00:00", self.db)
        found = interventions_from_log(self.db)
        self.assertEqual(sorted(i.concept for i in found), ["B", "C"])
        self.assertNotIn(draft, [i.id for i in found])

    def test_a_logged_action_keeps_the_time_the_professor_gave(self):
        log_action("c", "C", "updated the slide", "2026-10-05T09:00:00+00:00", self.db)
        self.assertEqual(interventions_from_log(self.db)[0].at, datetime(2026, 10, 5, 9, tzinfo=UTC))

    def test_with_nothing_logged_there_is_nothing_to_measure(self):
        self.assertEqual(measure_all(self.db), [])

    def test_times_are_read_leniently(self):
        self.assertEqual(parse_time("2026-10-05T09:00:00"), datetime(2026, 10, 5, 9, tzinfo=UTC))
        self.assertIsNone(parse_time("garbage"))
        self.assertIsNone(parse_time(None))


if __name__ == "__main__":
    unittest.main()
