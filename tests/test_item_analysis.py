"""
Tests for quiz item analysis. The class is built with KNOWN patterns planted in
it (a popular wrong answer, a faulty key, retakes), and the analysis has to find
exactly those. No model, index or database is needed.

Run:  python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_core.item_analysis import (
    FLAG_CHECK_KEY,
    FLAG_OUTVOTED,
    FLAG_TOO_EASY,
    FLAG_UNUSED,
    FLAG_VERY_HARD,
    analyze_quizzes,
    likely_misconceptions,
    questions_to_review,
)


def make_quiz(quiz_id, n_questions=3, topic=None):
    return {
        "id": quiz_id,
        "topic": topic or f"Topic {quiz_id}",
        "questions": [
            {"question": f"Q{i + 1}?", "options": ["right", "wrong-a", "wrong-b", "wrong-c"],
             "answer_index": 0, "source": {"file": "notes.pdf", "page": i + 1, "slide": "S"}}
            for i in range(n_questions)
        ],
    }


class Attempts:
    """Builds attempt rows the way learning_log.list_attempts returns them."""

    def __init__(self):
        self.rows = []

    def add(self, student, quiz, answers):
        correct = sum(1 for q, a in zip(quiz["questions"], answers) if a == q["answer_index"])
        self.rows.append({
            "id": len(self.rows) + 1, "student_id": student, "quiz_id": quiz["id"], "topic": quiz["topic"],
            "correct": correct, "total": len(quiz["questions"]), "answers": answers,
        })


def one(analyses, quiz_id=1):
    return next(a for a in analyses if a.quiz_id == quiz_id)


class CohortTests(unittest.TestCase):
    def test_small_class_gets_no_numbers_at_all(self):
        quiz, log = make_quiz(1), Attempts()
        for i in range(4):
            log.add(f"s{i}", quiz, [1, 1, 1])
        analysis = one(analyze_quizzes([quiz], log.rows))
        self.assertEqual((analysis.students, analysis.reportable), (4, False))
        self.assertEqual(analysis.questions, [])
        self.assertIsNone(analysis.average)

    def test_five_students_are_enough(self):
        quiz, log = make_quiz(1), Attempts()
        for i in range(5):
            log.add(f"s{i}", quiz, [0, 0, 1])
        self.assertTrue(one(analyze_quizzes([quiz], log.rows)).reportable)

    def test_quiz_nobody_took_is_left_out(self):
        self.assertEqual(analyze_quizzes([make_quiz(1)], []), [])


class MisconceptionTests(unittest.TestCase):
    def build(self, picks):
        quiz, log = make_quiz(1, n_questions=1), Attempts()
        for i, pick in enumerate(picks):
            log.add(f"s{i}", quiz, [pick])
        return one(analyze_quizzes([quiz], log.rows)).questions[0]

    def test_popular_wrong_answer_is_reported_as_the_misconception(self):
        q = self.build([2] * 12 + [0] * 5 + [1] * 2 + [3])
        self.assertEqual(q.misconception.text, "wrong-b")
        self.assertEqual(q.misconception.count, 12)
        self.assertAlmostEqual(q.misconception.share, 0.6)

    def test_popular_wrong_answer_that_outvotes_the_right_one_is_flagged(self):
        q = self.build([2] * 12 + [0] * 5 + [1] * 2 + [3])
        self.assertIn(FLAG_OUTVOTED, q.flags)
        self.assertIn(FLAG_VERY_HARD, q.flags)

    def test_errors_spread_evenly_are_not_a_misconception(self):
        q = self.build([0] * 8 + [1] * 4 + [2] * 4 + [3] * 4)
        self.assertIsNone(q.misconception)

    def test_wrong_answer_chosen_by_only_two_students_is_not_a_misconception(self):
        q = self.build([0] * 3 + [1] * 2)  # 40 percent, but only 2 students
        self.assertIsNone(q.misconception)

    def test_tie_between_wrong_answers_picks_the_first_option(self):
        q = self.build([1] * 40 + [2] * 40)
        self.assertEqual(q.misconception.text, "wrong-a")

    def test_luck_in_a_small_class_is_not_reported_as_a_misconception(self):
        # 24 students, 11 wrong: one wrong option got 7 (29 percent of the class). Chance alone does that often.
        q = self.build([0] * 13 + [1] * 7 + [2] * 2 + [3] * 2)
        self.assertGreaterEqual(q.options[1].share, 0.25)
        self.assertIsNone(q.misconception)
        self.assertNotIn(FLAG_OUTVOTED, q.flags)

    def test_options_count_and_shares_add_up(self):
        q = self.build([2] * 3 + [0] * 5 + [1] * 2)
        self.assertEqual([o.count for o in q.options], [5, 2, 3, 0])
        self.assertAlmostEqual(sum(o.share for o in q.options), 1.0)
        self.assertEqual(q.correct_option.text, "right")


class DifficultyTests(unittest.TestCase):
    def rate(self, correct, wrong):
        quiz, log = make_quiz(1, n_questions=1), Attempts()
        for i in range(correct + wrong):
            log.add(f"s{i}", quiz, [0 if i < correct else 1])
        return one(analyze_quizzes([quiz], log.rows)).questions[0]

    def test_easy_question(self):
        q = self.rate(9, 1)
        self.assertEqual(q.correct_rate, 0.9)
        self.assertIn(FLAG_TOO_EASY, q.flags)

    def test_hard_question(self):
        q = self.rate(3, 7)
        self.assertIn(FLAG_VERY_HARD, q.flags)

    def test_middle_difficulty_has_no_difficulty_flag(self):
        q = self.rate(6, 4)
        self.assertFalse({FLAG_TOO_EASY, FLAG_VERY_HARD} & set(q.flags))


class CountingRuleTests(unittest.TestCase):
    def test_only_the_first_attempt_counts(self):
        quiz, log = make_quiz(1, n_questions=1), Attempts()
        for i in range(5):
            log.add(f"s{i}", quiz, [1])          # everyone wrong the first time
        for i in range(5):
            log.add(f"s{i}", quiz, [0])          # everyone right on the retake
        analysis = one(analyze_quizzes([quiz], log.rows))
        self.assertEqual(analysis.students, 5)
        self.assertEqual(analysis.questions[0].correct_rate, 0.0)

    def test_skipped_question_counts_as_wrong_and_is_reported(self):
        quiz, log = make_quiz(1, n_questions=1), Attempts()
        for i in range(4):
            log.add(f"s{i}", quiz, [0])
        log.add("skipper", quiz, [None])
        q = one(analyze_quizzes([quiz], log.rows)).questions[0]
        self.assertEqual((q.skipped, q.correct_rate), (1, 0.8))
        self.assertEqual(sum(o.count for o in q.options), 4)

    def test_short_or_garbled_answer_lists_do_not_crash(self):
        quiz, log = make_quiz(1, n_questions=3), Attempts()
        log.add("a", quiz, [0])                  # shorter than the quiz
        log.add("b", quiz, [0, 99, -1])          # indexes outside the options
        log.add("c", quiz, [0, True, 0])         # a bool is not a valid pick
        log.add("d", quiz, [])
        log.add("e", quiz, [0, 0, 0])
        analysis = one(analyze_quizzes([quiz], log.rows))
        self.assertEqual([q.skipped for q in analysis.questions], [1, 4, 3])

    def test_average_is_the_mean_first_attempt_score(self):
        quiz, log = make_quiz(1, n_questions=2), Attempts()
        for i in range(5):
            log.add(f"s{i}", quiz, [0, 0] if i < 2 else [0, 1])
        self.assertAlmostEqual(one(analyze_quizzes([quiz], log.rows)).average, 0.7)

    def test_attempts_on_unknown_quizzes_are_ignored(self):
        quiz, log = make_quiz(1), Attempts()
        for i in range(5):
            log.add(f"s{i}", make_quiz(99), [0, 0, 0])
        self.assertEqual(analyze_quizzes([quiz], log.rows), [])


class DiscriminationTests(unittest.TestCase):
    """24 students whose ability rises with their number, taking two 4-question quizzes."""

    N = 24

    def build(self, reversed_item=None, students=N):
        quizzes = [make_quiz(1, 4), make_quiz(2, 4)]
        log = Attempts()
        for k in range(students):
            ability = k / students
            for quiz in quizzes:
                answers = []
                for i in range(4):
                    threshold = 0.2 + 0.15 * i + (0.0 if quiz["id"] == 1 else 0.1)
                    right = ability >= threshold
                    if reversed_item == (quiz["id"], i):
                        right = ability < 0.5     # the faulty-key pattern: strong students miss it
                    answers.append(0 if right else 1)
                log.add(f"s{k}", quiz, answers)
        return quizzes, log.rows

    def test_normal_questions_separate_strong_from_weak(self):
        quizzes, rows = self.build()
        for analysis in analyze_quizzes(quizzes, rows):
            for q in analysis.questions:
                self.assertGreater(q.discrimination, 0.3, (analysis.topic, q.number))
                self.assertNotIn(FLAG_CHECK_KEY, q.flags)

    def test_planted_faulty_key_is_the_only_question_flagged_for_the_key(self):
        quizzes, rows = self.build(reversed_item=(2, 1))
        flagged = [
            (a.quiz_id, q.number - 1)
            for a in analyze_quizzes(quizzes, rows) for q in a.questions if FLAG_CHECK_KEY in q.flags
        ]
        self.assertEqual(flagged, [(2, 1)])

    def test_too_few_students_gives_no_discrimination(self):
        quizzes, rows = self.build(students=8)
        for analysis in analyze_quizzes(quizzes, rows):
            self.assertTrue(analysis.reportable)
            self.assertTrue(all(q.discrimination is None for q in analysis.questions))

    def test_students_with_too_few_other_answers_are_not_used(self):
        quiz, log = make_quiz(1, 3), Attempts()
        for k in range(20):
            log.add(f"s{k}", quiz, [0 if k >= 10 else 1] * 3)   # 2 other items each: under the minimum of 4
        q = one(analyze_quizzes([quiz], log.rows)).questions[0]
        self.assertIsNone(q.discrimination)


class DiscriminationCautionTests(unittest.TestCase):
    """The key check must not cry wolf on noise, and must not judge questions almost nobody got right."""

    def noisy_class(self, n=24):
        """A deterministic class where one question is only weakly and unreliably related to ability."""
        import numpy as np
        from scipy.stats import pearsonr
        for seed in range(500):
            rng = np.random.RandomState(seed)
            ability = np.linspace(0.05, 0.95, n)
            right = rng.rand(n) < 0.5 - 0.15 * (ability - 0.5)       # a slight tilt against strong students
            others = [rng.rand(n) < 0.15 + 0.7 * ability for _ in range(5)]
            rest = np.mean(others, axis=0)
            if 8 <= right.sum() <= 16:
                r, p = pearsonr(right.astype(float), rest)
                if -0.45 < r <= -0.2 and p >= 0.05:
                    return right, others
        self.fail("could not build the noisy example")

    def test_mild_negative_correlation_is_not_called_a_faulty_key(self):
        right, others = self.noisy_class()
        quiz, log = make_quiz(1, 6), Attempts()
        for k in range(len(right)):
            answers = [0 if right[k] else 1] + [0 if o[k] else 1 for o in others]
            log.add(f"s{k}", quiz, answers)
        q = one(analyze_quizzes([quiz], log.rows)).questions[0]
        self.assertLessEqual(q.discrimination, -0.2)
        self.assertGreaterEqual(q.discrimination_p, 0.05)
        self.assertNotIn(FLAG_CHECK_KEY, q.flags)

    def test_question_almost_nobody_got_right_has_no_discrimination(self):
        quiz, log = make_quiz(1, 5), Attempts()
        for k in range(24):
            first = 0 if k < 4 else 1                         # only 4 students right
            log.add(f"s{k}", quiz, [first] + [0 if k % 2 else 1] * 4)
        q = one(analyze_quizzes([quiz], log.rows)).questions[0]
        self.assertIsNone(q.discrimination)
        self.assertNotIn(FLAG_CHECK_KEY, q.flags)


class UnusedOptionTests(unittest.TestCase):
    def test_easy_question_is_not_blamed_for_unused_wrong_options(self):
        quiz, log = make_quiz(1, 1), Attempts()
        for i in range(12):
            log.add(f"s{i}", quiz, [0])                       # everyone right, so no wrong option is used
        flags = one(analyze_quizzes([quiz], log.rows)).questions[0].flags
        self.assertIn(FLAG_TOO_EASY, flags)
        self.assertNotIn(FLAG_UNUSED, flags)

    def test_option_nobody_chose_is_flagged_only_with_enough_students(self):
        def build(n):
            quiz, log = make_quiz(1, 1), Attempts()
            for i in range(n):
                log.add(f"s{i}", quiz, [0 if i % 2 else 1])    # options 2 and 3 never picked
            return one(analyze_quizzes([quiz], log.rows)).questions[0]
        self.assertIn(FLAG_UNUSED, build(12).flags)
        self.assertNotIn(FLAG_UNUSED, build(8).flags)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.quizzes = [make_quiz(1, 2), make_quiz(2, 2)]
        log = Attempts()
        for i in range(10):
            # quiz 1: Q1 has a popular wrong answer; Q2 is fine
            log.add(f"s{i}", self.quizzes[0], [2 if i < 7 else 0, 0 if i < 9 else 1])
            # quiz 2: Q1 easy, Q2 has a smaller popular wrong answer
            log.add(f"s{i}", self.quizzes[1], [0, 3 if i >= 4 else 0])
        self.analyses = analyze_quizzes(self.quizzes, log.rows)

    def test_misconceptions_are_listed_most_popular_first(self):
        found = likely_misconceptions(self.analyses)
        self.assertEqual([(q.quiz_id, q.number) for q in found], [(1, 1), (2, 2)])
        self.assertAlmostEqual(found[0].misconception.share, 0.7)

    def test_review_list_puts_serious_flags_first(self):
        review = questions_to_review(self.analyses)
        self.assertEqual((review[0].quiz_id, review[0].number), (1, 1))   # a wrong answer beat the right one
        self.assertIn((2, 1), [(q.quiz_id, q.number) for q in review])    # the too-easy question is also listed

    def test_unreportable_quizzes_contribute_nothing(self):
        quiz, log = make_quiz(5, 1), Attempts()
        for i in range(3):
            log.add(f"s{i}", quiz, [1])
        analyses = analyze_quizzes([quiz], log.rows)
        self.assertEqual(likely_misconceptions(analyses) + questions_to_review(analyses), [])


if __name__ == "__main__":
    unittest.main()
