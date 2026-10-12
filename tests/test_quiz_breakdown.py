"""
Tests for the professor's optional "Show how the class answered" switch inside each published quiz.

Checked here:
  - there is no separate "Question analysis" section any more;
  - the breakdown is off by default (no percentages next to the options);
  - switched on for a quiz with too few students, it says how many more are needed and shows no numbers;
  - switched on for a quiz with enough students, each option shows the share who chose it, and a wrong answer
    that many students chose is pointed out under its question;
  - switching it on for one quiz does not show it on another.

Run:  python -m unittest discover -s tests -v
"""

import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_core.learning_log import create_quiz, record_attempt

ROOT = Path(__file__).resolve().parent.parent

PREFIX = '''
import rag_core.config as _c
from pathlib import Path
_c.DB_DIR = Path(r"{db}")
'''


def page_source() -> str:
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    cut = src.rfind("# ----", 0, src.index("# Sidebar + routing"))
    return re.sub(r'^"""[\s\S]*?"""\n', "", src[:cut], count=1)


def questions(topic: str):
    return [
        {"question": f"What is {topic}?", "options": ["right", "wrong-a", "wrong-b", "wrong-c"], "answer_index": 0,
         "explanation": "Because.", "source": {"file": "notes.pdf", "page": 3, "slide": topic}},
        {"question": f"Why does {topic} matter?", "options": ["wrong-a", "right", "wrong-b", "wrong-c"], "answer_index": 1,
         "explanation": "Because.", "source": {"file": "notes.pdf", "page": 4, "slide": topic}},
    ]


class QuizBreakdownTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def publish(self, topic):
        return create_quiz(topic, f"Quiz on {topic}", questions(topic), scope="class", db_dir=self.db)

    def take(self, quiz_id, topic, students, answers_for):
        for n in range(students):
            answers = answers_for(n)
            correct = sum(a == q["answer_index"] for a, q in zip(answers, questions(topic)))
            record_attempt(f"s{n}", quiz_id, topic, correct, 2, answers, db_dir=self.db)

    def page(self):
        from streamlit.testing.v1 import AppTest

        path = ROOT / "_tmp_breakdown_page.py"
        path.write_text(
            PREFIX.format(db=self.db) + page_source()
            + "\n_get_vectorstore = lambda: object()\n_render_class_quizzes()\n",
            encoding="utf-8",
        )
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        at = AppTest.from_file(str(path), default_timeout=120)
        at.run()
        self.assertEqual([e.value for e in at.exception], [])
        return at

    def text(self, at):
        return " | ".join(
            [t.value for t in at.text] + [c.value for c in at.caption] + [h.value for h in at.subheader]
        )

    def test_no_separate_question_analysis_section(self):
        quiz = self.publish("Spark")
        self.take(quiz, "Spark", 6, lambda n: [0, 1])
        text = self.text(self.page())
        self.assertNotIn("Question analysis", text)
        self.assertNotIn("Likely misconceptions", text)

    def test_off_by_default(self):
        quiz = self.publish("Spark")
        self.take(quiz, "Spark", 6, lambda n: [0, 1])
        at = self.page()
        self.assertEqual(len(at.toggle), 1)
        self.assertFalse(at.toggle[0].value)
        self.assertNotIn("chose this", self.text(at))

    def test_too_few_students_shows_how_many_more_are_needed(self):
        quiz = self.publish("Spark")
        self.take(quiz, "Spark", 2, lambda n: [0, 1])
        at = self.page()
        at.toggle[0].set_value(True).run()
        text = self.text(at)
        self.assertIn("2 of the 5 students needed so far", text)
        self.assertNotIn("chose this", text)

    def test_nobody_has_taken_it(self):
        self.publish("Spark")
        at = self.page()
        at.toggle[0].set_value(True).run()
        self.assertIn("No student has taken this quiz yet", self.text(at))

    def test_enough_students_shows_shares_and_the_common_wrong_answer(self):
        quiz = self.publish("Spark")
        # 10 students: question 1 - 7 pick the wrong option "wrong-a"; question 2 - everyone is right
        self.take(quiz, "Spark", 10, lambda n: [1 if n < 7 else 0, 1])
        at = self.page()
        at.toggle[0].set_value(True).run()
        text = self.text(at)
        self.assertIn("70% chose this", text)
        self.assertIn("30% of 10 students got this right", text)
        self.assertIn('70% of the class (7 of 10) chose "wrong-a"', text)

    def test_one_quiz_does_not_open_another(self):
        first = self.publish("Spark")
        second = self.publish("Hadoop")
        self.take(first, "Spark", 6, lambda n: [0, 1])
        self.take(second, "Hadoop", 6, lambda n: [0, 1])
        at = self.page()
        self.assertEqual(len(at.toggle), 2)
        at.toggle[0].set_value(True).run()
        shown = sum("chose this" in t.value for t in at.text)
        self.assertEqual(shown, 8)   # one quiz: 2 questions x 4 options


if __name__ == "__main__":
    unittest.main()
