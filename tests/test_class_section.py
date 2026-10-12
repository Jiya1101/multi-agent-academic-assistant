"""
Tests for the student's "Class" section: practice quizzes on topics the class found tricky, and what the teacher has
added to the course material.

Checked here:
  - only quizzes the teacher PUBLISHED to the class are offered (never drafts or another student's personal quiz);
  - taking one is saved under the student, counts for the teacher's class results, and the list shows their first score;
  - the added material lists the teacher's answers to class questions and approved notes, newest first, and never
    anything still in draft or dismissed;
  - nothing on the page identifies a student.

Run:  python -m unittest discover -s tests -v
"""

import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

from rag_core.advice_log import (
    KIND_CLARIFICATION, STATUS_APPROVED, STATUS_DISMISSED, add_advice, set_status,
)
from rag_core.learning_log import create_quiz, list_attempts, set_quiz_scope
from rag_core.query_log import list_queries, log_query, resolve_gaps, resolved_answers_dated

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


class ClassSectionTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def page(self, student="s1", part=None, direct=False):
        """`direct` renders the practice-quiz part on its own: Streamlit's test tool cannot re-run a page that holds
        a segmented control, so the quiz-taking tests skip the control and exercise the part itself."""
        from streamlit.testing.v1 import AppTest

        path = ROOT / "_tmp_class_page.py"
        body = f"_render_practice_quizzes(None, {student!r})" if direct else f"render_class_section(None, {student!r})"
        path.write_text(PREFIX.format(db=self.db) + page_source() + f"\n{body}\n", encoding="utf-8")
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        at = AppTest.from_file(str(path), default_timeout=120)
        if part:
            at.session_state["class_part"] = part
        at.run()
        return at

    def publish(self, topic, scope="class"):
        return create_quiz(topic, f"Quiz on {topic}", questions(topic), scope=scope, db_dir=self.db)

    @staticmethod
    def everything(at):
        return " | ".join(
            [m.value for m in at.markdown] + [c.value for c in at.caption] + [i.value for i in at.info]
            + [str(s.options) for s in at.selectbox]
        )


class PracticeQuizTests(ClassSectionTestCase):
    def test_with_no_published_quiz_it_says_so(self):
        at = self.page()
        self.assertEqual(at.exception, [])
        self.assertEqual([i.value for i in at.info], ["No practice quizzes yet."])

    def test_only_quizzes_published_to_the_class_are_offered(self):
        self.publish("Deadlock")
        self.publish("Paging")
        self.publish("Hidden Draft", scope="draft")
        self.publish("Someone's Own Quiz", scope="personal")
        at = self.page()
        options = at.selectbox(key="class_quiz_pick").options
        self.assertEqual(len(options), 2)
        joined = " ".join(options)
        self.assertIn("Deadlock", joined)
        self.assertIn("Paging", joined)
        self.assertNotIn("Hidden Draft", joined)
        self.assertNotIn("Own Quiz", joined)

    def test_a_draft_appears_once_the_teacher_publishes_it(self):
        draft = self.publish("Caching", scope="draft")
        self.assertEqual([i.value for i in self.page().info], ["No practice quizzes yet."])
        set_quiz_scope(draft, "class", self.db)
        self.assertEqual(len(self.page().selectbox(key="class_quiz_pick").options), 1)

    def test_the_list_says_how_many_questions_and_whether_it_was_taken(self):
        self.publish("Deadlock")
        label = self.page().selectbox(key="class_quiz_pick").options[0]
        self.assertEqual(label, "Deadlock - 2 questions - not taken")

    def test_the_wording_does_not_announce_who_made_the_quiz(self):
        self.publish("Deadlock")
        text = self.everything(self.page()).lower()
        for word in ("teacher", "professor", "faculty", "generated", "ai "):
            self.assertNotIn(word, text, word)
        self.assertIn("practice on topics the class has found tricky", text)

    def take(self, at, quiz_id, picks):
        for i, pick in enumerate(picks):
            at.session_state[f"pick_class_{quiz_id}_{i}"] = pick           # AppTest cannot drive a radio with a display function
        [b for b in at.button if b.label == "Submit answers"][0].click().run()

    def test_taking_a_quiz_saves_the_attempt_under_the_student_and_shows_the_first_score(self):
        quiz_id = self.publish("Deadlock")
        at = self.page(student="s1", direct=True)
        self.take(at, quiz_id, [0, 0])                                         # one right, one wrong
        self.assertEqual(at.exception, [])
        saved = list_attempts("s1", self.db)
        self.assertEqual([(a["quiz_id"], a["correct"], a["total"]) for a in saved], [(quiz_id, 1, 2)])
        self.assertEqual(self.page(student="s1").selectbox(key="class_quiz_pick").options[0], "Deadlock - 2 questions - done, 1/2")

    def test_a_retake_does_not_change_the_first_score_shown(self):
        quiz_id = self.publish("Deadlock")
        self.take(self.page(student="s1", direct=True), quiz_id, [0, 0])       # 1 of 2
        self.take(self.page(student="s1", direct=True), quiz_id, [0, 1])       # 2 of 2 on the retake
        label = self.page(student="s1").selectbox(key="class_quiz_pick").options[0]
        self.assertTrue(label.endswith("done, 1/2"), label)

    def test_another_students_attempt_is_not_shown_to_me_and_counts_in_the_class_results(self):
        quiz_id = self.publish("Deadlock")
        self.take(self.page(student="s2", direct=True), quiz_id, [0, 1])
        self.assertTrue(self.page(student="s1").selectbox(key="class_quiz_pick").options[0].endswith("not taken"))
        self.assertEqual([a["student_id"] for a in list_attempts(db_dir=self.db)], ["s2"])        # the teacher's view reads this


class AddedMaterialTests(ClassSectionTestCase):
    def resolve(self, question, answer, student="secret-student"):
        row = log_query(question, False, 1.5, [], db_dir=self.db, student_id=student)
        resolve_gaps([row], question, answer, self.db)

    def test_with_nothing_added_it_says_so(self):
        at = self.page(part="Added to the material")
        self.assertEqual(at.exception, [])
        self.assertEqual([i.value for i in at.info], ["Nothing has been added yet."])

    def test_the_teachers_answers_are_listed_with_the_question_and_the_date(self):
        self.resolve("What is a Kubernetes pod?", "A pod is the smallest deployable unit.")
        at = self.page(part="Added to the material")
        text = self.everything(at)
        self.assertIn("**What is a Kubernetes pod?**", text)
        self.assertIn("A pod is the smallest deployable unit.", [m.value for m in at.markdown])
        self.assertTrue(any(c.startswith("Added 20") for c in [c.value for c in at.caption]))

    def test_several_students_asking_the_same_thing_give_one_entry(self):
        for student in ("a", "b", "c"):
            self.resolve("What is a pod?", "A pod is a group of containers.", student=student)
        self.assertEqual(len(resolved_answers_dated(self.db)), 1)
        at = self.page(part="Added to the material")
        self.assertEqual(len([m for m in at.markdown if m.value == "**What is a pod?**"]), 1)

    def test_approved_notes_appear_but_drafts_and_dismissed_ones_do_not(self):
        approved = add_advice("deadlock", "Deadlock", KIND_CLARIFICATION, "t", "Deadlock needs a cycle of waiting.", db_dir=self.db)
        set_status(approved, STATUS_APPROVED, self.db)
        add_advice("paging", "Paging", KIND_CLARIFICATION, "t", "A draft nobody approved.", db_dir=self.db)
        dismissed = add_advice("cache", "Caching", KIND_CLARIFICATION, "t", "A note that was withdrawn.", db_dir=self.db)
        set_status(dismissed, STATUS_DISMISSED, self.db)
        at = self.page(part="Added to the material")
        values = [m.value for m in at.markdown]
        self.assertIn("**Note: Deadlock**", values)
        self.assertIn("Deadlock needs a cycle of waiting.", values)
        self.assertNotIn("A draft nobody approved.", values)
        self.assertNotIn("A note that was withdrawn.", values)

    def test_newest_first(self):
        import sqlite3
        from contextlib import closing
        from rag_core.config import QUERY_LOG_FILENAME

        self.resolve("Older question?", "Older answer.")
        self.resolve("Newer question?", "Newer answer.")
        with closing(sqlite3.connect(self.db / QUERY_LOG_FILENAME)) as conn, conn:
            conn.execute("UPDATE queries SET resolved_at = '2026-01-01T10:00:00+00:00' WHERE resolved_question = 'Older question?'")
            conn.execute("UPDATE queries SET resolved_at = '2026-02-01T10:00:00+00:00' WHERE resolved_question = 'Newer question?'")
        titles = [m.value for m in self.page(part="Added to the material").markdown if m.value.startswith("**")]
        self.assertEqual(titles, ["**Newer question?**", "**Older question?**"])

    def test_no_student_is_identified_anywhere(self):
        self.resolve("What is a pod?", "A pod is a group of containers.", student="secret-student")
        self.assertNotIn("secret-student", self.everything(self.page(part="Added to the material")))
        self.assertNotIn("secret-student", str(resolved_answers_dated(self.db)))


class StudentNavigationTests(unittest.TestCase):
    def test_the_student_navigation_includes_the_class_section_after_quiz(self):
        src = (ROOT / "app.py").read_text(encoding="utf-8")
        sections = re.search(r"sections = (\[[^\]]*\])", src).group(1)
        self.assertEqual(eval(sections), ["Material", "Ask", "Oral check", "Study notes", "Quiz", "Class", "My progress"])
        self.assertIn('if section == "Class":', src)


class ProfessorInsightTests(unittest.TestCase):
    def test_the_slides_students_land_on_chart_is_gone_from_class_insight(self):
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp)
            for i in range(4):
                log_query(f"What is deadlock {i}?", True, 0.4,
                          [Document(page_content="x", metadata={"source": "notes.pdf", "page": 1, "section": "Deadlock"})],
                          db_dir=db, student_id=f"s{i}")
            page = ROOT / "_tmp_insight_page.py"
            page.write_text(PREFIX.format(db=db) + page_source() + "\n_render_class_insight()\n", encoding="utf-8")
            try:
                at = AppTest.from_file(str(page), default_timeout=120)
                at.run()
            finally:
                page.unlink(missing_ok=True)
        self.assertEqual(at.exception, [])
        self.assertNotIn("Slides students land on most", [s.value for s in at.subheader])
        self.assertEqual(len(at.get("arrow_bar_chart")), 0)
        self.assertIn("Topics students are asking about", [s.value for s in at.subheader])    # the rest of the tab is intact


if __name__ == "__main__":
    unittest.main()
