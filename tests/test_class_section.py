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
    KIND_CLARIFICATION, STATUS_APPROVED, STATUS_DISMISSED, add_advice, list_advice, set_status,
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


def questions(topic: str, file: str = "notes.pdf"):
    return [
        {"question": f"What is {topic}?", "options": ["right", "wrong-a", "wrong-b", "wrong-c"], "answer_index": 0,
         "explanation": "Because.", "source": {"file": file, "page": 3, "slide": topic}},
        {"question": f"Why does {topic} matter?", "options": ["wrong-a", "right", "wrong-b", "wrong-c"], "answer_index": 1,
         "explanation": "Because.", "source": {"file": file, "page": 4, "slide": topic}},
    ]


class ClassSectionTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def page(self, student="s1", part=None, direct=False, subject=None, topic=None, started=None):
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
        for key, value in (("class_subject", subject), ("class_topic", topic), ("class_started", started)):
            if value is not None:
                at.session_state[key] = value
        at.run()
        return at

    def publish(self, topic, scope="class", file="notes.pdf"):
        return create_quiz(topic, f"Quiz on {topic}", questions(topic, file), scope=scope, db_dir=self.db)

    @staticmethod
    def everything(at):
        return " | ".join(
            [m.value for m in at.markdown] + [c.value for c in at.caption] + [i.value for i in at.info]
            + [str(s.options) for s in at.selectbox]
        )


class PracticeQuizTests(ClassSectionTestCase):
    def label(self, at):
        return at.selectbox(key="class_topic").options[0]

    def topic_labels(self, subject="notes"):
        """The topic choices a student sees after choosing a subject."""
        return list(self.page(subject=subject).selectbox(key="class_topic").options)

    def test_with_no_published_quiz_it_says_so(self):
        at = self.page()
        self.assertEqual(at.exception, [])
        self.assertEqual([i.value for i in at.info], ["No practice quizzes yet."])

    def test_nothing_opens_until_the_student_chooses(self):
        self.publish("Deadlock")
        at = self.page()
        self.assertEqual(at.exception, [])
        self.assertIsNone(at.selectbox(key="class_subject").value)
        self.assertEqual(len(at.selectbox), 1)                    # no topic list yet
        self.assertEqual(len(at.radio), 0)                        # and no questions
        at = self.page(subject="notes")
        self.assertIsNone(at.selectbox(key="class_topic").value)
        self.assertEqual(len(at.radio), 0)

    def test_subjects_follow_the_course_document_the_questions_came_from(self):
        self.publish("Deadlock", file="Operating Systems.pdf")
        self.publish("Paging", file="Operating Systems.pdf")
        self.publish("Spark", file="Big Data.pdf")
        at = self.page()
        self.assertEqual(list(at.selectbox(key="class_subject").options), ["Big Data", "Operating Systems"])
        self.assertEqual(len(self.topic_labels("Operating Systems")), 2)
        self.assertEqual(len(self.topic_labels("Big Data")), 1)

    def test_only_quizzes_published_to_the_class_are_offered(self):
        self.publish("Deadlock")
        self.publish("Paging")
        self.publish("Hidden Draft", scope="draft")
        self.publish("A Personal Quiz", scope="personal")
        labels = self.topic_labels()
        joined = " ".join(labels)
        self.assertEqual(len(labels), 2)
        self.assertIn("Deadlock", joined)
        self.assertIn("Paging", joined)
        self.assertNotIn("Hidden Draft", joined)
        self.assertNotIn("Personal", joined)

    def test_a_draft_appears_once_the_teacher_publishes_it(self):
        draft = self.publish("Caching", scope="draft")
        self.assertEqual([i.value for i in self.page().info], ["No practice quizzes yet."])
        set_quiz_scope(draft, "class", self.db)
        self.assertEqual(len(self.topic_labels()), 1)

    def test_the_list_says_how_many_questions_and_whether_it_was_taken(self):
        self.publish("Deadlock")
        self.assertEqual(self.topic_labels(), ["Deadlock - 2 questions - not taken"])

    def test_the_wording_does_not_announce_who_made_the_quiz(self):
        self.publish("Deadlock")
        quiz_id = self.publish("Paging")
        text = self.everything(self.page(subject="notes", topic=quiz_id)).lower()
        for word in ("teacher", "professor", "faculty", "generated", "ai ", "tricky", "found "):
            self.assertNotIn(word, text, word)

    def test_no_explanatory_text_around_the_choices(self):
        self.publish("Deadlock")
        at = self.page(subject="notes")
        self.assertEqual([m.value for m in at.markdown if m.value.strip() and not m.value.startswith("<style")], [])
        self.assertEqual([c.value for c in at.caption], [])

    def test_choosing_a_topic_still_does_not_start_the_quiz(self):
        quiz_id = self.publish("Deadlock")
        at = self.page(subject="notes", topic=quiz_id)
        self.assertEqual(len(at.radio), 0)
        self.assertEqual([b.label for b in at.button], ["Start quiz"])

    def test_start_quiz_shows_the_questions(self):
        quiz_id = self.publish("Deadlock")
        at = self.page(direct=True, subject="notes", topic=quiz_id)
        [b for b in at.button if b.label == "Start quiz"][0].click().run()
        self.assertEqual(at.exception, [])
        self.assertEqual(len(at.radio), 2)

    def test_changing_the_subject_clears_the_topic(self):
        deadlock = self.publish("Deadlock", file="Operating Systems.pdf")
        self.publish("Spark", file="Big Data.pdf")
        at = self.page(direct=True, subject="Operating Systems", topic=deadlock)
        self.assertEqual(at.selectbox(key="class_topic").value, deadlock)
        at.selectbox(key="class_subject").set_value("Big Data").run()
        self.assertIsNone(at.selectbox(key="class_topic").value)
        self.assertEqual(len(at.radio), 0)

    def take(self, at, quiz_id, picks):
        for i, pick in enumerate(picks):
            at.session_state[f"pick_class_{quiz_id}_{i}"] = pick           # AppTest cannot drive a radio with a display function
        [b for b in at.button if b.label == "Submit answers"][0].click().run()

    def started(self, quiz_id, student="s1"):
        return self.page(student=student, direct=True, subject="notes", topic=quiz_id, started=quiz_id)

    def test_taking_a_quiz_saves_the_attempt_under_the_student_and_shows_the_first_score(self):
        quiz_id = self.publish("Deadlock")
        at = self.started(quiz_id)
        self.take(at, quiz_id, [0, 0])                                         # one right, one wrong
        self.assertEqual(at.exception, [])
        saved = list_attempts("s1", self.db)
        self.assertEqual([(a["quiz_id"], a["correct"], a["total"]) for a in saved], [(quiz_id, 1, 2)])
        self.assertEqual(self.topic_labels(), ["Deadlock - 2 questions - done, 1/2"])

    def test_a_retake_does_not_change_the_first_score_shown(self):
        quiz_id = self.publish("Deadlock")
        self.take(self.started(quiz_id), quiz_id, [0, 0])       # 1 of 2
        self.take(self.started(quiz_id), quiz_id, [0, 1])       # 2 of 2 on the retake
        self.assertTrue(self.topic_labels()[0].endswith("done, 1/2"))

    def test_another_students_attempt_is_not_shown_to_me_and_counts_in_the_class_results(self):
        quiz_id = self.publish("Deadlock")
        self.take(self.started(quiz_id, student="s2"), quiz_id, [0, 1])
        self.assertTrue(self.topic_labels()[0].endswith("not taken"))
        self.assertEqual([a["student_id"] for a in list_attempts(db_dir=self.db)], ["s2"])        # the teacher's view reads this


class AddedMaterialTests(ClassSectionTestCase):
    def resolve(self, question, answer, student="secret-student", scope=None, subject=None):
        row = log_query(question, False, 1.5, [], db_dir=self.db, student_id=student, scope=scope)
        resolve_gaps([row], question, answer, self.db, subject=subject)

    def added(self, subject=None, item=None):
        """The Added-to-the-material page; `item` is a question or note title to open."""
        state = {}
        if subject:
            state["added_subject"] = subject
        if item:
            entries = [(f"{a['at']}|{a['question']}", a["question"]) for a in resolved_answers_dated(self.db)]
            entries += [(f"{n['decided_at']}|Note: {n['concept']}", f"Note: {n['concept']}")
                        for n in list_advice(kind=KIND_CLARIFICATION, status=STATUS_APPROVED, db_dir=self.db)]
            state["added_item"] = next(key for key, title in entries if title == item)
        from streamlit.testing.v1 import AppTest

        path = ROOT / "_tmp_added_page.py"
        path.write_text(PREFIX.format(db=self.db) + page_source() + "\nrender_class_section(None, 's1')\n", encoding="utf-8")
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        at = AppTest.from_file(str(path), default_timeout=120)
        at.session_state["class_part"] = "Added to the material"
        for key, value in state.items():
            at.session_state[key] = value
        at.run()
        self.assertEqual(at.exception, [])
        return at

    def approve_note(self, concept, body):
        note = add_advice(concept.lower(), concept, KIND_CLARIFICATION, "t", body, db_dir=self.db)
        set_status(note, STATUS_APPROVED, self.db)

    def test_with_nothing_added_it_says_so(self):
        at = self.added()
        self.assertEqual([i.value for i in at.info], ["Nothing has been added yet."])

    def test_nothing_is_shown_until_the_student_chooses(self):
        self.resolve("What is a Kubernetes pod?", "A pod is the smallest deployable unit.")
        at = self.added()
        self.assertIsNone(at.selectbox(key="added_subject").value)
        self.assertEqual(len(at.selectbox), 1)
        self.assertNotIn("A pod is the smallest deployable unit.", self.everything(at))
        at = self.added(subject="General")
        self.assertIsNone(at.selectbox(key="added_item").value)
        self.assertNotIn("A pod is the smallest deployable unit.", self.everything(at))

    def test_the_answer_and_date_appear_once_chosen(self):
        self.resolve("What is a Kubernetes pod?", "A pod is the smallest deployable unit.")
        at = self.added(subject="General", item="What is a Kubernetes pod?")
        self.assertIn("A pod is the smallest deployable unit.", [m.value for m in at.markdown])
        self.assertTrue(any(c.value.startswith("Added 20") for c in at.caption))

    def test_items_are_grouped_by_the_subject_the_student_was_studying(self):
        self.resolve("What is a pod?", "A group of containers.", scope=["Containers.pdf"])
        self.resolve("What is a lock?", "A way to control access.", scope=["Operating Systems.pdf"])
        self.resolve("What is a latch?", "A short lock.", scope=["Operating Systems.pdf"])
        self.resolve("Anything else?", "Yes.", scope=["Containers.pdf", "Operating Systems.pdf"])    # no single subject
        at = self.added()
        self.assertEqual(list(at.selectbox(key="added_subject").options), ["Containers", "General", "Operating Systems"])
        self.assertEqual(len(self.added(subject="Operating Systems").selectbox(key="added_item").options), 2)
        self.assertEqual(len(self.added(subject="Containers").selectbox(key="added_item").options), 1)

    def test_the_subject_the_teacher_chose_wins_over_what_the_student_had_selected(self):
        self.resolve("What is a pod?", "A group of containers.", scope=["Containers.pdf"], subject="Operating Systems.pdf")
        self.resolve("What is a lock?", "Access control.", scope=["Operating Systems.pdf"], subject="General")
        self.resolve("What is a latch?", "A short lock.", scope=["Operating Systems.pdf", "Containers.pdf"], subject="Containers.pdf")
        self.assertEqual(list(self.added().selectbox(key="added_subject").options), ["Containers", "General", "Operating Systems"])
        self.assertEqual(list(self.added(subject="Operating Systems").selectbox(key="added_item").options), ["What is a pod?"])
        self.assertEqual(list(self.added(subject="General").selectbox(key="added_item").options), ["What is a lock?"])

    def test_a_note_without_a_known_subject_goes_under_general(self):
        self.approve_note("Deadlock", "Deadlock needs a cycle of waiting.")
        self.assertEqual(list(self.added().selectbox(key="added_subject").options), ["General"])
        at = self.added(subject="General", item="Note: Deadlock")
        self.assertIn("Deadlock needs a cycle of waiting.", [m.value for m in at.markdown])

    def test_several_students_asking_the_same_thing_give_one_entry(self):
        for student in ("a", "b", "c"):
            self.resolve("What is a pod?", "A pod is a group of containers.", student=student)
        self.assertEqual(len(resolved_answers_dated(self.db)), 1)
        self.assertEqual(len(self.added(subject="General").selectbox(key="added_item").options), 1)

    def test_approved_notes_appear_but_drafts_and_dismissed_ones_do_not(self):
        self.approve_note("Deadlock", "Deadlock needs a cycle of waiting.")
        add_advice("paging", "Paging", KIND_CLARIFICATION, "t", "A draft nobody approved.", db_dir=self.db)
        dismissed = add_advice("cache", "Caching", KIND_CLARIFICATION, "t", "A note that was withdrawn.", db_dir=self.db)
        set_status(dismissed, STATUS_DISMISSED, self.db)
        self.assertEqual(list(self.added(subject="General").selectbox(key="added_item").options), ["Note: Deadlock"])

    def test_newest_first(self):
        import sqlite3
        from contextlib import closing
        from rag_core.config import QUERY_LOG_FILENAME

        self.resolve("Older question?", "Older answer.")
        self.resolve("Newer question?", "Newer answer.")
        with closing(sqlite3.connect(self.db / QUERY_LOG_FILENAME)) as conn, conn:
            conn.execute("UPDATE queries SET resolved_at = '2026-01-01T10:00:00+00:00' WHERE resolved_question = 'Older question?'")
            conn.execute("UPDATE queries SET resolved_at = '2026-02-01T10:00:00+00:00' WHERE resolved_question = 'Newer question?'")
        options = list(self.added(subject="General").selectbox(key="added_item").options)
        self.assertEqual(options, ["Newer question?", "Older question?"])

    def test_no_explanatory_text_around_the_choices(self):
        self.resolve("What is a pod?", "A pod is a group of containers.")
        at = self.added(subject="General")
        self.assertEqual([m.value for m in at.markdown if m.value.strip() and not m.value.startswith("<style")], [])
        self.assertEqual([c.value for c in at.caption], [])

    def test_no_student_is_identified_anywhere(self):
        self.resolve("What is a pod?", "A pod is a group of containers.", student="secret-student")
        self.assertNotIn("secret-student", self.everything(self.added(subject="General", item="What is a pod?")))
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
