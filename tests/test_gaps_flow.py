"""
End-to-end tests for Pending Gaps: students ask something the notes do not cover, the teacher answers it on the
Pending Gaps tab, and the next student who asks gets that answer.

The journey checked here:
  1. questions the notes cannot answer are queued, and similar ones are grouped;
  2. the teacher writes one answer for a group and it is added to the course material;
  3. every question in the group is marked resolved, and the queue shrinks;
  4. a student asking the same thing (or a paraphrase) now gets a grounded answer that cites the teacher's answer;
  5. a rebuild of the index keeps the answer;
  6. nothing is saved for an empty answer or when no index exists.

Run:  python -m unittest discover -s tests -v
"""

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import AgentContext
from rag_core.config import FACULTY_SOURCE_NAME
from rag_core.embeddings import get_embeddings
from rag_core.query_log import STATUS_GAP, STATUS_RESOLVED, list_queries, log_query, resolved_answers
from rag_core.vectorstore import (
    build_vectorstore,
    faculty_documents,
    load_vectorstore,
    save_vectorstore,
    vectorstore_exists,
)
from test_agents import AgentTestCase, fake

ROOT = Path(__file__).resolve().parent.parent

POD_QUESTIONS = ["What is a Kubernetes pod?", "Explain pods in Kubernetes", "how do kubernetes pods work"]
DEADLINE = "What is the deadline for assignment 2?"
POD_ANSWER = "A Kubernetes pod is the smallest unit you deploy: one or more containers that share storage and a network address."

PAGE_PREFIX = '''
import rag_core.config as _c
from pathlib import Path
_c.DB_DIR = Path(r"{db}")
'''


def page_source() -> str:
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    cut = src.rfind("# ----", 0, src.index("# Sidebar + routing"))
    return re.sub(r'^"""[\s\S]*?"""\n', "", src[:cut], count=1)


class GapsTestCase(AgentTestCase):
    def setUp(self):
        super().setUp()
        save_vectorstore(self.vectorstore, self.db)                       # the teacher's page needs a course index on disk

    def student_asks(self, question, student="s1"):
        ctx = AgentContext(vectorstore=self.vectorstore, db_dir=self.db, student_id=student, llm=fake("unused"))
        ctx.session_id = f"sess-{student}"
        return self.orch.handle(question, ctx)

    def queue_gaps(self):
        for i, q in enumerate(POD_QUESTIONS):
            self.student_asks(q, student=f"s{i}")
        self.student_asks(DEADLINE, student="s9")

    def open_page(self):
        from streamlit.testing.v1 import AppTest

        page = ROOT / "_tmp_gaps_page.py"
        page.write_text(PAGE_PREFIX.format(db=self.db) + page_source() + "\n_render_pending_gaps()\n", encoding="utf-8")
        self.addCleanup(lambda: page.unlink(missing_ok=True))
        at = AppTest.from_file(str(page), default_timeout=300)
        at.run()
        return at

    @staticmethod
    def group_markdowns(at):
        return [m.value for m in at.markdown if "student question(s)" in m.value]

    @staticmethod
    def add_button(at, nth=0):
        return [b for b in at.button if b.label == "Add to knowledge base"][nth]

    def pod_box(self, at):
        """The answer box of the group that holds the Kubernetes questions."""
        for box in at.text_area:
            if box.key.startswith("answer_"):
                container_row_id = int(box.key.split("_")[1])
                row = next(r for r in list_queries(db_dir=self.db) if r["id"] == container_row_id)
                if "pod" in row["question"].lower():
                    return box
        raise AssertionError("no answer box for the pod group")


class QueueingTests(GapsTestCase):
    def test_questions_the_notes_cannot_answer_are_queued_as_gaps(self):
        self.queue_gaps()
        gaps = list_queries(status=STATUS_GAP, db_dir=self.db)
        self.assertEqual(sorted(g["question"] for g in gaps), sorted([*POD_QUESTIONS, DEADLINE]))

    def test_similar_questions_are_grouped_and_the_rest_stand_alone(self):
        self.queue_gaps()
        at = self.open_page()
        self.assertEqual(at.exception, [])
        self.assertEqual(sorted(self.group_markdowns(at)),
                         sorted(['**3 student question(s)** — e.g. "' + self.rep_text(at) + '"',
                                 '**1 student question(s)** — e.g. "' + DEADLINE + '"']))

    @staticmethod
    def rep_text(at):
        line = next(m.value for m in at.markdown if m.value.startswith("**3 student question(s)**"))
        return re.search(r'e\.g\. "(.*)"', line).group(1)

    def test_with_nothing_waiting_it_says_so(self):
        at = self.open_page()
        self.assertEqual([s.value for s in at.success], ["No unresolved questions. Everything students asked was covered."])


class AnsweringTests(GapsTestCase):
    def answer_pods(self, text=POD_ANSWER):
        self.queue_gaps()
        at = self.open_page()
        self.pod_box(at).set_value(text)
        at.run()
        buttons = [b for b in at.button if b.label == "Add to knowledge base"]
        # the pod group is the one whose box holds the text; click its button (buttons and boxes are in the same order)
        index = [b.key for b in at.text_area if b.key.startswith("answer_")].index(self.pod_box(at).key)
        buttons[index].click().run()
        return at

    def test_the_answer_resolves_every_question_in_the_group_and_only_those(self):
        self.answer_pods()
        rows = list_queries(db_dir=self.db)
        resolved = [r for r in rows if r["status"] == STATUS_RESOLVED]
        self.assertEqual(sorted(r["question"] for r in resolved), sorted(POD_QUESTIONS))
        self.assertTrue(all(r["faculty_answer"] == POD_ANSWER for r in resolved))
        self.assertEqual([r["question"] for r in rows if r["status"] == STATUS_GAP], [DEADLINE])

    def course_files(self):
        from rag_core.vectorstore import list_indexed_sources
        return [f for f in list_indexed_sources(self.vectorstore) if f != FACULTY_SOURCE_NAME]

    def pod_subject_box(self, at):
        return at.selectbox(key=self.pod_box(at).key.replace("answer_", "subject_"))

    def test_the_teacher_files_the_answer_under_a_subject(self):
        self.queue_gaps()
        at = self.open_page()
        box = self.pod_subject_box(at)
        files = self.course_files()
        self.assertTrue(files)
        self.assertEqual(len(box.options), len(files) + 1)
        self.assertEqual(box.value, "General")                      # nothing says which subject it belongs to
        self.assertEqual(box.options[-1], "General")

    def test_the_chosen_subject_is_saved_and_students_find_the_answer_under_it(self):
        self.queue_gaps()
        at = self.open_page()
        self.pod_box(at).set_value(POD_ANSWER)
        self.pod_subject_box(at).set_value(self.course_files()[0])
        at.run()
        index = [b.key for b in at.text_area if b.key.startswith("answer_")].index(self.pod_box(at).key)
        [b for b in at.button if b.label == "Add to knowledge base"][index].click().run()
        saved = {r["resolved_subject"] for r in list_queries(db_dir=self.db) if r["status"] == STATUS_RESOLVED}
        self.assertEqual(saved, {self.course_files()[0]})
        from rag_core.query_log import resolved_answers_dated
        self.assertEqual([a["subject"] for a in resolved_answers_dated(self.db)], [self.course_files()[0]])

    def test_a_subject_left_as_general_is_saved_as_general(self):
        self.answer_pods()
        saved = {r["resolved_subject"] for r in list_queries(db_dir=self.db) if r["status"] == STATUS_RESOLVED}
        self.assertEqual(saved, {"General"})

    def test_the_subject_defaults_to_the_one_file_the_students_were_studying(self):
        first = self.course_files()[0]
        for i, q in enumerate(POD_QUESTIONS):
            ctx = AgentContext(vectorstore=self.vectorstore, db_dir=self.db, student_id=f"s{i}", llm=fake("unused"),
                               source_filenames=[first])
            ctx.session_id = f"sess-{i}"
            self.orch.handle(q, ctx)
        at = self.open_page()
        self.assertEqual(self.pod_subject_box(at).value, first)

    def test_the_queue_shrinks_and_the_page_shows_what_is_left(self):
        at = self.answer_pods()
        self.assertEqual(at.exception, [])
        self.assertEqual(len(self.group_markdowns(at)), 1)
        self.assertIn(DEADLINE, self.group_markdowns(at)[0])

    def test_the_answer_is_added_to_the_course_index_as_a_teacher_answer(self):
        self.answer_pods()
        index = load_vectorstore(self.db, get_embeddings())
        added = [d for d in index.docstore._dict.values() if d.metadata.get("source") == FACULTY_SOURCE_NAME]
        self.assertEqual(len(added), 1)
        self.assertIn(POD_ANSWER, added[0].page_content)

    def test_a_student_asking_again_now_gets_an_answer_that_cites_the_teachers_answer(self):
        self.answer_pods()
        index = load_vectorstore(self.db, get_embeddings())
        for question in [*POD_QUESTIONS, "What does a pod do in Kubernetes?"]:          # the last one was never asked before
            ctx = AgentContext(vectorstore=index, db_dir=self.db, student_id="new", llm=fake("A pod groups containers [Chunk 1]."))
            result = self.orch.handle(question, ctx).primary
            self.assertTrue(result.grounded, question)
            self.assertIn(FACULTY_SOURCE_NAME, [d.metadata.get("source") for d in result.sources], question)

    def test_other_unanswered_questions_stay_unanswered(self):
        self.answer_pods()
        index = load_vectorstore(self.db, get_embeddings())
        ctx = AgentContext(vectorstore=index, db_dir=self.db, student_id="new", llm=fake("unused"))
        self.assertFalse(self.orch.handle(DEADLINE, ctx).primary.grounded)

    def test_a_rebuild_of_the_index_keeps_the_answer(self):
        self.answer_pods()
        rebuilt = build_vectorstore(
            list(self.vectorstore.docstore._dict.values()) + faculty_documents(resolved_answers(self.db)), get_embeddings()
        )
        added = [d for d in rebuilt.docstore._dict.values() if d.metadata.get("source") == FACULTY_SOURCE_NAME]
        self.assertEqual(len(added), 1)


class EdgeCaseTests(GapsTestCase):
    def click_pod_button(self, at):
        keys = [b.key for b in at.text_area if b.key.startswith("answer_")]
        buttons = [b for b in at.button if b.label == "Add to knowledge base"]
        buttons[keys.index(self.pod_box(at).key)].click().run()

    def test_an_empty_answer_saves_nothing(self):
        self.queue_gaps()
        at = self.open_page()
        self.click_pod_button(at)
        self.assertIn("Write an answer first.", [w.value for w in at.warning])
        self.assertEqual(len(list_queries(status=STATUS_GAP, db_dir=self.db)), 4)
        index = load_vectorstore(self.db, get_embeddings())
        self.assertFalse([d for d in index.docstore._dict.values() if d.metadata.get("source") == FACULTY_SOURCE_NAME])

    def test_without_a_course_index_it_asks_for_one_and_saves_nothing(self):
        import shutil
        from rag_core.config import FAISS_INDEX_NAME

        self.queue_gaps()
        at = self.open_page()
        self.pod_box(at).set_value(POD_ANSWER)
        at.run()
        shutil.rmtree(self.db / FAISS_INDEX_NAME)
        self.assertFalse(vectorstore_exists(self.db))
        self.click_pod_button(at)
        self.assertIn("Build the index first (Course Material tab).", [e.value for e in at.error])
        self.assertEqual(len(list_queries(status=STATUS_GAP, db_dir=self.db)), 4)

    def test_an_answer_typed_while_a_new_similar_question_arrives_is_not_lost(self):
        """The teacher is typing; meanwhile a student asks one more pod question. Pressing the button must still
        save what was typed, not complain that the box is empty."""
        self.queue_gaps()
        at = self.open_page()
        self.pod_box(at).set_value(POD_ANSWER)
        at.run()
        self.student_asks("what is a pod in k8s", student="late")              # the group grows to four
        self.click_pod_button(at)
        self.assertNotIn("Write an answer first.", [w.value for w in at.warning])
        resolved = [r for r in list_queries(db_dir=self.db) if r["status"] == STATUS_RESOLVED]
        self.assertGreaterEqual(len(resolved), 3)
        self.assertTrue(all(r["faculty_answer"] == POD_ANSWER for r in resolved))
