"""
Tests for the Concepts to Revisit tab as a teacher sees it: a ranked list of topics, one detail panel, and a button
that shows what could be done about the topic only when it is pressed.

The real tab code is rendered on a temporary database holding a simulated class. Checked here:
  - the explanatory sentences, the "Agents involved" panel and the "Did it help?" section are gone;
  - suggestions are hidden until asked for, and are only worked out at that moment;
  - picking another topic changes the detail panel.

Run:  python -m unittest discover -s tests -v
"""

import builtins
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

from rag_core.advice import diagnose
from rag_core.confusion import compute_confusion
from test_confusion import Builder, students

ROOT = Path(__file__).resolve().parent.parent

PREFIX = '''
import builtins
import rag_core.config as _c
from pathlib import Path
_c.DB_DIR = Path(r"{db}")
import agents.orchestrator as _o
builtins._advice_calls = getattr(builtins, "_advice_calls", [])
if not hasattr(builtins, "_original_advise"):          # wrap once: the script re-runs on every interaction
    builtins._original_advise = _o.Orchestrator.advise_class

    def _counted(self, *args, **kwargs):
        builtins._advice_calls.append(kwargs.get("only"))
        return builtins._original_advise(self, *args, **kwargs)

    _o.Orchestrator.advise_class = _counted
'''


def app_source_for_page() -> str:
    """The app up to the sidebar, without its opening description (Streamlit would show that as text when wrapped)."""
    import re

    src = (ROOT / "app.py").read_text(encoding="utf-8")
    cut = src.rfind("# ----", 0, src.index("# Sidebar + routing"))
    return re.sub(r'^"""[\s\S]*?"""\n', "", src[:cut], count=1)


def build_database(db: Path) -> None:
    """A simulated class (several confusing topics, one merely popular) written into a temporary database."""
    from evaluate_intervention import simulate
    from rag_core.learning_log import create_quiz, record_attempt
    from rag_core.query_log import log_query

    from rag_core.embeddings import get_embeddings
    from rag_core.vectorstore import build_vectorstore, save_vectorstore

    # The drafting buttons only appear when a course index exists, so give the database a tiny one.
    save_vectorstore(build_vectorstore([
        Document(page_content="Concept 0 Topic is explained in this simulated slide with enough words to be indexed.",
                 metadata={"source": "notes.pdf", "page": 0, "section": "Concept 0 Topic"}),
    ], get_embeddings()), db)

    rows, quizzes, attempts, _oral, _iv = simulate(40, 0.8, 0.0, seed=2)
    for quiz in quizzes:
        create_quiz(quiz["topic"], "simulated", [], scope="class", db_dir=db)
    for r in rows:
        log_query(r["question"], True, r["top_score"], [Document(page_content="x", metadata=r["retrieved"][0])], db_dir=db,
                  student_id=r["student_id"], agent=r["agent"], session_id=r["session_id"], asked_at=r["asked_at"])
    for a in attempts:
        record_attempt(a["student_id"], a["quiz_id"], a["topic"], a["correct"], a["total"], a["answers"], db, a["taken_at"])


class ConceptsPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.db = Path(cls._tmp.name)
        build_database(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def open_page(self):
        from streamlit.testing.v1 import AppTest

        page = ROOT / "_tmp_concepts_page.py"
        page.write_text(PREFIX.format(db=self.db) + app_source_for_page() + "\n_render_concept_confusion()\n", encoding="utf-8")
        self.addCleanup(lambda: page.unlink(missing_ok=True))
        builtins._advice_calls = []
        at = AppTest.from_file(str(page), default_timeout=300)
        at.run()
        return at

    @staticmethod
    def button(at, label):
        found = [b for b in at.button if b.label == label]
        return found[0] if found else None

    def second_key(self, at):
        """The key of the second-ranked topic, worked out the same way the page ranks them."""
        from rag_core.confusion import load_and_compute
        return load_and_compute(self.db).concepts[1].key

    def test_it_opens_on_a_ranked_list_and_the_top_topics_detail(self):
        at = self.open_page()
        self.assertEqual(at.exception, [])
        options = at.radio(key="concept_pick").options
        self.assertGreaterEqual(len(options), 4)
        self.assertTrue(options[0].startswith("1. "))
        self.assertEqual(len(at.subheader), 1)                                   # one detail panel, for the first topic
        self.assertIn(at.subheader[0].value, options[0])

    def test_the_detail_shows_score_evidence_and_nothing_technical(self):
        at = self.open_page()
        self.assertTrue(any(m.value.startswith("Score **") and "out of 100" in m.value for m in at.markdown))
        table = next(df.value for df in at.dataframe)
        self.assertEqual(list(table.columns), ["Evidence", "What we saw"])
        self.assertGreaterEqual(len(table), 3)
        for df in at.dataframe:
            self.assertNotIn("Weight", df.value.columns)
            self.assertNotIn("Stays in top 3 when weights change", df.value.columns)

    def test_the_explanations_the_agents_panel_and_did_it_help_are_gone(self):
        at = self.open_page()
        self.button(at, "What could I do about this?").click().run()
        everything = " ".join(
            [c.value for c in at.caption] + [m.value for m in at.markdown] + [s.value for s in at.subheader]
            + [i.value for i in at.info] + [e.label for e in at.expander]
        )
        for removed in ("ranking aid", "A concept is a slide topic", "judgment calls", "Based on", "Did it help",
                        "Agents involved", "What you could do about it", "Log something you did", "rule-based",
                        "no AI model is involved", "Confidence: Low means"):
            at_position = everything.find(removed)
            self.assertEqual(at_position, -1, f"{removed!r} found in: ...{everything[max(at_position - 60, 0):at_position + 80]}...")

    def test_suggestions_are_hidden_until_the_button_is_pressed_and_worked_out_only_then(self):
        at = self.open_page()
        self.assertEqual(builtins._advice_calls, [])                              # nothing computed yet
        self.assertIsNotNone(self.button(at, "What could I do about this?"))
        self.assertFalse([m for m in at.markdown if m.value in ("**What students ask**", "**Draft for students**")])
        self.assertIsNone(self.button(at, "Draft a clarification note"))

        self.button(at, "What could I do about this?").click().run()
        self.assertEqual(at.exception, [])
        self.assertEqual(len(builtins._advice_calls), 1)
        self.assertIsNotNone(self.button(at, "Hide suggestions"))
        self.assertIsNone(self.button(at, "What could I do about this?"))
        self.assertIsNotNone(self.button(at, "Draft a clarification note"))        # the drafting tools live behind the button
        self.assertIsNotNone(self.button(at, "Write a remedial quiz"))
        self.assertTrue([m for m in at.markdown if m.value == "**Draft for students**"])

        self.button(at, "Hide suggestions").click().run()
        self.assertIsNone(self.button(at, "Draft a clarification note"))
        self.assertIsNotNone(self.button(at, "What could I do about this?"))

    def test_suggestions_are_worked_out_for_the_selected_topic_only(self):
        at = self.open_page()
        first_key = at.radio(key="concept_pick").value
        self.button(at, "What could I do about this?").click().run()
        self.assertEqual(builtins._advice_calls, [first_key])

    def test_picking_another_topic_changes_the_panel_and_starts_with_suggestions_hidden(self):
        at = self.open_page()
        self.button(at, "What could I do about this?").click().run()
        first_title = at.subheader[0].value
        second_key = self.second_key(at)
        at.session_state["concept_pick"] = second_key       # AppTest cannot drive a radio that has a display function
        at.run()
        self.assertEqual(at.exception, [])
        self.assertNotEqual(at.subheader[0].value, first_title)
        self.assertIsNotNone(self.button(at, "What could I do about this?"))
        self.assertIsNone(self.button(at, "Draft a clarification note"))

    def test_scoring_weights_sit_in_a_collapsed_section_at_the_bottom(self):
        at = self.open_page()
        labels = [e.label for e in at.expander]
        self.assertEqual(labels[-1], "Adjust scoring")
        self.assertEqual(len(at.slider), 7)
        self.assertEqual({s.key for s in at.slider}, {f"weight_{n}" for n in (
            "breadth", "repeat", "follow_up", "explain", "quiz", "oral", "persistence")})

    def test_changing_a_weight_re_ranks_without_error(self):
        at = self.open_page()
        before = at.radio(key="concept_pick").options
        for slider in at.slider:
            slider.set_value(100 if slider.key == "weight_quiz" else 0)
        at.run()
        self.assertEqual(at.exception, [])
        self.assertEqual(len(at.radio(key="concept_pick").options), len(before))

    def test_with_no_data_it_says_so_in_one_line(self):
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory() as empty:
            page = ROOT / "_tmp_concepts_empty_page.py"
            page.write_text(PREFIX.format(db=empty) + app_source_for_page() + "\n_render_concept_confusion()\n", encoding="utf-8")
            try:
                at = AppTest.from_file(str(page), default_timeout=120)
                at.run()
            finally:
                page.unlink(missing_ok=True)
        self.assertEqual(at.exception, [])
        self.assertEqual([i.value for i in at.info], ["No topics have enough data yet."])


class DiagnoseOnlyTests(unittest.TestCase):
    """The advisor can work out suggestions for one chosen topic instead of the top few."""

    def build(self):
        b = Builder()
        for j in range(4):
            for s in students(6):
                b.ask(s, f"Topic {j}", day=1 + j)
        return b, b.report()

    def test_only_returns_just_that_topic(self):
        b, report = self.build()
        target = report.concepts[2]
        advice = diagnose(report, b.rows, only=target.key)
        self.assertEqual([a.key for a in advice], [target.key])

    def test_only_works_for_a_topic_below_the_top_n(self):
        b, report = self.build()
        last = report.concepts[-1]
        self.assertEqual([a.key for a in diagnose(report, b.rows, top_n=1, only=last.key)], [last.key])

    def test_an_unknown_topic_gives_nothing(self):
        b, report = self.build()
        self.assertEqual(diagnose(report, b.rows, only="no such topic"), [])

    def test_without_only_the_top_n_are_returned_as_before(self):
        b, report = self.build()
        self.assertEqual(len(diagnose(report, b.rows, top_n=2)), 2)


if __name__ == "__main__":
    unittest.main()
