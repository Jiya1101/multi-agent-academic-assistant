"""
Tests for follow-up questions in the Ask section.

The promises checked here:
  - a follow-up that only makes sense after the previous question is rewritten into a standalone one, and the
    student is told how it was understood;
  - a message that already stands on its own is NEVER rewritten, even if it opens with "why" or "how";
  - when the model is unavailable or gives a poor rewrite, a plain join with the previous topic is used;
  - nothing is rewritten after a question the notes could not answer;
  - the teacher's log holds the standalone question (so grouping by meaning works) and keeps what was typed.

Run:  python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from agents import AgentContext, FollowUp, Turn, remember, resolve_follow_up, turn_from
from agents.conversation import (
    MAX_HISTORY,
    _join,
    last_answered_turn,
    looks_like_follow_up,
    rewrite_follow_up,
)
from rag_core.query_log import list_queries
from test_agents import AgentTestCase, CountingFake, fake

ACID_TURN = Turn(
    question="What is ACID?", standalone="What is ACID?", topic="ACID DATABASE CONSISTENCY MODEL",
    grounded=True, log_id=1, file="notes.pdf",
)
UNANSWERED_TURN = Turn(question="What is a pod?", standalone="What is a pod?", topic=None, grounded=False)


class Offline(FakeListChatModel):
    def _call(self, *args, **kwargs):
        raise ConnectionError("ollama is not running")


class FollowUpDetectionTests(unittest.TestCase):
    def test_continuation_words_and_pronouns_mark_a_follow_up(self):
        for message in [
            "And what about its disadvantages?", "Why is that?", "Give an example", "How does it work?",
            "What about BASE?", "Explain that more simply", "Can you give an example?", "Is it used in RDS?",
            "But why do they need it?", "More detail please",
        ]:
            self.assertTrue(looks_like_follow_up(message), message)

    def test_plain_new_questions_are_not_follow_ups(self):
        for message in [
            "What is ACID?", "What is the difference between ACID and BASE?", "Explain the ACID properties",
            "Define deadlock", "List the types of storage", "Quiz me on RDS",
        ]:
            self.assertFalse(looks_like_follow_up(message), message)

    def test_a_long_message_is_a_question_in_its_own_right(self):
        long = "Why does it matter that a database transaction either completes in full or leaves no trace at all when systems fail"
        self.assertGreater(len(long.split()), 18)
        self.assertFalse(looks_like_follow_up(long))

    def test_empty_messages_are_not_follow_ups(self):
        self.assertFalse(looks_like_follow_up(""))
        self.assertFalse(looks_like_follow_up("   "))


class HistoryTests(unittest.TestCase):
    def test_the_last_answered_turn_skips_questions_the_notes_could_not_answer(self):
        self.assertEqual(last_answered_turn([ACID_TURN, UNANSWERED_TURN]), ACID_TURN)
        self.assertIsNone(last_answered_turn([UNANSWERED_TURN]))
        self.assertIsNone(last_answered_turn([]))
        self.assertIsNone(last_answered_turn(None))

    def test_only_the_most_recent_turns_are_kept(self):
        history = []
        for i in range(MAX_HISTORY + 5):
            history = remember(history, Turn(f"q{i}", f"q{i}", "T", True))
        self.assertEqual(len(history), MAX_HISTORY)
        self.assertEqual(history[-1].question, f"q{MAX_HISTORY + 4}")

    def test_remembering_nothing_changes_nothing(self):
        self.assertEqual(remember([ACID_TURN], None), [ACID_TURN])


class RewriteTests(unittest.TestCase):
    def test_a_good_model_rewrite_is_used(self):
        text, how = rewrite_follow_up(
            "Why is it important?", ACID_TURN, fake("Why is the ACID model important for database transactions?")
        )
        self.assertEqual((text, how), ("Why is the ACID model important for database transactions?", "model"))

    def test_a_rewrite_that_drifted_from_the_topic_is_replaced_by_a_join(self):
        text, how = rewrite_follow_up("Why is it important?", ACID_TURN, fake("What is the capital of France today?"))
        self.assertEqual(how, "joined")
        self.assertIn("ACID DATABASE CONSISTENCY MODEL", text)

    def test_a_rewrite_that_just_repeats_the_message_is_replaced_by_a_join(self):
        self.assertEqual(rewrite_follow_up("Why is it important?", ACID_TURN, fake("Why is it important?"))[1], "joined")

    def test_an_empty_or_one_word_rewrite_is_replaced_by_a_join(self):
        self.assertEqual(rewrite_follow_up("Why is it important?", ACID_TURN, fake(""))[1], "joined")
        self.assertEqual(rewrite_follow_up("Why is it important?", ACID_TURN, fake("ACID"))[1], "joined")

    def test_an_unavailable_model_falls_back_to_a_join(self):
        text, how = rewrite_follow_up("Why is it important?", ACID_TURN, Offline(responses=["x"]))
        self.assertEqual(how, "joined")
        self.assertEqual(text, _join("Why is it important?", ACID_TURN))

    def test_the_join_names_the_topic_and_keeps_the_students_words(self):
        self.assertEqual(
            _join("What about its disadvantages?", ACID_TURN),
            "What about its disadvantages (about ACID DATABASE CONSISTENCY MODEL)?",
        )

    def test_a_model_label_and_quotes_are_stripped(self):
        text, how = rewrite_follow_up(
            "Why is it important?", ACID_TURN, fake('Question: "Why is the ACID model important for transactions?"')
        )
        self.assertEqual(how, "model")
        self.assertFalse(text.startswith(("Question", '"')))


class ResolveFollowUpTests(AgentTestCase):
    def ctx_with(self, history, llm=None):
        context = self.ctx(llm=llm)
        context.history = history
        return context

    def test_without_history_nothing_is_rewritten(self):
        result = resolve_follow_up("Why is it important?", self.ctx_with([], fake("never used")))
        self.assertEqual(result, FollowUp("Why is it important?"))

    def test_a_follow_up_is_rewritten_using_the_previous_turn(self):
        result = resolve_follow_up(
            "Why is it important?", self.ctx_with([ACID_TURN], fake("Why is the ACID model important for database transactions?"))
        )
        self.assertEqual(result.how, "model")
        self.assertEqual(result.previous, ACID_TURN)

    def test_a_message_that_stands_on_its_own_is_never_rewritten_and_the_model_is_not_called(self):
        model = CountingFake(responses=["unused"])
        message = "Why does ACID guarantee atomicity, consistency, isolation and durability for transactions?"
        result = resolve_follow_up(message, self.ctx_with([ACID_TURN], model))
        self.assertEqual((result.text, result.how), (message, "none"))
        self.assertEqual(model.calls, 0)

    def test_nothing_is_rewritten_after_a_question_the_notes_could_not_answer(self):
        model = CountingFake(responses=["unused"])
        result = resolve_follow_up("Why is it important?", self.ctx_with([UNANSWERED_TURN], model))
        self.assertEqual(result.how, "none")
        self.assertEqual(model.calls, 0)

    def test_a_new_plain_question_after_an_answer_is_left_alone(self):
        model = CountingFake(responses=["unused"])
        result = resolve_follow_up("What is OLTP?", self.ctx_with([ACID_TURN], model))
        self.assertEqual(result.how, "none")
        self.assertEqual(model.calls, 0)


class ConversationEndToEndTests(AgentTestCase):
    """A real two-question conversation through the Orchestrator, with a scripted model."""

    FIRST = "What is the difference between ACID and BASE?"
    FOLLOW = "Why is it important?"
    REWRITE = "Why is the ACID model important for database transactions?"

    def converse(self, answers_and_rewrite, history_after_first=True):
        context = self.ctx(llm=fake(*answers_and_rewrite), student_id="s1")
        context.session_id = "sess-1"
        first = self.orch.handle(self.FIRST, context)
        history = remember([], turn_from(first))
        context = self.ctx(llm=context.llm, student_id="s1")
        context.session_id = "sess-1"
        context.history = history if history_after_first else []
        return first, self.orch.handle(self.FOLLOW, context)

    def test_the_follow_up_is_answered_as_the_standalone_question_and_the_student_is_told(self):
        _, second = self.converse(["First answer.", self.REWRITE, "Second answer."])
        self.assertEqual(second.understood_as, self.REWRITE)
        self.assertEqual(second.primary.data["question"], self.REWRITE)
        self.assertEqual(second.message, self.FOLLOW)               # what the student typed is kept
        self.assertTrue(second.trace[0].startswith("Follow-up: understood as"))
        self.assertIn("rewritten by the model", second.trace[0])

    def test_the_log_holds_the_standalone_question_what_was_typed_and_the_real_link(self):
        first, _ = self.converse(["First answer.", self.REWRITE, "Second answer."])
        rows = list_queries(db_dir=self.db)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["question"], self.REWRITE)         # the professor sees a meaningful question
        self.assertEqual(rows[1]["original_question"], self.FOLLOW)
        self.assertEqual(rows[1]["follow_up_of"], rows[0]["id"])
        self.assertIsNone(rows[0]["original_question"])

    def test_without_the_conversation_the_same_message_is_answered_as_typed(self):
        _, second = self.converse(["First answer.", "Second answer."], history_after_first=False)
        self.assertIsNone(second.understood_as)
        self.assertEqual(second.primary.data["question"], self.FOLLOW)

    def test_an_unavailable_model_still_resolves_the_follow_up_by_joining_the_topic(self):
        context = self.ctx(llm=fake("First answer."), student_id="s1")
        first = self.orch.handle(self.FIRST, context)
        offline = self.ctx(llm=Offline(responses=["x"]), student_id="s1")
        offline.history = remember([], turn_from(first))
        second = self.orch.handle(self.FOLLOW, offline)
        self.assertIn("joined with the topic", second.trace[0])
        self.assertIn("(about ", second.understood_as)

    def test_a_turn_is_made_only_from_a_question_and_answer(self):
        first = self.orch.handle(self.FIRST, self.ctx(llm=fake("First answer.")))
        turn = turn_from(first)
        self.assertEqual((turn.question, turn.grounded, turn.file), (self.FIRST, True, "notes.pdf"))
        self.assertTrue(turn.topic)
        quiz = self.orch.handle("quiz me on RDS read replicas", self.ctx(llm=fake('{"questions": []}')))
        self.assertIsNone(turn_from(quiz))

    def test_a_turn_records_that_the_notes_could_not_answer(self):
        result = self.orch.handle("What is a Kubernetes pod?", self.ctx(llm=fake("unused")))
        turn = turn_from(result)
        self.assertFalse(turn.grounded)
        self.assertIsNone(last_answered_turn([turn]))


class AskPageTests(unittest.TestCase):
    """The real Ask screen: a thread of answers, a visible 'Understood as', and a way into a quiz."""

    REWRITE = "Why is the ACID model important for database transactions?"

    PAGE_PREFIX = '''
import rag_core.config as _c
from pathlib import Path
_c.DB_DIR = Path(r"{db}")
import rag_core.chain as _chain
import agents.conversation as _conv
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel


class _Scripted(FakeListChatModel):
    """Rewrites when asked to rewrite, otherwise gives a short answer."""

    def _call(self, messages, *args, **kwargs):
        if "rewrite a student" in messages[0].content.lower():
            return "{rewrite}"
        return "ACID means atomicity, consistency, isolation and durability for transactions [Chunk 1]."


_model = _Scripted(responses=["x"])
_chain.get_llm = lambda: _model
_conv.get_rewrite_llm = lambda: _model
'''

    PAGE_BODY = '''
from rag_core.vectorstore import build_vectorstore
_docs = [
    Document(page_content="The ACID model guarantees atomicity, consistency, isolation and durability for database transactions. "
                          "ACID matters because a failure in the middle of a transaction must never leave half-done changes.",
             metadata={"source": "notes.pdf", "page": 0, "section": "ACID DATABASE CONSISTENCY MODEL"}),
    Document(page_content="BASE stands for basically available, soft state and eventual consistency. BASE relaxes the ACID guarantees to scale.",
             metadata={"source": "notes.pdf", "page": 1, "section": "BASE DATABASE CONSISTENCY MODEL"}),
]
render_ask(build_vectorstore(_docs, _load_embeddings()), "student1")
'''

    def run_page(self, steps):
        import tempfile
        from streamlit.testing.v1 import AppTest

        root = Path(__file__).resolve().parent.parent
        src = (root / "app.py").read_text(encoding="utf-8")
        cut = src.rfind("# ----", 0, src.index("# Sidebar + routing"))
        with tempfile.TemporaryDirectory() as tmp:
            page = root / "_tmp_ask_page.py"
            page.write_text(
                self.PAGE_PREFIX.format(db=tmp, rewrite=self.REWRITE) + src[:cut] + "\n" + self.PAGE_BODY, encoding="utf-8"
            )
            try:
                at = AppTest.from_file(str(page), default_timeout=300)
                at.session_state["active_files"] = ["notes.pdf"]
                at.run()
                steps(at)
            finally:
                page.unlink(missing_ok=True)
            return at

    @staticmethod
    def ask(at, text):
        at.text_input(key="ask_message").set_value(text)
        [b for b in at.button if b.label == "Ask"][0].click().run()

    def test_a_follow_up_joins_the_thread_with_its_understanding_shown(self):
        def steps(at):
            self.ask(at, "What is the difference between ACID and BASE?")
            self.assertEqual(at.exception, [])
            self.assertEqual(len(at.session_state["chat"]), 1)
            self.assertFalse([c for c in at.caption if c.value.startswith("Understood as")])
            self.assertEqual(at.text_input(key="ask_message").value, "")          # the box is empty again
            self.ask(at, "Why is it important?")
            self.assertEqual(at.exception, [])
            self.assertEqual(len(at.session_state["chat"]), 2)
            self.assertEqual(at.session_state["chat"][1]["resp"].understood_as, self.REWRITE)
            captions = [c.value for c in at.caption]
            self.assertIn(f'Understood as: "{self.REWRITE}"', captions)
            labels = [e.label for e in at.expander]
            self.assertEqual([l for l in labels if l[:2] in ("2.", "1.")], ["2. Why is it important?", "1. What is the difference between ACID and BASE?"])
        self.run_page(steps)

    def test_the_conversation_can_be_reset(self):
        def steps(at):
            self.ask(at, "What is the difference between ACID and BASE?")
            [b for b in at.button if b.label == "New conversation"][0].click().run()
            self.assertNotIn("chat", at.session_state)
            self.assertNotIn("chat_history", at.session_state)
            self.ask(at, "Why is it important?")                                    # no history: answered as typed
            self.assertIsNone(at.session_state["chat"][0]["resp"].understood_as)
        self.run_page(steps)

    def test_an_answer_offers_a_quiz_on_its_topic(self):
        def steps(at):
            self.ask(at, "What is the difference between ACID and BASE?")
            test_me = [b for b in at.button if b.label.startswith("Test me on")]
            self.assertEqual(len(test_me), 1)
            test_me[0].click().run()
            self.assertEqual(at.session_state["main_nav"], "Quiz")
            self.assertEqual(at.session_state["quiz_request"]["file"], "notes.pdf")
        self.run_page(steps)

    def test_an_empty_message_asks_nothing(self):
        def steps(at):
            [b for b in at.button if b.label == "Ask"][0].click().run()
            self.assertEqual([w.value for w in at.warning], ["Please enter a message first."])
            self.assertNotIn("chat", at.session_state)
        self.run_page(steps)


if __name__ == "__main__":
    unittest.main()
