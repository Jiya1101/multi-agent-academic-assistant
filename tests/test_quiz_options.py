"""
Tests for choosing how many quiz questions to get and how hard they should be.

The AI model is replaced by a fake that records exactly what it was asked, so these check what the quiz generator
asks the model for (count per call, difficulty wording), how it copes when the model falls short or is offline, and
what the rule-based writer does for each difficulty. No Ollama is needed.

Run:  python -m unittest discover -s tests -v
"""

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from agents import AgentContext, ProgressTracker, QuizGenerator
from agents import quiz_generator
from agents.quiz_generator import (
    BATCH,
    DEFAULT_DIFFICULTY,
    LEVEL_NOTES,
    MAX_QUESTIONS,
    _definition_question,
    _heading_questions,
    mixed_fallback_quiz,
    normalise_level,
)
from rag_core.embeddings import get_embeddings
from rag_core.learning_log import record_attempt
from rag_core.vectorstore import build_vectorstore

TOPICS = [
    ("Mutual Exclusion", "Mutual exclusion is a rule that lets only one process use a shared resource at any moment. "
     "The rule prevents two processes from changing the same data at the same time. "
     "Most systems enforce it with locks that a process must hold before entering the critical section."),
    ("Deadlock", "Deadlock is a state in which a group of processes each wait for a resource held by another one. "
     "None of the waiting processes can ever continue, so the whole group is stuck for good. "
     "Systems avoid it by ordering resource requests or by detecting cycles in the wait graph."),
    ("Paging", "Paging is a memory scheme that splits memory into fixed size blocks called pages and frames. "
     "A page table maps each virtual page of a program to a frame in physical memory. "
     "A TLB = Translation Lookaside Buffer keeps recent translations so lookups stay fast."),
    ("Caching", "Caching is the practice of keeping recently used data in a small fast store near the processor. "
     "A hit returns the data quickly while a miss forces a slow read from the larger memory. "
     "The LRU = Least Recently Used policy removes the entry that has been idle the longest."),
    ("Load Balancing", "Load balancing is a technique that spreads incoming requests across several servers. "
     "It stops any single server from becoming a bottleneck and keeps the service available during failures. "
     "A DNS = Domain Name System entry can point one name at many balanced servers."),
    ("Hashing", "Hashing is a method that turns a key into a fixed size number used to find a bucket quickly. "
     "A good hash function spreads keys evenly so that few keys end up sharing a bucket. "
     "A CDN = Content Delivery Network uses hashing to decide which edge server holds a file."),
    ("Indexing", "Indexing is a way of building a separate sorted structure that lets a database find rows quickly. "
     "A query with a matching index reads a few pages instead of scanning the whole table. "
     "Every extra index makes writes slower because each one must be updated as well."),
    ("Scheduling", "Scheduling is the decision about which ready process receives the processor next. "
     "Round robin gives each process a short turn so that no process waits for very long. "
     "Priority scheduling serves important work first but can leave low priority work starving."),
]


def docs():
    return [
        Document(page_content=f"{title.upper()} {text}", metadata={"source": "course.pdf", "page": i, "section": title.upper()})
        for i, (title, text) in enumerate(TOPICS)
    ]


def headings():
    """Headings in the shape `major_headings` returns, with sentences long enough for statement questions."""
    out = []
    for i, (title, text) in enumerate(TOPICS):
        sentences = [s.strip() + "." for s in text.split(". ") if s.strip()]
        sentences = [s.replace("..", ".") for s in sentences]
        out.append({"heading": title, "sentences": sentences, "file": "course.pdf", "pages": [i + 1], "text": text})
    return out


def batch_json(prefix: str, n: int) -> str:
    return json.dumps({"questions": [
        {"question": f"{prefix} question number {i}?", "options": [f"{prefix} right {i}", f"{prefix} b {i}", f"{prefix} c {i}", f"{prefix} d {i}"],
         "answer_index": 0, "chunk": 1, "explanation": "The notes say so."}
        for i in range(n)
    ]})


class RecordingFake(FakeListChatModel):
    """A fake chat model that records every prompt it receives."""

    prompts: list = []

    def _call(self, messages, *args, **kwargs):
        self.prompts.append(messages[0].content if messages else "")
        return super()._call(messages, *args, **kwargs)


class Offline(FakeListChatModel):
    def _call(self, *args, **kwargs):
        raise ConnectionError("ollama is not running")


class QuizTestCase(unittest.TestCase):
    # The made-up questions in these tests are not backed by any slide, so the answer-key check is switched off for
    # them (KeyCheckTests turns it back on and tests it).
    check_keys = False

    @classmethod
    def setUpClass(cls):
        cls.vectorstore = build_vectorstore(docs(), get_embeddings())

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name)
        self._saved_threshold = quiz_generator.KEY_MIN_SUPPORT
        if not self.check_keys:
            quiz_generator.KEY_MIN_SUPPORT = 0.0

    def tearDown(self):
        quiz_generator.KEY_MIN_SUPPORT = self._saved_threshold
        self._tmp.cleanup()

    def ctx(self, llm, student_id=None):
        return AgentContext(vectorstore=self.vectorstore, db_dir=self.db, student_id=student_id, llm=llm)

    def quiz(self, llm, n=5, level="medium", **kw):
        return QuizGenerator().run_from_material(self.ctx(llm), n_questions=n, difficulty=level, **kw)


class QuestionCountTests(QuizTestCase):
    def test_the_requested_number_is_written_and_reported(self):
        result = self.quiz(RecordingFake(responses=[batch_json("a", 3)]), n=3)
        self.assertEqual((result.data["requested"], result.data["written"], len(result.data["questions"])), (3, 3, 3))

    def test_more_than_the_model_writes_at_once_is_asked_for_in_batches(self):
        model = RecordingFake(responses=[batch_json("a", 5), batch_json("b", 5), batch_json("c", 2)])
        model.prompts = []
        result = self.quiz(model, n=12)
        self.assertEqual(result.data["written"], 12)
        self.assertEqual(result.data["from_model"], 12)
        counts = [int(p.split("Write exactly ")[1].split(" ")[0]) for p in model.prompts]
        self.assertEqual(counts, [5, 5, 2])                       # never more than BATCH per call
        self.assertLessEqual(max(counts), BATCH)

    def test_questions_are_not_repeated_across_batches(self):
        model = RecordingFake(responses=[batch_json("a", 5), batch_json("a", 5), batch_json("b", 5)])
        model.prompts = []
        texts = [q["question"] for q in self.quiz(model, n=10).data["questions"]]
        self.assertEqual(len(texts), len(set(texts)))

    def test_asking_for_too_many_or_too_few_is_kept_in_range(self):
        self.assertEqual(self.quiz(Offline(responses=["x"]), n=100).data["requested"], MAX_QUESTIONS)
        self.assertEqual(self.quiz(Offline(responses=["x"]), n=0).data["requested"], 1)

    def test_questions_the_student_already_saw_are_not_repeated(self):
        model = RecordingFake(responses=[batch_json("a", 5), batch_json("b", 5)])
        model.prompts = []
        seen = [f"a question number {i}?" for i in range(5)]
        texts = [q["question"] for q in self.quiz(model, n=5, avoid=seen).data["questions"]]
        self.assertFalse(set(texts) & set(seen))

    def test_every_question_names_the_slide_it_came_from(self):
        for q in self.quiz(RecordingFake(responses=[batch_json("a", 5)]), n=5).data["questions"]:
            self.assertEqual(q["source"]["file"], "course.pdf")
            self.assertTrue(q["source"]["slide"])


DYNAMO_SLIDE = (
    "AMAZON DYNAMO DB BENEFITS Fully managed Supports multi-region deployment Multi-master deployment Fine-grained "
    "identity and access control Seamless integration with IAM security In-memory caching for fast retrieval "
    "Supports ACID transactions Encrypts all data by default"
)
AURORA_SLIDE = (
    "AMAZON AURORA FEATURES Serverless configuration Aurora Serverless is a configuration of Aurora that offers "
    "auto-scaling features on-demand. With this configuration the database can start up, shut down, and adjust its capacity."
)
GSI_SLIDE = (
    "Amazon Dynamo DB supports two different kinds of indexes. A global secondary index is an index with a partition "
    "and sort key that can be different from those on the table. A table can only have one local secondary index, "
    "but you can have multiple global secondary indexes."
)


def q(question, options, keyed):
    return {"question": question, "options": options, "answer_index": keyed}


class KeyIsBackedTests(unittest.TestCase):
    """Real cases from a Hard quiz written by Llama 3 on Module 4.pdf, checked against the slide text."""

    def test_a_not_question_whose_four_options_are_all_on_the_slide_is_rejected(self):
        wrong_key = q("Which of the following is NOT a benefit of Amazon Dynamo DB?",
                      ["Supports multi-region deployment", "Encrypts all data by default", "Fully managed", "Supports ACID transactions"], 1)
        self.assertFalse(quiz_generator.key_is_backed(wrong_key, DYNAMO_SLIDE))

    def test_the_same_not_question_with_a_genuinely_absent_option_is_kept(self):
        fixed = q("Which of the following is NOT a benefit of Amazon Dynamo DB?",
                  ["Supports multi-region deployment", "Runs entirely offline on a phone", "Fully managed", "Supports ACID transactions"], 1)
        self.assertTrue(quiz_generator.key_is_backed(fixed, DYNAMO_SLIDE))

    def test_a_keyed_answer_the_slide_does_say_is_kept(self):
        good = q("How does Aurora Serverless differ from a standard database?",
                 ["It is a relational database that uses SQL", "It is a database that does not use SQL", "It is a cloud-based RDBMS",
                  "It is a configuration of Aurora that offers auto-scaling features on-demand"], 3)
        self.assertTrue(quiz_generator.key_is_backed(good, AURORA_SLIDE))

    def test_a_keyed_answer_the_slide_never_says_is_rejected(self):
        unbacked = q("Why would you use a Global Secondary Index instead of a Local Secondary Index?",
                     ["Because GSI is more expensive", "Because GSI is more complex", "Because GSI can handle larger data sizes",
                      "Because GSI is more suitable for eventual consistency"], 3)
        self.assertFalse(quiz_generator.key_is_backed(unbacked, GSI_SLIDE))

    def test_a_lowercase_not_in_an_ordinary_question_does_not_make_it_a_not_question(self):
        ordinary = q("Why does the table not need a sort key?", ["Because a global secondary index has a partition key",
                     "Because it is cheaper", "Because it is faster", "Because it is older"], 0)
        self.assertTrue(quiz_generator.key_is_backed(ordinary, GSI_SLIDE))


class KeyCheckInTheQuizTests(QuizTestCase):
    check_keys = True

    def test_a_model_question_with_an_unbacked_answer_is_discarded_and_replaced(self):
        # All 8 chunks are in the batch for a request of 5, so chunk 1 is the "Mutual Exclusion" slide.
        backed = {"question": "What does mutual exclusion let a process do?", "answer_index": 0, "chunk": 1, "explanation": "x",
                  "options": ["Use a shared resource while no other process does", "Paint the screen", "Print a page", "Open a socket"]}
        unbacked = {"question": "Why is mutual exclusion useful?", "answer_index": 0, "chunk": 1, "explanation": "x",
                    "options": ["Because rainbows follow storms", "Because it is a rule", "Because it is fast", "Because it is old"]}
        result = self.quiz(RecordingFake(responses=[json.dumps({"questions": [backed, unbacked]})]), n=5)
        texts = [x["question"] for x in result.data["questions"]]
        self.assertIn(backed["question"], texts)
        self.assertNotIn(unbacked["question"], texts)
        self.assertEqual(result.data["discarded"], 1)
        self.assertEqual(result.data["written"], 5)                   # topped up by the rule-based writer
        self.assertEqual(result.data["from_model"], 1)

    def test_nothing_is_discarded_when_every_answer_is_backed(self):
        good = [{"question": f"What is the point of rule {i}?", "answer_index": 0, "chunk": 1, "explanation": "x",
                 "options": ["Only one process uses a shared resource at any moment", f"Option b {i}", f"Option c {i}", f"Option d {i}"]}
                for i in range(5)]
        result = self.quiz(RecordingFake(responses=[json.dumps({"questions": good})]), n=5)
        self.assertEqual((result.data["discarded"], result.data["from_model"]), (0, 5))


class ModelFallsShortTests(QuizTestCase):
    def test_when_the_model_supplies_too_few_the_rule_based_writer_tops_up(self):
        result = self.quiz(RecordingFake(responses=[batch_json("a", 2)]), n=5)
        self.assertEqual(result.data["written"], 5)
        self.assertEqual(result.data["from_model"], 2)

    def test_when_the_model_is_offline_the_whole_quiz_is_rule_based(self):
        result = self.quiz(Offline(responses=["x"]), n=5)
        self.assertEqual((result.data["written"], result.data["from_model"]), (5, 0))
        for q in result.data["questions"]:
            self.assertEqual(len(q["options"]), 4)
            self.assertEqual(len({o.lower() for o in q["options"]}), 4)
            self.assertIn("source", q)

    def test_a_model_that_only_returns_junk_still_gives_a_quiz(self):
        result = self.quiz(RecordingFake(responses=["not json at all"]), n=4)
        self.assertEqual(result.data["written"], 4)
        self.assertEqual(result.data["from_model"], 0)


class DifficultyTests(QuizTestCase):
    def prompt_for(self, level):
        model = RecordingFake(responses=[batch_json("a", 5)])
        model.prompts = []
        self.quiz(model, n=5, level=level)
        return model.prompts[0]

    def test_each_level_puts_its_own_wording_into_the_model_prompt(self):
        self.assertIn("Difficulty: EASY.", self.prompt_for("easy"))
        self.assertIn("Difficulty: MEDIUM.", self.prompt_for("medium"))
        self.assertIn("Difficulty: HARD.", self.prompt_for("hard"))
        self.assertNotIn("Difficulty: HARD.", self.prompt_for("easy"))

    def test_hard_asks_for_plausible_look_alike_wrong_answers_and_reasoning_questions(self):
        self.assertIn("plausible", LEVEL_NOTES["hard"])
        self.assertIn("NOT", LEVEL_NOTES["hard"])
        self.assertIn("At least half", LEVEL_NOTES["hard"])          # a concrete quota: a bare "make it hard" was ignored
        self.assertIn("Do not write plain", LEVEL_NOTES["hard"])

    def test_an_unknown_level_becomes_the_default(self):
        self.assertEqual(normalise_level("extreme"), DEFAULT_DIFFICULTY)
        self.assertEqual(normalise_level(None), DEFAULT_DIFFICULTY)
        self.assertEqual(normalise_level(" HARD "), "hard")
        self.assertIn("Difficulty: MEDIUM.", self.prompt_for("extreme"))

    def test_the_level_is_saved_with_the_quiz_and_each_question(self):
        result = self.quiz(RecordingFake(responses=[batch_json("a", 5)]), n=5, level="hard")
        self.assertEqual(result.data["difficulty"], "hard")
        self.assertTrue(all(q["level"] == "hard" for q in result.data["questions"]))
        from rag_core.learning_log import get_quiz
        self.assertIn("(hard)", get_quiz(result.data["quiz_id"], self.db)["prompt"])

    def test_a_quiz_on_a_topic_keeps_the_targeted_prompt_prefix(self):
        """Progress tracking recognises a topic test by its prompt starting with 'Quiz on '."""
        result = self.quiz(RecordingFake(responses=[batch_json("a", 5)]), n=5, level="easy", focus="Deadlock")
        from rag_core.learning_log import get_quiz
        self.assertTrue(get_quiz(result.data["quiz_id"], self.db)["prompt"].startswith("Quiz on "))
        self.assertEqual(result.data["topic"], "Deadlock")


class RuleBasedDifficultyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = docs()
        cls.headings = headings()

    def quiz(self, level, n=3, seed=1):
        return mixed_fallback_quiz(self.chunks, self.headings, n, random.Random(seed), level=level)

    def test_easy_uses_recognition_questions_not_true_statement_questions(self):
        for seed in range(6):
            stems = [q["question"] for q in self.quiz("easy", seed=seed)]
            self.assertFalse(any(s.startswith("Which statement is true") for s in stems), stems)

    def test_hard_leans_on_true_statement_questions(self):
        for seed in range(6):
            stems = [q["question"] for q in self.quiz("hard", seed=seed)]
            self.assertTrue(any(s.startswith("Which statement is true") for s in stems), stems)

    def test_a_short_pool_is_topped_up_from_the_other_types(self):
        quiz = mixed_fallback_quiz(self.chunks, [], 4, random.Random(1), level="easy")   # no headings: no topic/statement
        self.assertGreaterEqual(len(quiz), 3)

    def test_hard_picks_the_most_look_alike_wrong_definitions(self):
        item = {"term": "Cache", "verb": "is", "rest": "a small fast store near the processor holding recent data", "chunk": 1,
                "sentence": "Cache is a small fast store near the processor holding recent data."}
        # Much longer than the right answer, so a writer that only matches lengths would never choose it.
        look_alike_rest = "a small fast store placed near the disk controller that holds recent writes before they reach the slower disk"
        look_alike = {"term": "Buffer", "verb": "is", "rest": look_alike_rest, "chunk": 2, "sentence": "x"}
        far = [{"term": f"T{i}", "verb": "is", "rest": f"an unrelated idea about {w} that has nothing in common", "chunk": 3 + i, "sentence": "x"}
               for i, w in enumerate(["routing", "painting", "voltage"])]
        wanted = look_alike_rest[:1].upper() + look_alike_rest[1:] + "."
        for seed in range(5):
            hard = _definition_question(item, [item, look_alike, *far], random.Random(seed), close=True)
            plain = _definition_question(item, [item, look_alike, *far], random.Random(seed), close=False)
            self.assertIn(wanted, hard["options"])
            self.assertNotIn(wanted, plain["options"])      # the ordinary writer matches lengths and leaves it out

    def test_hard_statement_questions_use_the_most_related_other_slides(self):
        questions = _heading_questions(self.headings, random.Random(2), close=True)
        statement = next(q for q in questions if q["kind"] == "statement")
        wrong = [o for i, o in enumerate(statement["options"]) if i != statement["answer_index"]]
        self.assertEqual(len(wrong), 3)
        self.assertNotIn(statement["options"][statement["answer_index"]], wrong)


class GarbledHeadingTests(unittest.TestCase):
    def test_headings_run_together_by_the_pdf_are_not_used_as_answer_options(self):
        from agents.quiz_generator import _garbled_heading
        for bad in ["XXPresentation title108", "DB USAGE MODELAnnual Revenue Growth", "AWS DATABASE SERVICESModule 4", "Slide12 notes"]:
            self.assertTrue(_garbled_heading(bad), bad)
        for fine in ["Amazon RDS Benefits", "OLTP VS OLAP", "ACID Database Consistency MODEL", "Spark Architecture", "Module 4"]:
            self.assertFalse(_garbled_heading(fine), fine)

    def test_a_which_topic_question_never_offers_a_garbled_heading(self):
        mess = headings() + [
            {"heading": "XXPresentation title108", "sentences": headings()[0]["sentences"], "file": "c.pdf", "pages": [9], "text": ""},
            {"heading": "DB USAGE MODELAnnual Revenue", "sentences": headings()[1]["sentences"], "file": "c.pdf", "pages": [10], "text": ""},
        ]
        for seed in range(8):
            for q in _heading_questions(mess, random.Random(seed)):
                self.assertFalse(any("XXPresentation" in o or "MODELAnnual" in o for o in q["options"]), q["options"])

    def test_too_few_clean_headings_means_no_heading_questions(self):
        few = [{"heading": "XXPresentation title108", "sentences": ["A sentence that is long enough to be used here."] * 3,
                "file": "c.pdf", "pages": [1], "text": ""}] * 6
        self.assertEqual(_heading_questions(few, random.Random(1)), [])


class AdaptiveDifficultyTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def suggest(self, student="s1"):
        return ProgressTracker().suggested_difficulty(AgentContext(vectorstore=None, db_dir=self.db, student_id=student))

    def scores(self, *results, student="s1"):
        for i, correct in enumerate(results):
            record_attempt(student, i + 1, "T", correct, 10, [], self.db)

    def test_no_student_or_no_history_starts_at_medium_and_says_why(self):
        level, why = self.suggest(student=None)
        self.assertEqual(level, "medium")
        self.assertIn("no quiz history", why)
        self.assertEqual(self.suggest()[0], "medium")

    def test_strong_recent_results_give_hard(self):
        self.scores(9, 8, 9)
        level, why = self.suggest()
        self.assertEqual(level, "hard")
        self.assertIn("87%", why)

    def test_weak_recent_results_give_easy(self):
        self.scores(3, 4, 2)
        self.assertEqual(self.suggest()[0], "easy")

    def test_middling_results_give_medium(self):
        self.scores(6, 7, 6)
        self.assertEqual(self.suggest()[0], "medium")

    def test_only_the_last_three_quizzes_count(self):
        self.scores(1, 1, 1, 9, 9, 9)                            # weak long ago, strong lately
        self.assertEqual(self.suggest()[0], "hard")

    def test_another_students_results_do_not_leak_in(self):
        self.scores(9, 9, 9, student="other")
        self.assertEqual(self.suggest("s1")[0], "medium")


class QuizControlsPageTests(unittest.TestCase):
    """The real quiz controls render, and 'Adapt to me' explains its choice."""

    def page(self, tmp: Path, body: str):
        root = Path(__file__).resolve().parent.parent
        src = (root / "app.py").read_text(encoding="utf-8")
        cut = src.rfind("# ----", 0, src.index("# Sidebar + routing"))
        path = root / "_tmp_quiz_controls_page.py"
        path.write_text(
            "import rag_core.config as _c\nfrom pathlib import Path\n_c.DB_DIR = Path(r'" + str(tmp) + "')\n" + src[:cut] + "\n" + body,
            encoding="utf-8",
        )
        return path

    def test_controls_show_a_count_slider_and_four_difficulty_choices(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as tmp:
            path = self.page(Path(tmp), "_quiz_options(None, 'student1')\n")
            try:
                at = AppTest.from_file(str(path), default_timeout=120)
                at.run()
            finally:
                path.unlink(missing_ok=True)
            self.assertEqual(at.exception, [])
            slider = at.slider(key="quiz_count")
            self.assertEqual((slider.min, slider.max, slider.value), (1, MAX_QUESTIONS, 5))
            self.assertEqual(at.radio(key="quiz_level_choice").options, ["Easy", "Medium", "Hard", "Adapt to me"])
            self.assertTrue(any("minute" in c.value for c in at.caption))

    def test_adapt_to_me_explains_the_level_it_chose(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as tmp:
            for i, correct in enumerate((9, 9, 8)):
                record_attempt("student1", i + 1, "T", correct, 10, [], tmp)
            path = self.page(Path(tmp), "_quiz_options(None, 'student1')\n")
            try:
                at = AppTest.from_file(str(path), default_timeout=120)
                at.run()
                at.radio(key="quiz_level_choice").set_value("Adapt to me").run()
            finally:
                path.unlink(missing_ok=True)
            self.assertEqual(at.exception, [])
            self.assertTrue(any(c.value.startswith("Hard, because you got 87%") for c in at.caption), [c.value for c in at.caption])


class QuizScreenEndToEndTests(unittest.TestCase):
    """Clicking Generate quiz on the real quiz screen uses the chosen count and difficulty."""

    PREFIX = '''
import sys
sys.path.insert(0, r"{tests}")
import rag_core.config as _c
from pathlib import Path
_c.DB_DIR = Path(r"{db}")
import json
import agents.quiz_generator as _qg
from langchain_core.language_models.fake_chat_models import FakeListChatModel


class _Backed(FakeListChatModel):
    """Writes the number of questions it is asked for, each keyed to a sentence that is on the first slide."""

    def _call(self, messages, *args, **kwargs):
        text = messages[0].content
        n = int(text.split("Write exactly ")[1].split(" ")[0])
        level = "HARD" if "Difficulty: HARD" in text else "EASY" if "Difficulty: EASY" in text else "MEDIUM"
        return json.dumps({{"questions": [
            {{"question": f"{{level}} question {{i}} about the first rule?", "answer_index": 0, "chunk": 1, "explanation": "x",
              "options": ["Only one process uses a shared resource at any moment", f"b{{i}}", f"c{{i}}", f"d{{i}}"]}}
            for i in range(n)]}})


_qg.get_json_llm = lambda: _Backed(responses=["x"])
'''

    BODY = '''
from test_quiz_options import docs
from rag_core.vectorstore import build_vectorstore
render_quiz_builder(build_vectorstore(docs(), _load_embeddings()), "student1")
'''

    def test_the_chosen_count_and_difficulty_reach_the_quiz(self):
        import tempfile
        from streamlit.testing.v1 import AppTest

        root = Path(__file__).resolve().parent.parent
        src = (root / "app.py").read_text(encoding="utf-8")
        cut = src.rfind("# ----", 0, src.index("# Sidebar + routing"))
        with tempfile.TemporaryDirectory() as tmp:
            page = root / "_tmp_quiz_screen_page.py"
            page.write_text(
                self.PREFIX.format(tests=str(Path(__file__).resolve().parent), db=tmp) + src[:cut] + "\n" + self.BODY,
                encoding="utf-8",
            )
            try:
                at = AppTest.from_file(str(page), default_timeout=300)
                at.session_state["active_files"] = ["course.pdf"]
                at.run()
                at.slider(key="quiz_count").set_value(5)
                at.radio(key="quiz_level_choice").set_value("Hard").run()
                [b for b in at.button if b.label == "Generate quiz"][0].click().run()
                self.assertEqual(at.exception, [])
                picks = [r for r in at.radio if str(r.key).startswith("pick_")]
                captions = [c.value for c in at.caption]
                markdown = [m.value for m in at.markdown]
            finally:
                page.unlink(missing_ok=True)
        self.assertEqual(len(picks), 5)                                            # five questions on screen
        self.assertTrue(any(c == "Difficulty: Hard" for c in captions), captions)
        self.assertTrue(any("HARD question" in m for m in markdown))               # the model was told it was a Hard quiz


if __name__ == "__main__":
    unittest.main()
