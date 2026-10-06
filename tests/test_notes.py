"""
Tests for note generation: the Note Generator agent, its filters and flags, the
PDF export, the notes route, and topic listing.

Run:  python -m unittest discover -s tests -v
"""

import io
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pypdf import PdfReader

from agents.note_generator import (
    NO_EXAMPLE,
    NoteGenerator,
    novel_words,
    parse_notes_json,
    prettify_heading,
    support,
    _stems,
)
from agents.routing import extract_notes_topics, route_message
from langchain_core.documents import Document
from rag_core.pdf_export import CHECK_LEGEND, notes_to_pdf, pdf_safe
from rag_core.query_log import list_queries
from rag_core.vectorstore import build_vectorstore, list_indexed_topics
from rag_core.embeddings import get_embeddings
from test_agents import AgentTestCase, fake


def notes_json(**overrides) -> str:
    payload = {
        "heading": "ACID Model",
        "definition": "ACID is a set of rules that keep database transactions safe.",
        "key_points": [
            "ACID guarantees atomicity, consistency, isolation and durability",
            "These guarantees apply to database transactions",
        ],
        "use_case": "Read replicas let Amazon RDS serve read traffic from copies of the database.",
        "chunks": [1],
    }
    payload.update(overrides)
    return json.dumps(payload)


class ParsingTests(unittest.TestCase):
    def test_accepts_valid(self):
        parsed = parse_notes_json(notes_json(), n_chunks=2)
        self.assertEqual(parsed["heading"], "ACID Model")
        self.assertEqual(len(parsed["key_points"]), 2)
        self.assertEqual(parsed["chunks"], [1])

    def test_rejects_unusable_output(self):
        for raw in ["not json", "", "null", "[]", json.dumps({"definition": "", "key_points": ["x"]}),
                    json.dumps({"definition": "d", "key_points": "not a list"}),
                    json.dumps({"definition": "d", "key_points": []}),
                    json.dumps({"definition": "d", "key_points": ["", "  "]})]:
            with self.subTest(raw=raw):
                self.assertIsNone(parse_notes_json(raw, 2))

    def test_limits_bullets_and_ignores_bad_chunk_numbers(self):
        parsed = parse_notes_json(notes_json(key_points=[f"point {i}" for i in range(12)], chunks=[0, 1, 9, True, "2"]), 3)
        self.assertEqual(len(parsed["key_points"]), 6)
        self.assertEqual(parsed["chunks"], [1])

    def test_missing_use_case_becomes_the_no_example_message(self):
        raw = json.dumps({"definition": "d", "key_points": ["p"], "chunks": [1]})
        self.assertEqual(parse_notes_json(raw, 1)["use_case"], NO_EXAMPLE)


class HelperTests(unittest.TestCase):
    def test_support_and_novel_words(self):
        context = _stems("Read replicas let Amazon RDS serve read traffic from copies of the database")
        self.assertEqual(support("Read replicas serve read traffic", context), 1.0)
        self.assertEqual(novel_words("Read replicas serve read traffic", context), 0)
        self.assertLess(support("Kubernetes schedules containers across clusters", context), 0.2)
        self.assertEqual(support("", context), 1.0)

    def test_stems_match_plurals(self):
        self.assertEqual(_stems("replica"), _stems("replicas"))

    def test_prettify_heading(self):
        self.assertEqual(prettify_heading("OLTP VS OLAP SYSTEMS"), "OLTP vs OLAP Systems")
        self.assertEqual(prettify_heading("AMAZON RDS BENEFITS"), "Amazon RDS Benefits")
        self.assertEqual(prettify_heading("ACID Model"), "ACID Model")  # not all caps: untouched


class NoteGeneratorTests(AgentTestCase):
    def run_notes(self, *responses, topics=("ACID",), **kwargs):
        llm = fake(*responses)
        result = NoteGenerator().run("notes", self.ctx(llm), topics=list(topics), **kwargs)
        return result, llm

    def test_builds_a_structured_section_with_sources(self):
        result, _ = self.run_notes(notes_json())
        self.assertEqual(result.kind, "notes")
        self.assertTrue(result.grounded)
        section = result.data["sections"][0]
        self.assertEqual(section["heading"], "ACID Model")
        self.assertEqual(len(section["key_points"]), 2)
        self.assertEqual(section["sources"][0]["file"], "notes.pdf")
        self.assertEqual(result.data["files"], ["notes.pdf"])
        self.assertEqual(result.data["dropped"], 0)

    def test_unsupported_key_point_is_dropped_and_counted(self):
        bad = "Kubernetes schedules containers across clusters automatically"
        result, _ = self.run_notes(notes_json(key_points=["ACID guarantees atomicity, consistency, isolation and durability", bad]))
        section = result.data["sections"][0]
        self.assertNotIn(bad, section["key_points"])
        self.assertEqual(result.data["dropped"], 1)

    def test_section_with_no_supported_points_counts_as_failed(self):
        result, _ = self.run_notes(notes_json(key_points=["Kubernetes schedules containers across clusters"]))
        self.assertEqual(result.data["sections"], [])
        self.assertEqual(result.data["failed"], ["ACID"])

    def test_unsupported_use_case_is_replaced(self):
        result, _ = self.run_notes(notes_json(use_case="Airlines use it for quantum encryption of passenger luggage"))
        self.assertEqual(result.data["sections"][0]["use_case"], NO_EXAMPLE)

    def test_made_up_detail_in_a_mostly_supported_point_is_flagged_not_dropped(self):
        point = "ACID guarantees atomicity and isolation for secure banking"
        result, _ = self.run_notes(notes_json(key_points=["ACID guarantees atomicity, consistency, isolation and durability", point]))
        section = result.data["sections"][0]
        self.assertIn(point, section["key_points"])
        self.assertEqual(section["check"]["key_points"], [1])

    def test_clean_section_has_nothing_flagged(self):
        result, _ = self.run_notes(notes_json())
        self.assertEqual(result.data["sections"][0]["check"], {"key_points": [], "use_case": False})

    def test_uncovered_topic_is_skipped_without_calling_the_model_and_logged_as_a_gap(self):
        result, llm = self.run_notes(notes_json(), topics=("Kubernetes pods",))
        self.assertEqual(llm.calls, 0)
        self.assertEqual(result.data["skipped"], ["Kubernetes pods"])
        self.assertFalse(result.grounded)
        gaps = [r for r in list_queries(db_dir=self.db) if r["status"] == "gap"]
        self.assertEqual([g["question"] for g in gaps], ["Notes on: Kubernetes pods"])

    def test_retries_once_on_invalid_output_then_gives_up(self):
        result, llm = self.run_notes("garbage", notes_json())
        self.assertEqual(len(result.data["sections"]), 1)
        self.assertEqual(llm.calls, 2)
        result, llm = self.run_notes("garbage", "worse")
        self.assertEqual(result.data["failed"], ["ACID"])
        self.assertEqual(llm.calls, 2)

    def test_several_topics_and_progress_callback(self):
        events = []
        result, _ = self.run_notes(notes_json(), topics=("ACID", "BASE"), on_progress=lambda i, n, t: events.append((i, n, t)))
        self.assertEqual(len(result.data["sections"]), 2)
        self.assertEqual(events, [(0, 2, "ACID"), (1, 2, "BASE"), (2, 2, "")])

    def test_mixed_request_keeps_covered_topics_and_reports_the_skipped_one(self):
        result, _ = self.run_notes(notes_json(), topics=("ACID", "Kubernetes pods"))
        self.assertEqual(len(result.data["sections"]), 1)
        self.assertEqual(result.data["skipped"], ["Kubernetes pods"])
        self.assertTrue(result.grounded)


class PdfTests(unittest.TestCase):
    SECTION = {
        "topic": "ACID", "heading": "ACID Model",
        "definition": "ACID keeps transactions safe — it’s used by banks & shops.",
        "key_points": ["Atomicity: all or nothing.", "Data stays valid <before> and after.", "Saved data survives a crash."],
        "use_case": "A bank transfer must not lose money.",
        "check": {"key_points": [1], "use_case": False},
        "sources": [{"file": "Module 4.pdf", "page": 15}, {"file": "Module 4.pdf", "page": 16}],
    }

    def text_of(self, data: bytes) -> str:
        return "\n".join(page.extract_text() for page in PdfReader(io.BytesIO(data)).pages)

    def test_valid_pdf_with_all_parts(self):
        data = notes_to_pdf([self.SECTION], "Study Notes: ACID", ["Module 4.pdf"])
        self.assertTrue(data.startswith(b"%PDF"))
        text = self.text_of(data)
        for needle in ["Study Notes: ACID", "ACID Model", "Definition", "Key points", "Use case",
                       "Atomicity: all or nothing.", "Source: Module 4.pdf, pages 15, 16", "Page 1"]:
            self.assertIn(needle, text)

    def test_special_characters_are_safe(self):
        text = self.text_of(notes_to_pdf([self.SECTION], "T"))
        self.assertIn("banks & shops", text)
        self.assertIn("<before>", text)
        self.assertNotIn("—", text)  # em dash replaced with a plain hyphen

    def test_pdf_safe_replaces_unsupported_characters(self):
        self.assertEqual(pdf_safe("a’b"), "a'b")
        self.assertEqual(pdf_safe("x中y"), "x?y")

    def test_flagged_statements_are_marked_with_a_legend(self):
        text = self.text_of(notes_to_pdf([self.SECTION], "T"))
        self.assertIn("Data stays valid <before> and after. *", text)
        self.assertIn(CHECK_LEGEND.split(".")[0], text)

    def test_no_legend_when_nothing_is_flagged(self):
        clean = {**self.SECTION, "check": {"key_points": [], "use_case": False}}
        self.assertNotIn("Check it before relying", self.text_of(notes_to_pdf([clean], "T")))

    def test_bullets_use_a_real_font_glyph(self):
        reader = PdfReader(io.BytesIO(notes_to_pdf([self.SECTION], "T")))
        fonts = {str(f.get_object().get("/BaseFont")) for f in reader.pages[0]["/Resources"]["/Font"].values()}
        self.assertIn("/ZapfDingbats", fonts)
        self.assertNotIn(b"(\\177)", reader.pages[0].get_contents().get_data())

    def test_empty_notes_produce_a_readable_pdf(self):
        self.assertIn("No notes could be generated", self.text_of(notes_to_pdf([], "T")))

    def test_long_notes_run_over_several_pages(self):
        many = [{**self.SECTION, "heading": f"Topic {i}", "key_points": ["Some key point about the topic."] * 6}
                for i in range(12)]
        reader = PdfReader(io.BytesIO(notes_to_pdf(many, "T")))
        self.assertGreater(len(reader.pages), 1)
        self.assertIn("Topic 11", self.text_of(notes_to_pdf(many, "T")))


class NotesRouteTests(AgentTestCase):
    def test_rule_router_sends_notes_requests_to_notes(self):
        for message in ["make notes on ACID", "Create a PDF of notes on OLTP and OLAP", "Notes on read replicas, please",
                        "Summarize the ACID properties for me", "give me a cheat sheet for RDS"]:
            with self.subTest(message=message):
                self.assertEqual(route_message(message).name, "notes")

    def test_questions_that_only_mention_notes_stay_doubts(self):
        for message in ["Where can I find the lecture notes?", "Are these notes enough to pass the exam?",
                        "Is this topic covered in the notes?", "Who prepared the course notes?"]:
            with self.subTest(message=message):
                self.assertEqual(route_message(message).name, "doubt")

    def test_topic_extraction(self):
        cases = {
            "make notes on ACID": ["ACID"],
            "Create a PDF of notes on OLTP and OLAP": ["OLTP", "OLAP"],
            "make notes on ACID, BASE and OLTP": ["ACID", "BASE", "OLTP"],
            "Generate notes with definitions and use cases for NoSQL": ["NoSQL"],
            "I need a one-page summary of YARN": ["YARN"],
            "Summarize the ACID properties for me": ["ACID properties"],
            "Prepare quick notes: BASE model": ["BASE model"],
            "make me notes": [],
        }
        for message, expected in cases.items():
            with self.subTest(message=message):
                self.assertEqual(extract_notes_topics(message), expected)

    def test_at_most_three_topics_from_chat(self):
        self.assertEqual(len(extract_notes_topics("notes on ACID, BASE, OLTP, OLAP and RDS")), 3)

    def test_orchestrator_runs_the_note_generator(self):
        result = self.orch.handle("make notes on ACID", self.ctx(fake(notes_json())))
        self.assertEqual(result.route.name, "notes")
        self.assertEqual(result.primary.kind, "notes")
        self.assertEqual(result.primary.agent, "Note Generator")
        self.assertEqual(len(result.primary.data["sections"]), 1)

    def test_notes_without_a_topic_asks_which(self):
        result = self.orch.handle("make me notes", self.ctx(fake(notes_json())))
        self.assertEqual(result.primary.kind, "clarify")

    def test_uncovered_notes_request_hands_off_to_the_gap_handler(self):
        result = self.orch.handle("make notes on Kubernetes pods", self.ctx(fake(notes_json())))
        self.assertEqual([r.agent for r in result.results], ["Note Generator", "Gap Handler"])
        self.assertTrue(any("Note Generator -> Gap Handler" in line for line in result.trace))


class TopicListTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        docs = [
            Document(page_content="alpha text", metadata={"source": "a.pdf", "page": 2, "section": "SECOND SLIDE TITLE"}),
            Document(page_content="beta text", metadata={"source": "a.pdf", "page": 1, "section": "FIRST SLIDE TITLE"}),
            Document(page_content="beta again", metadata={"source": "a.pdf", "page": 3, "section": "FIRST SLIDE TITLE"}),
            Document(page_content="gamma text", metadata={"source": "b.pdf", "page": 1, "section": "OTHER FILE TITLE"}),
            Document(page_content="Q: x A: y", metadata={"source": "Faculty Answers", "page": 0, "section": "A faculty question"}),
            Document(page_content="no title here", metadata={"source": "a.pdf", "page": 9}),
        ]
        cls.store = build_vectorstore(docs, get_embeddings())

    def test_document_order_without_duplicates_or_faculty_answers(self):
        self.assertEqual(list_indexed_topics(self.store),
                         ["FIRST SLIDE TITLE", "SECOND SLIDE TITLE", "OTHER FILE TITLE"])

    def test_filter_by_file(self):
        self.assertEqual(list_indexed_topics(self.store, ["b.pdf"]), ["OTHER FILE TITLE"])


if __name__ == "__main__":
    unittest.main()
