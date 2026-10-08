"""
Tests for the oral check on slide decks written as terse bullets. Such a deck has almost no full sentences,
which used to make the check say "not covered" about material it simply could not read as prose.

Run:  python -m unittest discover -s tests -v
"""

import json
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

from agents import OralAssessor
from agents.oral_assessor import MAX_MODEL_POINTS, NO_HEADING_MESSAGE, _bullet_headings
from rag_core.embeddings import get_embeddings
from rag_core.normalize import bullet_points as _bullet_points
from rag_core.vectorstore import build_vectorstore
from test_agents import AgentTestCase, fake

DECK = (
    "MUTUAL EXCLUSION 1. Goal of Coordination ● Processes in a distributed system must coordinate actions. "
    "● Example: spacecraft control → all subsystems must agree. ● No global clock. "
    "● Failures are common. 2. Failure Assumptions ● Crash failures only → process stops. "
    "○ Works via “I am alive” messages. ○ Large timeout → late detection. "
    "3. Distributed Mutual Exclusion ● Definition : Only one process at a time enters the critical section. "
    "● Pros: simple, easy. ● Cons: single point of failure, bottleneck. c) Ricart & Agrawala’s Algorithm "
    "(Multicast + Logical Clocks) ● Uses multicast requests + Lamport time stamps. "
    "● Fairness : Requests served in order. \U0001F4DD Recall Notes (Quick Revision)."
)


class BulletPointTests(unittest.TestCase):
    def test_bullets_become_points_and_headings_before_the_first_bullet_are_skipped(self):
        points = _bullet_points(DECK, "MUTUAL EXCLUSION")
        self.assertIn("Processes in a distributed system must coordinate actions.", points)
        self.assertFalse(any(p.startswith("1. Goal") for p in points))

    def test_arrows_are_kept_because_they_are_part_of_the_slide(self):
        self.assertIn("Example: spacecraft control → all subsystems must agree.", _bullet_points(DECK, "MUTUAL EXCLUSION"))

    def test_a_sub_heading_glued_to_the_end_of_a_bullet_is_removed(self):
        points = _bullet_points(DECK, "MUTUAL EXCLUSION")
        self.assertIn("Cons: single point of failure, bottleneck.", points)
        self.assertFalse(any("Ricart" in p for p in points))
        self.assertFalse(any("Failure Assumptions" in p for p in points))

    def test_an_icon_label_at_the_end_is_removed(self):
        points = _bullet_points(DECK, "MUTUAL EXCLUSION")
        self.assertFalse(any("Recall Notes" in p for p in points))
        self.assertIn("Fairness : Requests served in order.", points)

    def test_fragments_and_repeats_are_dropped(self):
        points = _bullet_points("TITLE ● Yes. ● Fine. ● Only one process may enter. ● Only one process may enter.", "TITLE")
        self.assertEqual(points, ["Only one process may enter."])

    def test_text_without_bullets_gives_nothing(self):
        self.assertEqual(_bullet_points("A heading and one plain sentence with no bullets at all"), [])


class OralQuestionOnBulletDeckTests(AgentTestCase):
    @classmethod
    def setUpClass(cls):
        cls.vectorstore = build_vectorstore(
            [Document(page_content=DECK, metadata={"source": "me.pdf", "page": 0, "section": "MUTUAL EXCLUSION"})],
            get_embeddings(),
        )

    def test_a_bullet_deck_now_gets_a_question(self):
        result = OralAssessor().generate_question_from_material(
            self.ctx(llm=fake(json.dumps({"question": "In your own words, what is the goal of mutual exclusion?"}))),
            rng=random.Random(0),
        )
        self.assertTrue(result.grounded is not False)
        self.assertEqual(result.kind, "oral_question")
        self.assertIn("goal of mutual exclusion", result.text)
        self.assertEqual(result.data["topic"], "Mutual Exclusion")

    def test_the_model_answer_stays_short_and_comes_from_the_slides(self):
        result = OralAssessor().generate_question_from_material(
            self.ctx(llm=fake(json.dumps({"question": "What is mutual exclusion and why does it matter?"}))),
            rng=random.Random(0),
        )
        lines = result.data["best_answer"].splitlines()
        self.assertLessEqual(len(lines), MAX_MODEL_POINTS)
        self.assertGreaterEqual(len(lines), 3)
        self.assertTrue(all(line.startswith("- ") for line in lines))

    def test_the_points_closest_to_the_question_are_kept(self):
        result = OralAssessor().generate_question_from_material(
            self.ctx(llm=fake(json.dumps({"question": "Explain the critical section definition."}))),
            rng=random.Random(0),
        )
        self.assertIn("critical section", result.data["best_answer"])

    def test_the_headings_come_from_the_bullet_fallback_only_when_sentences_fail(self):
        self.assertEqual([h["heading"] for h in _bullet_headings(self.ctx())], ["Mutual Exclusion"])

    def test_material_with_nothing_to_ask_says_so_instead_of_claiming_it_is_not_covered(self):
        empty = build_vectorstore(
            [Document(page_content="Just a short line.", metadata={"source": "x.pdf", "page": 0, "section": "Tiny"})],
            get_embeddings(),
        )
        ctx = self.ctx()
        ctx.vectorstore = empty
        result = OralAssessor().generate_question_from_material(ctx)
        self.assertFalse(result.grounded)
        self.assertEqual(result.data["error"], "no_headings")
        self.assertEqual(result.text, NO_HEADING_MESSAGE)
        self.assertNotIn("do not contain information", result.text)


if __name__ == "__main__":
    unittest.main()
