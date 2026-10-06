"""
Tests for the learned router, the hybrid, the feedback loop and topic extraction.

Run:  python -m unittest discover -s tests -v
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.router_data import ROUTES, TEST, TRAIN
from agents.routing import HybridRouter, LearnedRouter, extract_quiz_topic, route_message
from rag_core.embeddings import get_embeddings
from rag_core.learning_log import list_route_feedback, record_route_feedback


def flatten(data):
    return [(text, route) for route, texts in data.items() for text in texts]


class DatasetTests(unittest.TestCase):
    def test_every_route_has_enough_examples(self):
        for route in ROUTES:
            self.assertGreaterEqual(len(TRAIN[route]), 30, route)
            self.assertGreaterEqual(len(TEST[route]), 10, route)

    def test_test_set_is_not_in_the_training_set(self):
        train = {text.lower().strip(" ?.!") for text, _ in flatten(TRAIN)}
        leaked = [t for t, _ in flatten(TEST) if t.lower().strip(" ?.!") in train]
        self.assertEqual(leaked, [])

    def test_no_message_has_two_labels(self):
        seen = {}
        for text, route in flatten(TRAIN) + flatten(TEST):
            key = text.lower().strip(" ?.!")
            if key in seen:
                self.assertEqual(seen[key], route, f"{text!r} is labelled both {seen[key]} and {route}")
            seen[key] = route


class TopicExtractionTests(unittest.TestCase):
    def test_topics(self):
        cases = {
            "quiz me on RDS read replicas": "RDS read replicas",
            "Quiz me about ACID?": "ACID",
            "can you test my knowledge of RDS": "RDS",
            "give me 5 questions on read replicas": "read replicas",
            "check if I understand OLAP": "OLAP",
            "I need practice on ACID": "ACID",
            "let's do a quick quiz": "",
            "quiz me": "",
            "test me": "",
        }
        for message, expected in cases.items():
            with self.subTest(message=message):
                self.assertEqual(extract_quiz_topic(message), expected)


class LearnedRouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.embeddings = get_embeddings()
        cls.learned = LearnedRouter(embeddings=cls.embeddings)

    def test_understands_wording_the_rules_do_not(self):
        # None of these contain a keyword the rules look for.
        self.assertEqual(route_message("can you test my knowledge of RDS").name, "doubt")
        self.assertEqual(self.learned.route("can you test my knowledge of RDS").name, "quiz")
        self.assertEqual(route_message("how well am I doing in this course").name, "doubt")
        self.assertEqual(self.learned.route("how well am I doing in this course").name, "progress")

    def test_route_carries_confidence_and_source(self):
        route = self.learned.route("quiz me on Spark")
        self.assertEqual((route.name, route.source), ("quiz", "learned"))
        self.assertTrue(0.0 < route.confidence <= 1.0)
        self.assertEqual(route.params["topic"], "Spark")

    def test_explain_level_comes_from_the_wording(self):
        self.assertEqual(self.learned.route("explain BASE in depth").params.get("level"), "detailed")

    def test_accuracy_on_held_out_test_is_not_worse_than_rules(self):
        test = flatten(TEST)
        learned = sum(self.learned.route(t).name == r for t, r in test) / len(test)
        rules = sum(route_message(t).name == r for t, r in test) / len(test)
        self.assertGreaterEqual(learned, rules, f"learned {learned:.2f} vs rules {rules:.2f}")
        self.assertGreaterEqual(learned, 0.80, f"learned accuracy {learned:.2f}")


class HybridRouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.embeddings = get_embeddings()

    def test_falls_back_to_rules_when_not_confident(self):
        router = HybridRouter(threshold=1.01, embeddings=self.embeddings)  # never confident
        route = router.route("quiz me on Spark")
        self.assertEqual((route.name, route.source), ("quiz", "rule"))

    def test_uses_learned_prediction_when_confident(self):
        router = HybridRouter(threshold=0.0, embeddings=self.embeddings)
        route = router.route("can you test my knowledge of RDS")
        self.assertEqual((route.name, route.source), ("quiz", "learned"))

    def test_corrections_are_learned(self):
        with tempfile.TemporaryDirectory() as tmp:
            router = HybridRouter(threshold=0.0, embeddings=self.embeddings)
            message = "zxq blorp frobnicate"
            router.route(message, tmp)
            self.assertEqual(router._feedback_count, 0)
            record_route_feedback(message, "doubt", "quiz", tmp)
            record_route_feedback(message, "doubt", "quiz", tmp)
            self.assertEqual(router.route(message, tmp).name, "quiz")
            self.assertEqual(router._feedback_count, 2, "router must retrain when corrections change")
            self.assertEqual(len(list_route_feedback(tmp)), 2)

    def test_ignores_feedback_with_unknown_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            record_route_feedback("some message", "doubt", "not_a_route", tmp)
            router = HybridRouter(threshold=0.0, embeddings=self.embeddings)
            router.route("hello", tmp)  # must not raise


if __name__ == "__main__":
    unittest.main()
