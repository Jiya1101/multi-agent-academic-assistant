"""
Tests for spoken-answer metrics and oral assessment storage.

Run: python -m unittest tests.test_oral_assessment -v
"""

import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf

from rag_core.audio_metrics import SAMPLE_RATE, analyze_audio_response
from rag_core.oral_assessment import (
    LEVEL_DEVELOPING,
    LEVEL_NEEDS_PRACTICE,
    LEVEL_STRONG,
    infer_understanding_level,
    list_oral_assessments,
    record_oral_assessment,
)


def wav_bytes(samples: np.ndarray, rate: int = SAMPLE_RATE) -> bytes:
    buffer = BytesIO()
    sf.write(buffer, samples, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def tone(seconds: float, amplitude: float = 0.25) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype("float32")


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype="float32")


class AudioMetricTests(unittest.TestCase):
    def test_pause_and_fluency_metrics_from_audio_and_transcript(self):
        samples = np.concatenate([
            silence(0.5),
            tone(1.0),
            silence(0.8),
            tone(1.0),
            silence(1.2),
            tone(0.8),
            silence(0.3),
        ])
        transcript = "Um ACID means atomicity consistency isolation and durability. I think it protects transactions."

        metrics = analyze_audio_response(wav_bytes(samples), transcript)

        self.assertAlmostEqual(metrics.duration_seconds, 5.6, delta=0.05)
        self.assertEqual(metrics.speech_event_count, 3)
        self.assertEqual(metrics.pause_count, 2)
        self.assertEqual(metrics.long_pause_count, 1)
        self.assertGreater(metrics.leading_silence_seconds, 0.4)
        self.assertEqual(metrics.word_count, 13)
        self.assertEqual(metrics.filler_count, 2)
        self.assertGreater(metrics.words_per_minute, 100)

    def test_silence_has_zero_speech_metrics(self):
        metrics = analyze_audio_response(wav_bytes(silence(2.0)), "")
        self.assertEqual(metrics.speech_event_count, 0)
        self.assertEqual(metrics.pause_count, 0)
        self.assertEqual(metrics.speech_seconds, 0.0)
        self.assertAlmostEqual(metrics.leading_silence_seconds, 2.0, delta=0.02)


class OralAssessmentStorageTests(unittest.TestCase):
    def test_level_rubric(self):
        strong_metrics = {"long_pause_count": 0, "pause_ratio": 0.1, "words_per_minute": 120, "filler_rate_per_minute": 1}
        hesitant_metrics = {"long_pause_count": 3, "pause_ratio": 0.5, "words_per_minute": 40, "filler_rate_per_minute": 9}

        self.assertEqual(infer_understanding_level(0.9, strong_metrics), LEVEL_STRONG)
        self.assertEqual(infer_understanding_level(0.65, strong_metrics), LEVEL_DEVELOPING)
        self.assertEqual(infer_understanding_level(0.5, hesitant_metrics), LEVEL_NEEDS_PRACTICE)

    def test_assessment_round_trip(self):
        samples = np.concatenate([tone(1.0), silence(0.5), tone(1.0)])
        metrics = analyze_audio_response(wav_bytes(samples), "ACID has four properties")
        with tempfile.TemporaryDirectory() as tmp:
            assessment_id = record_oral_assessment(
                topic="ACID",
                prompt="Explain ACID in your own words.",
                transcript="ACID has four properties",
                content_score=0.8,
                metrics=metrics,
                feedback="Good coverage.",
                student_id="s1",
                db_dir=tmp,
            )
            rows = list_oral_assessments(student_id="s1", db_dir=tmp)

        self.assertEqual(rows[0].id, assessment_id)
        self.assertEqual(rows[0].topic, "ACID")
        self.assertEqual(rows[0].student_id, "s1")
        self.assertEqual(rows[0].understanding_level, LEVEL_STRONG)
        self.assertEqual(rows[0].metrics["word_count"], 4)


if __name__ == "__main__":
    unittest.main()
