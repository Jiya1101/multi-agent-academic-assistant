"""
Tests for speech-to-text handling: decoding, silence gating, chunking, cleanup.

Whisper itself is replaced by a stand-in, so these run instantly without the
model. A real-model test runs only when RUN_SLOW_TESTS=1 and the model is cached.

Run:  python -m unittest discover -s tests -v
"""

import os
import sys
import unittest
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf

from rag_core.speech import (
    MIN_SECONDS,
    SAMPLE_RATE,
    SpeechRecognizer,
    course_vocabulary_hint,
    decode_audio,
    spectral_flatness,
)


def wav_bytes(samples: np.ndarray, rate: int = SAMPLE_RATE) -> bytes:
    buffer = BytesIO()
    sf.write(buffer, samples, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def tone(seconds: float, rate: int = SAMPLE_RATE, amplitude: float = 0.2) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype("float32")


class FakeWhisper(SpeechRecognizer):
    """Records what it was asked to transcribe instead of running the model."""

    def __init__(self, replies=("What is ACID?",)):
        super().__init__()
        self.replies = list(replies)
        self.calls = []

    def _run_whisper(self, audio, hint):
        self.calls.append((len(audio) / SAMPLE_RATE, hint))
        return self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]


class DecodeTests(unittest.TestCase):
    def test_16k_mono_passes_through(self):
        audio = decode_audio(wav_bytes(tone(1.0)))
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(len(audio), SAMPLE_RATE)

    def test_stereo_44k_becomes_mono_16k(self):
        stereo = np.stack([tone(2.0, 44100), tone(2.0, 44100)], axis=1)
        audio = decode_audio(wav_bytes(stereo, 44100))
        self.assertEqual(audio.ndim, 1)
        self.assertAlmostEqual(len(audio) / SAMPLE_RATE, 2.0, delta=0.01)

    def test_resampling_keeps_the_signal(self):
        original = tone(1.0, 22050, amplitude=0.3)
        audio = decode_audio(wav_bytes(original, 22050))
        self.assertAlmostEqual(float(np.sqrt(np.mean(audio ** 2))), float(np.sqrt(np.mean(original ** 2))), delta=0.01)

    def test_garbage_bytes_raise(self):
        with self.assertRaises(Exception):
            decode_audio(b"this is not audio")


class TranscribeTests(unittest.TestCase):
    def test_returns_cleaned_text(self):
        recognizer = FakeWhisper(["  What   is\nACID? "])
        result = recognizer.transcribe(wav_bytes(tone(2.0)))
        self.assertEqual((result.text, result.problem), ("What is ACID?", None))
        self.assertAlmostEqual(result.seconds, 2.0, delta=0.01)

    def test_accidental_click_is_rejected_before_the_model_runs(self):
        recognizer = FakeWhisper()
        result = recognizer.transcribe(wav_bytes(tone(MIN_SECONDS / 2)))
        self.assertEqual((result.text, result.problem), ("", "too_short"))
        self.assertEqual(recognizer.calls, [])

    def test_silence_and_room_noise_are_rejected_before_the_model_runs(self):
        recognizer = FakeWhisper()
        rng = np.random.default_rng(0)
        for name, samples in {"silence": np.zeros(SAMPLE_RATE * 2, dtype="float32"),
                              "hiss": (rng.standard_normal(SAMPLE_RATE * 2) * 0.001).astype("float32")}.items():
            with self.subTest(name=name):
                result = recognizer.transcribe(wav_bytes(samples))
                self.assertEqual((result.text, result.problem), ("", "silent"))
        self.assertEqual(recognizer.calls, [])

    def test_phantom_phrases_are_discarded(self):
        for phantom in ["Thank you.", "thanks for watching!", " You ", ""]:
            with self.subTest(phantom=phantom):
                result = FakeWhisper([phantom]).transcribe(wav_bytes(tone(2.0)))
                self.assertEqual((result.text, result.problem), ("", "no_speech"))

    def test_a_real_question_containing_thank_you_is_kept(self):
        result = FakeWhisper(["Thank you, but what is ACID?"]).transcribe(wav_bytes(tone(2.0)))
        self.assertEqual(result.text, "Thank you, but what is ACID?")

    def test_hint_reaches_the_model(self):
        recognizer = FakeWhisper()
        recognizer.transcribe(wav_bytes(tone(2.0)), hint="ACID, OLTP")
        self.assertEqual(recognizer.calls[0][1], "ACID, OLTP")

    def test_long_recording_is_split_into_windows_and_joined(self):
        recognizer = FakeWhisper(["first part", "second part"])
        result = recognizer.transcribe(wav_bytes(tone(34.0)))
        self.assertEqual(len(recognizer.calls), 2)
        self.assertTrue(all(seconds <= 30.01 for seconds, _ in recognizer.calls))
        self.assertEqual(result.text, "first part second part")


class NoiseGateTests(unittest.TestCase):
    def test_white_noise_is_rejected_before_the_model_runs(self):
        recognizer = FakeWhisper()
        noise = (np.random.default_rng(0).standard_normal(SAMPLE_RATE * 3) * 0.05).astype("float32")
        result = recognizer.transcribe(wav_bytes(noise))
        self.assertEqual((result.text, result.problem), ("", "noise"))
        self.assertEqual(recognizer.calls, [])

    def test_tonal_and_noisy_voice_like_audio_passes(self):
        rng = np.random.default_rng(0)
        voiceish = tone(2.0) + (0.04 * rng.standard_normal(SAMPLE_RATE * 2)).astype("float32")
        self.assertLess(spectral_flatness(voiceish), 0.5)
        self.assertEqual(FakeWhisper(["ok"]).transcribe(wav_bytes(voiceish)).text, "ok")

    def test_flatness_ordering(self):
        noise = np.random.default_rng(1).standard_normal(SAMPLE_RATE * 2).astype("float32") * 0.1
        self.assertGreater(spectral_flatness(noise), 0.5)
        self.assertLess(spectral_flatness(tone(2.0)), 0.05)
        self.assertEqual(spectral_flatness(np.zeros(10, dtype="float32")), 0.0)  # shorter than one frame


class VocabularyHintTests(unittest.TestCase):
    NOTES = [
        "ACID DATABASE CONSISTENCY MODEL The ACID model guarantees atomicity. ACID is strict.",
        "BASE DATABASE CONSISTENCY MODEL The BASE model relaxes ACID. A database can use BASE.",
        "OLTP VS OLAP SYSTEMS OLTP handles many transactions while OLAP does analysis. OLAP is read heavy.",
        "AMAZON RDS BENEFITS Amazon RDS is a managed database service. Amazon RDS scales. RDS backs up.",
        "AMAZON AURORA Amazon Aurora is faster than other database engines. The database is managed.",
    ]

    def test_acronyms_are_kept_and_title_only_words_are_dropped(self):
        hint = course_vocabulary_hint(self.NOTES)
        self.assertTrue(hint.startswith("Terms: "))
        terms = hint[len("Terms: "):].split(", ")
        for acronym in ["ACID", "OLTP", "OLAP", "RDS", "BASE"]:
            self.assertIn(acronym, terms)
        # Words that are capitals only because slide titles are in capitals.
        for word in ["DATABASE", "MODEL", "SYSTEMS", "AMAZON", "BENEFITS", "VS"]:
            self.assertNotIn(word, terms)

    def test_no_hint_when_there_is_nothing_to_add(self):
        self.assertEqual(course_vocabulary_hint([]), "")
        self.assertEqual(course_vocabulary_hint(["plain lowercase text only"]), "")

    def test_limit(self):
        texts = [" ".join(f"AB{i:02d} AB{i:02d}" for i in range(60))]
        self.assertLessEqual(len(course_vocabulary_hint(texts, limit=10)[len("Terms: "):].split(", ")), 10)


@unittest.skipUnless(os.environ.get("RUN_SLOW_TESTS") == "1", "set RUN_SLOW_TESTS=1 to run with the real Whisper model")
class RealModelTests(unittest.TestCase):
    def test_transcribes_a_spoken_question(self):
        audio_dir = os.environ.get("SPEECH_TEST_AUDIO")
        if not audio_dir or not (Path(audio_dir) / "q1_acid_base.wav").exists():
            self.skipTest("SPEECH_TEST_AUDIO folder with q1_acid_base.wav not provided")
        from rag_core.speech import get_recognizer
        result = get_recognizer().transcribe((Path(audio_dir) / "q1_acid_base.wav").read_bytes())
        self.assertIn("difference", result.text.lower())


if __name__ == "__main__":
    unittest.main()
