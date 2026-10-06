"""
Speech to text: turn a recorded question into text, fully offline.

Uses OpenAI's Whisper (the model named in `config.SPEECH_MODEL_NAME`) through
`transformers` on the CPU. The audio never leaves the computer.

Around the model there is some care, because Whisper has a known habit of
inventing text ("Thank you.") when given silence or noise:
  - audio that is too short or too quiet is rejected before the model runs;
  - a few well-known phantom phrases are discarded;
  - recordings longer than Whisper's 30-second window are split and joined.

An optional `hint` (course vocabulary such as "ACID, OLTP, YARN") is given to
Whisper as a starting prompt so it spells domain words the way the notes do.
"""

import math
import re
import threading
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from math import gcd
from typing import List, Optional

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from rag_core.config import SPEECH_MODEL_NAME

SAMPLE_RATE = 16000          # what Whisper expects
WINDOW_SECONDS = 30          # Whisper's fixed window
MIN_SECONDS = 0.4            # shorter recordings are treated as accidental clicks
SILENCE_RMS = 0.004          # root-mean-square amplitude below this is treated as silence
NOISE_FLATNESS = 0.5         # spectral flatness above this is treated as noise, not speech
# Phrases Whisper tends to invent for silence or noise. A student never needs to ask these.
_PHANTOM_PHRASES = frozenset({"", "thank you", "thanks", "thanks for watching", "thank you for watching", "you", "bye"})


@dataclass
class Transcript:
    text: str
    seconds: float
    problem: Optional[str] = None  # "too_short" | "silent" | "noise" | "no_speech" when text is empty


def decode_audio(data: bytes) -> np.ndarray:
    """WAV/FLAC/OGG bytes -> mono float32 audio at 16 kHz."""
    audio, rate = sf.read(BytesIO(data), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)  # stereo -> mono
    if rate != SAMPLE_RATE:
        divisor = gcd(rate, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // divisor, rate // divisor).astype(np.float32)
    return audio


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class SpeechRecognizer:
    """Whisper wrapper. The model is loaded on first use, then kept in memory."""

    def __init__(self, model_name: str = SPEECH_MODEL_NAME) -> None:
        self.model_name = model_name
        self._processor = None
        self._model = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        with self._lock:
            if self._model is None:
                import torch
                from transformers import WhisperForConditionalGeneration, WhisperProcessor

                self._processor = WhisperProcessor.from_pretrained(self.model_name)
                self._model = WhisperForConditionalGeneration.from_pretrained(self.model_name).eval()
                self._torch = torch

    def _run_whisper(self, audio: np.ndarray, hint: Optional[str]) -> str:
        """Transcribe one chunk of at most 30 seconds."""
        self._load()
        features = self._processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        kwargs = {"max_new_tokens": 200}
        if hint:
            kwargs["prompt_ids"] = self._processor.get_prompt_ids(hint, return_tensors="pt")
        with self._torch.inference_mode():
            ids = self._model.generate(features, **kwargs)
        text = self._processor.batch_decode(ids, skip_special_tokens=True)[0]
        if hint and text.strip().startswith(hint.strip()):  # some versions echo the prompt back
            text = text.strip()[len(hint.strip()):]
        return text

    def transcribe(self, data: bytes, hint: Optional[str] = None) -> Transcript:
        audio = decode_audio(data)
        seconds = len(audio) / SAMPLE_RATE
        if seconds < MIN_SECONDS:
            return Transcript("", seconds, "too_short")
        if float(np.sqrt(np.mean(np.square(audio)))) < SILENCE_RMS:
            return Transcript("", seconds, "silent")
        if spectral_flatness(audio) > NOISE_FLATNESS:
            return Transcript("", seconds, "noise")

        window = WINDOW_SECONDS * SAMPLE_RATE
        pieces: List[str] = [
            _clean(self._run_whisper(audio[start:start + window], hint))
            for start in range(0, len(audio), window)
        ]
        text = _clean(" ".join(p for p in pieces if p))
        if text.lower().strip(" .!?,") in _PHANTOM_PHRASES:
            return Transcript("", seconds, "no_speech")
        return Transcript(text, seconds)


_recognizer: Optional[SpeechRecognizer] = None


def get_recognizer() -> SpeechRecognizer:
    """One shared recognizer per process, so the model is loaded once."""
    global _recognizer
    if _recognizer is None:
        _recognizer = SpeechRecognizer()
    return _recognizer


def course_vocabulary_hint(texts: List[str], limit: int = 30) -> str:
    """
    The course's own terms as a short starting prompt for Whisper, so it writes
    "ACID" and "OLTP" instead of "acid" and "old TP". Example: "Terms: ACID, OLAP, RDS".

    A term counts as an acronym when the notes write it in capitals far more
    often than in lowercase. That keeps real acronyms (ACID, RDS) and drops
    ordinary words that only appear capitalised in slide titles (DATABASE,
    SERVICES), which would otherwise teach Whisper to answer in capitals.
    Rare capitalised names (Hadoop, Redshift) are kept too.
    """
    upper: Counter = Counter()        # ACID
    capitalised: Counter = Counter()  # Hadoop
    lower: Counter = Counter()        # hadoop
    plain: Counter = Counter()        # written lowercase OR Capitalised: "amazon", "Amazon"
    for text in texts:
        upper.update(re.findall(r"\b[A-Z][A-Z0-9]{1,5}\b", text))  # acronyms are short
        capitalised.update(re.findall(r"\b[A-Z][a-z]{3,}\b", text))
        lower.update(re.findall(r"\b[a-z]{2,}\b", text))
        plain.update(w.lower() for w in re.findall(r"\b[A-Za-z][a-z]+\b", text))

    acronyms = [w for w, n in upper.items() if n >= 2 and n > 2 * plain[w.lower()]]
    names = [w for w, n in capitalised.items() if n >= 3 and lower[w.lower()] == 0 and _is_rare_word(w)]
    ranked = sorted(acronyms, key=lambda w: (-upper[w], w)) + sorted(names, key=lambda w: (-capitalised[w], w))
    return "Terms: " + ", ".join(ranked[:limit]) if ranked else ""


def _is_rare_word(word: str) -> bool:
    """True if `word` is not among the roughly 5,000 most common English words."""
    try:
        import wordninja
        model = wordninja.DEFAULT_LANGUAGE_MODEL._wordcost
        cost = model.get(word.lower())
        return cost is None or math.exp(cost) / math.log(len(model)) - 1 > 5000
    except Exception:
        return True


def spectral_flatness(audio: np.ndarray) -> float:
    """
    How noise-like the loudest part of a recording is: near 0 for speech and
    tones, near 1 for white noise. Only the loudest 30 percent of 25 ms frames
    are measured, so the quiet gaps around speech do not count.
    """
    frame = SAMPLE_RATE // 40
    count = len(audio) // frame
    if count == 0:
        return 0.0
    frames = audio[: count * frame].reshape(count, frame)
    loud = frames[np.argsort(np.square(frames).mean(axis=1))[-max(1, int(count * 0.3)):]]
    power = np.abs(np.fft.rfft(loud * np.hanning(frame), axis=1)) ** 2 + 1e-12
    return float(np.mean(np.exp(np.log(power).mean(axis=1)) / power.mean(axis=1)))
