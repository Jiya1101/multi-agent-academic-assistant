"""
Audio timing and fluency metrics for spoken understanding checks.

This module deliberately uses lightweight signal processing rather than a
separate VAD model. It is good enough for project/demo use: estimate where the
student was speaking, measure pauses, and combine that with transcript-derived
signals such as word rate and filler words. A stronger VAD can replace the
segmentation later without changing the public metric shape.
"""

import re
from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np

from rag_core.speech import SAMPLE_RATE, SILENCE_RMS, decode_audio

FRAME_SECONDS = 0.025
HOP_SECONDS = 0.010
MIN_SPEECH_EVENT_SECONDS = 0.15
MIN_PAUSE_SECONDS = 0.60
LONG_PAUSE_SECONDS = 1.00
MERGE_GAP_SECONDS = 0.35

_FILLERS = {
    "uh",
    "um",
    "umm",
    "ummm",
    "erm",
    "er",
    "ah",
    "hmm",
    "like",
    "basically",
    "actually",
    "maybe",
    "probably",
}
_FILLER_PHRASES = ("i think", "i guess", "i mean", "sort of", "kind of")


@dataclass(frozen=True)
class TimeSpan:
    """One detected speech or pause interval in seconds."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class AudioMetrics:
    """Signals used to infer recall difficulty during a spoken answer."""

    duration_seconds: float
    speech_seconds: float
    pause_seconds: float
    speech_ratio: float
    pause_ratio: float
    speech_event_count: int
    pause_count: int
    long_pause_count: int
    mean_pause_seconds: float
    max_pause_seconds: float
    leading_silence_seconds: float
    trailing_silence_seconds: float
    word_count: int
    words_per_minute: float
    articulation_words_per_minute: float
    filler_count: int
    filler_rate_per_minute: float

    def to_dict(self) -> dict:
        return asdict(self)


def _frames(audio: np.ndarray) -> tuple[np.ndarray, int, int]:
    frame = max(1, int(FRAME_SECONDS * SAMPLE_RATE))
    hop = max(1, int(HOP_SECONDS * SAMPLE_RATE))
    if len(audio) < frame:
        padded = np.pad(audio, (0, frame - len(audio)))
        return padded.reshape(1, frame), frame, hop
    starts = np.arange(0, len(audio) - frame + 1, hop)
    return np.stack([audio[start:start + frame] for start in starts]), frame, hop


def _speech_threshold(rms: np.ndarray) -> float:
    # Adapt to microphone volume while never going below the silence gate used
    # before Whisper. Percentile-based scaling keeps one loud click from making
    # all ordinary speech look silent.
    loud_reference = float(np.percentile(rms, 75)) if len(rms) else 0.0
    return max(SILENCE_RMS, loud_reference * 0.35)


def _mask_to_spans(mask: np.ndarray, hop: int, frame: int, total_samples: int) -> List[TimeSpan]:
    spans: List[TimeSpan] = []
    start: Optional[int] = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        elif not active and start is not None:
            spans.append(_span_from_frames(start, index - 1, hop, frame, total_samples))
            start = None
    if start is not None:
        spans.append(_span_from_frames(start, len(mask) - 1, hop, frame, total_samples))
    return spans


def _span_from_frames(start_frame: int, end_frame: int, hop: int, frame: int, total_samples: int) -> TimeSpan:
    start = start_frame * hop / SAMPLE_RATE
    end = min(total_samples, end_frame * hop + frame) / SAMPLE_RATE
    return TimeSpan(round(start, 3), round(end, 3))


def _merge_speech_spans(spans: List[TimeSpan]) -> List[TimeSpan]:
    filtered = [span for span in spans if span.duration >= MIN_SPEECH_EVENT_SECONDS]
    if not filtered:
        return []
    merged = [filtered[0]]
    for span in filtered[1:]:
        previous = merged[-1]
        if span.start - previous.end <= MERGE_GAP_SECONDS:
            merged[-1] = TimeSpan(previous.start, span.end)
        else:
            merged.append(span)
    return merged


def speech_spans(audio: np.ndarray) -> List[TimeSpan]:
    """Estimate intervals where the student is speaking."""
    if len(audio) == 0:
        return []
    frames, frame, hop = _frames(audio)
    windowed = frames * np.hanning(frame)
    rms = np.sqrt(np.mean(np.square(windowed), axis=1))
    spans = _mask_to_spans(rms >= _speech_threshold(rms), hop, frame, len(audio))
    return _merge_speech_spans(spans)


def pause_spans(spans: List[TimeSpan]) -> List[TimeSpan]:
    """Silent gaps between detected speech events."""
    pauses: List[TimeSpan] = []
    for left, right in zip(spans, spans[1:]):
        gap = right.start - left.end
        if gap >= MIN_PAUSE_SECONDS:
            pauses.append(TimeSpan(left.end, right.start))
    return pauses


def _word_count(transcript: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", transcript))


def _filler_count(transcript: str) -> int:
    lowered = transcript.lower()
    tokens = re.findall(r"\b[a-z']+\b", lowered)
    count = sum(1 for token in tokens if token.strip("'") in _FILLERS)
    count += sum(len(re.findall(rf"\b{re.escape(phrase)}\b", lowered)) for phrase in _FILLER_PHRASES)
    return count


def analyze_audio_response(data: bytes, transcript: str = "") -> AudioMetrics:
    """
    Decode a recording and compute pause/fluency metrics.

    `transcript` should be the text produced by Whisper or edited by the
    student. Audio-only metrics still work when it is empty.
    """
    audio = decode_audio(data)
    duration = len(audio) / SAMPLE_RATE if len(audio) else 0.0
    spans = speech_spans(audio)
    pauses = pause_spans(spans)

    speech_seconds = sum(span.duration for span in spans)
    pause_seconds = sum(span.duration for span in pauses)
    word_count = _word_count(transcript)
    filler_count = _filler_count(transcript)
    minutes = duration / 60 if duration else 0.0
    speech_minutes = speech_seconds / 60 if speech_seconds else 0.0

    leading = spans[0].start if spans else duration
    trailing = max(0.0, duration - spans[-1].end) if spans else duration
    pause_durations = [span.duration for span in pauses]

    return AudioMetrics(
        duration_seconds=round(duration, 3),
        speech_seconds=round(speech_seconds, 3),
        pause_seconds=round(pause_seconds, 3),
        speech_ratio=round(speech_seconds / duration, 3) if duration else 0.0,
        pause_ratio=round(pause_seconds / duration, 3) if duration else 0.0,
        speech_event_count=len(spans),
        pause_count=len(pauses),
        long_pause_count=sum(1 for p in pauses if p.duration >= LONG_PAUSE_SECONDS),
        mean_pause_seconds=round(float(np.mean(pause_durations)), 3) if pause_durations else 0.0,
        max_pause_seconds=round(max(pause_durations), 3) if pause_durations else 0.0,
        leading_silence_seconds=round(leading, 3),
        trailing_silence_seconds=round(trailing, 3),
        word_count=word_count,
        words_per_minute=round(word_count / minutes, 1) if minutes else 0.0,
        articulation_words_per_minute=round(word_count / speech_minutes, 1) if speech_minutes else 0.0,
        filler_count=filler_count,
        filler_rate_per_minute=round(filler_count / minutes, 1) if minutes else 0.0,
    )
