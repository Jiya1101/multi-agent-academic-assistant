"""
Routing: deciding which agent handles a student's message.

Three routers, all with the same interface `route(message, db_dir=None) -> Route`:

  RuleRouter     hand-written regex rules. Instant and predictable, but it only
                 knows the phrasings it has rules for.
  LearnedRouter  a classifier trained on labelled example messages. It works on
                 the MEANING of a message (sentence embeddings), so "test my
                 knowledge of RDS" is recognised as a quiz request with no
                 keyword rule for it.
  HybridRouter   the one the app uses: take the learned prediction when it is
                 confident, otherwise fall back to the rules. It also retrains
                 when corrections from students ("wrong kind of help") are
                 stored, so it improves with use.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from agents.router_data import ROUTES, TRAIN
from rag_core.embeddings import get_embeddings
from rag_core.learning_log import list_route_feedback

ROUTE_LABELS = {
    "doubt": "Answer a question",
    "explain": "Explain a concept",
    "quiz": "Quiz me",
    "progress": "Show my progress",
    "study_next": "Tell me what to study next",
    "notes": "Make study notes",
}
_REASONS = {
    "notes": "asks for study notes to be written",
    "doubt": "a question about the notes",
    "explain": "asks for a concept to be explained",
    "quiz": "asks to be quizzed",
    "progress": "asks about their own progress",
    "study_next": "asks what to study next",
}


@dataclass
class Route:
    name: str   # one of ROUTES
    reason: str
    params: Dict[str, Any] = field(default_factory=dict)
    source: str = "rule"                 # "rule" | "learned"
    confidence: Optional[float] = None   # learned router's probability, if used


# --------------------------------------------------------------------------- #
# Rule-based router                                                          #
# --------------------------------------------------------------------------- #
_STUDY_NEXT = re.compile(
    r"\b(what should i (study|revise|learn|focus)|study plan|help me (improve|revise))\b", re.I
)
_PROGRESS = re.compile(
    r"\b(my progress|how am i doing|my performance|my (weak|strong) (topics?|areas?)|my scores?|my results?)\b",
    re.I,
)
_QUIZ = re.compile(r"\b(quiz|test me|mcqs?|practice (questions?|problems?)|self[- ]test)\b", re.I)
_NOTES = re.compile(
    r"\b(notes? (on|for|about)|(make|create|generate|write|prepare|compile|give me|get me)\b[^.?!]{0,30}\bnotes?\b|"
    r"revision notes|study (notes|guide|sheet)|cheat ?sheet|summari[sz]e|one[- ]page summary)\b",
    re.I,
)
_EXPLAIN = re.compile(
    r"\b(simply|simple terms|eli5|like i('| a)?m (5|five)|for a beginner|beginner|step[- ]by[- ]step|"
    r"intuition|analogy|walk me through|in detail|in depth|detailed|advanced)\b",
    re.I,
)
_DETAILED = re.compile(r"\b(in detail|in depth|detailed|advanced|thorough)\b", re.I)
_QUIZ_TOPIC = re.compile(
    r"\b(?:quiz me|test me|mcqs?|practice (?:questions?|problems?)|self[- ]test|quiz)\s*"
    r"(?:on|about|for|in)?\s*(.*)$",
    re.I,
)
_PREPOSITION_TOPIC = re.compile(r"\b(?:on|about|of|for|regarding|covering)\s+(.+)$", re.I)
_QUIZ_FILLER = frozenset(
    "quiz quizzes test tests me my knowledge understanding understand questions question mcq mcqs multiple "
    "choice practice problems give can could you please i id want would like some few quick short a an the "
    "make generate create ask let lets let's do check if how well know on about of for set start mock pop "
    "time try throw at something to be tested self examine evaluate what remember just learned s d".split()
)


def extract_quiz_topic(message: str) -> str:
    """Best-effort topic of a quiz request ('' if none can be found)."""
    match = _QUIZ_TOPIC.search(message)
    topic = (match.group(1) if match else "").strip(" ?.!,")
    if topic:
        return topic
    match = _PREPOSITION_TOPIC.search(message)
    if match:
        topic = match.group(1).strip(" ?.!,")
        if topic:
            return topic
    words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", message)
             if w.lower() not in _QUIZ_FILLER and not w.isdigit()]
    return " ".join(words)


_NOTES_FILLER = frozenset(
    "make me generate create write prepare give get compile put together some short quick brief concise "
    "simple heading wise headingwise subheading detailed revision study exam exams notes note summary "
    "summarize summarise cheat sheet guide handout pdf download a an the i id need want would like please "
    "can could you with definition definitions use case cases example examples turn into this topic "
    "condense one page as up bullet point points for on about of s d".split()
)
_NOTES_SEPARATORS = frozenset({",", "&", "and", "plus"})
MAX_NOTES_TOPICS = 3


def _split_topics(text: str) -> list:
    topics, current = [], []
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*|,|&", text):
        lowered = token.lower()
        if lowered in _NOTES_SEPARATORS:
            if current:
                topics.append(" ".join(current))
            current = []
        elif lowered not in _NOTES_FILLER:
            current.append(token)
    if current:
        topics.append(" ".join(current))
    return [t for t in topics if len(t) > 1 and not t.isdigit()]


def extract_notes_topics(message: str) -> list:
    """Topics named in a notes request ('notes on ACID and BASE' -> ['ACID', 'BASE']); [] if none."""
    match = _PREPOSITION_TOPIC.search(message)
    topics = _split_topics(match.group(1)) if match else []
    # "... the ACID properties for me": the first preposition led nowhere, so scan everything.
    return (topics or _split_topics(message))[:MAX_NOTES_TOPICS]


def _params_for(name: str, message: str) -> Dict[str, Any]:
    if name == "notes":
        return {"topics": extract_notes_topics(message)}
    if name == "quiz":
        return {"topic": extract_quiz_topic(message)}
    if name == "explain":
        return {"level": "detailed" if _DETAILED.search(message) else "beginner"}
    return {}


def route_message(message: str) -> Route:
    """Rule-based routing."""
    if _STUDY_NEXT.search(message):
        name = "study_next"
    elif _PROGRESS.search(message):
        name = "progress"
    elif _NOTES.search(message):
        name = "notes"
    elif _QUIZ.search(message):
        name = "quiz"
    elif _EXPLAIN.search(message):
        name = "explain"
    else:
        name = "doubt"
    reason = _REASONS[name]
    if name == "explain":
        reason = f"asks for a {_params_for(name, message)['level']} explanation"
    return Route(name, reason, _params_for(name, message), source="rule")


class RuleRouter:
    def route(self, message: str, db_dir=None) -> Route:
        return route_message(message)


# --------------------------------------------------------------------------- #
# Learned router                                                             #
# --------------------------------------------------------------------------- #
class LearnedRouter:
    """Logistic regression over sentence embeddings of labelled example messages."""

    def __init__(
        self,
        examples: Optional[List[Tuple[str, str]]] = None,
        embeddings=None,
        use_words: bool = True,
    ) -> None:
        self.embeddings = embeddings or get_embeddings()
        self.use_words = use_words
        if examples is None:
            examples = [(text, route) for route, texts in TRAIN.items() for text in texts]
        self.fit(examples)

    def _embed(self, texts: List[str]) -> np.ndarray:
        return np.array(self.embeddings.embed_documents(list(texts)))

    def _features(self, texts: List[str], fit: bool = False):
        """Meaning (sentence embedding) plus the exact words used (TF-IDF).

        Embeddings capture what a message is about but blur small, decisive words
        like "simply", "my" or "quiz"; the word features keep those."""
        dense = self._embed(texts)
        if not self.use_words:
            return dense
        words = self.vectorizer.fit_transform(texts) if fit else self.vectorizer.transform(texts)
        return hstack([csr_matrix(dense), words]).tocsr()

    def fit(self, examples: List[Tuple[str, str]]) -> None:
        texts, labels = zip(*examples)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True)
        self.model = LogisticRegression(C=10.0, max_iter=3000, class_weight="balanced")
        self.model.fit(self._features(list(texts), fit=True), list(labels))

    def predict_proba(self, messages: List[str]) -> List[Dict[str, float]]:
        probs = self.model.predict_proba(self._features(list(messages)))
        return [dict(zip(self.model.classes_, row)) for row in probs]

    def predict(self, message: str) -> Tuple[str, float]:
        probs = self.predict_proba([message])[0]
        name = max(probs, key=probs.get)
        return name, float(probs[name])

    def route(self, message: str, db_dir=None) -> Route:
        name, confidence = self.predict(message)
        return Route(name, _REASONS[name], _params_for(name, message), "learned", confidence)


class HybridRouter:
    """Use the learned prediction when it is confident, otherwise the rules."""

    def __init__(self, threshold: float = 0.4, embeddings=None) -> None:
        self.threshold = threshold
        self._embeddings = embeddings
        self._learned: Optional[LearnedRouter] = None
        self._feedback_count = -1

    def _model(self, db_dir) -> LearnedRouter:
        """(Re)train when the number of stored corrections changes."""
        feedback = list_route_feedback(db_dir) if db_dir is not None else []
        if self._learned is None or len(feedback) != self._feedback_count:
            examples = [(text, route) for route, texts in TRAIN.items() for text in texts]
            examples += [(f["message"], f["correct"]) for f in feedback if f["correct"] in ROUTES]
            self._learned = LearnedRouter(examples, self._embeddings)
            self._feedback_count = len(feedback)
        return self._learned

    def route(self, message: str, db_dir=None) -> Route:
        learned = self._model(db_dir).route(message)
        if learned.confidence is not None and learned.confidence >= self.threshold:
            return learned
        return route_message(message)


_default_router: Optional[HybridRouter] = None


def default_router() -> HybridRouter:
    """One shared router per process, so the classifier is trained once."""
    global _default_router
    if _default_router is None:
        _default_router = HybridRouter()
    return _default_router
