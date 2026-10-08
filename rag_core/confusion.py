"""
Concept Confusion: which course concepts the class seems to be struggling with.

A pile of questions about a concept does not mean students are confused by it:
they may simply be curious, or it may be the most interesting slide. So no
single count is used. Several different kinds of evidence about the same
concept (a slide topic) are combined, and the result is shown WITH its
evidence, not just as a number:

  breadth       how much of the active class asked about it
  repeat        how many of those students asked more than once
  follow-up     how often a question came back to a topic already asked about
  explain       how many requests were "explain this to me" rather than a question
  quiz          how many students missed it on the class quizzes
  oral          how many oral checks were not yet strong
  persistence   whether it was asked about on several different days

What this is and is not:
  - It is a RANKING AID for deciding what to revisit first. It is not a measurement of
    understanding, and it has not been checked against real exam results.
  - Every weight and cut-off below is a judgment call. The dashboard lets the professor change the
    weights, and `rank_stability` reports how much the ranking moves when they change.
  - Missing evidence is never counted as "no confusion". A signal with too little data is filled
    in with the class average for that signal, so a concept is neither rewarded nor punished for
    having less data, and its confidence is shown as Low.
  - A signal about people (breadth, repeat, quiz, oral) is used only when at least MIN_COHORT
    different students contribute to it.
  - Questions the notes could not answer have no slide, so they cannot belong to a concept. They
    stay in the Pending Gaps queue instead.
  - A concept is one slide topic. Slides about the same idea are not merged yet.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from rag_core.config import MIN_COHORT
from rag_core.insights import topic_title
from rag_core.item_analysis import first_attempts
from rag_core.normalize import human_title
from rag_core.oral_assessment import LEVEL_STRONG

CONCEPT_EXPLAINER = "Concept Explainer"

# name -> (label, default weight). The weights sum to 1; the quiz is the heaviest because it is the only
# direct measure of performance. They are starting points, not findings.
SIGNALS: Dict[str, Tuple[str, float]] = {
    "breadth": ("Share of the class who asked", 0.15),
    "repeat": ("Students who asked more than once", 0.10),
    "follow_up": ("Questions that were follow-ups", 0.15),
    "explain": ("Requests to explain it", 0.10),
    "quiz": ("Students who missed it on the class quiz", 0.30),
    "oral": ("Oral checks not yet strong", 0.10),
    "persistence": ("Asked about on several days", 0.10),
}
DEFAULT_WEIGHTS = {name: weight for name, (_label, weight) in SIGNALS.items()}

MIN_QUESTIONS = 5            # questions about a concept before a signal built from questions is used
BREADTH_SATURATION = 0.25    # a quarter of the active class asking counts as the maximum breadth signal
FOLLOW_UP_SATURATION = 0.50  # half the questions being follow-ups counts as the maximum
EXPLAIN_SATURATION = 0.50    # half the requests being "explain" counts as the maximum
PERSISTENCE_DAYS = 3         # being asked about on this many different days counts as the maximum

QUESTION_SIGNALS = ("breadth", "repeat", "follow_up", "explain", "persistence")


@dataclass
class Signal:
    name: str
    label: str
    weight: float
    value: Optional[float]      # 0 to 1, higher = more confusion; None when there is not enough data
    evidence: str = ""          # plain-language support for the value ("" when missing)


@dataclass
class ConceptConfusion:
    key: str
    concept: str
    questions: int
    signals: List[Signal]
    score: float = 0.0          # 0 to 100
    confidence: str = "Low"     # Low / Medium / High: how many different kinds of evidence back it
    stands_out: str = ""        # the signal that is furthest above the class average
    sources: int = 0


@dataclass
class ConfusionReport:
    concepts: List[ConceptConfusion]                       # scored, highest first
    unscored: List[Tuple[str, int]]                        # (concept, questions): too little data to say anything
    active_students: int
    means: Dict[str, float]                                # class average per signal, used to fill gaps
    weights: Dict[str, float]


@dataclass
class Stability:
    top_share: float            # share of re-weightings in which the concept stays in the top k
    best_rank: int
    worst_rank: int


def concept_key(topic: str) -> str:
    """One name for a slide topic across the question log, quizzes and oral checks."""
    return human_title(topic).casefold()


def _identity(row: Dict[str, Any]) -> Optional[str]:
    return row.get("student_id") or row.get("session_id")


def _days(rows: Iterable[Dict[str, Any]]) -> Set[Any]:
    days = set()
    for row in rows:
        try:
            days.add(datetime.fromisoformat(row["asked_at"]).date())
        except (KeyError, TypeError, ValueError):
            pass
    return days


def _normalised(weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    merged = {name: max(0.0, float((weights or {}).get(name, DEFAULT_WEIGHTS[name]))) for name in SIGNALS}
    total = sum(merged.values())
    if total == 0:
        return dict(DEFAULT_WEIGHTS)
    return {name: w / total for name, w in merged.items()}


def _question_signals(rows: List[Dict[str, Any]], active: int, min_students: int) -> Dict[str, Tuple[Optional[float], str]]:
    out: Dict[str, Tuple[Optional[float], str]] = {name: (None, "") for name in QUESTION_SIGNALS}
    per_student = Counter(i for i in map(_identity, rows) if i)
    asking = len(per_student)

    if asking >= min_students and active >= min_students:
        share = asking / active
        out["breadth"] = (min(1.0, share / BREADTH_SATURATION), f"{asking} of {active} students asked about it ({share:.0%})")
    if asking >= min_students:
        repeat = sum(1 for n in per_student.values() if n >= 2)
        out["repeat"] = (repeat / asking, f"{repeat} of {asking} students asked more than once")

    with_session = [r for r in rows if r.get("session_id")]
    if len(with_session) >= MIN_QUESTIONS:
        follow = sum(1 for r in with_session if r.get("follow_up_of"))
        rate = follow / len(with_session)
        out["follow_up"] = (min(1.0, rate / FOLLOW_UP_SATURATION),
                            f"{follow} of {len(with_session)} questions came back to a topic already asked about in the session")

    with_agent = [r for r in rows if r.get("agent")]
    if len(with_agent) >= MIN_QUESTIONS:
        explain = sum(1 for r in with_agent if r["agent"] == CONCEPT_EXPLAINER)
        out["explain"] = (min(1.0, explain / len(with_agent) / EXPLAIN_SATURATION),
                          f"{explain} of {len(with_agent)} requests were asking for it to be explained")

    days = _days(rows)
    if len(rows) >= MIN_QUESTIONS and days:
        out["persistence"] = (min(1.0, len(days) / PERSISTENCE_DAYS), f"asked about on {len(days)} different day(s)")
    return out


def _quiz_signal(key: str, quizzes_by_id, first, min_students: int) -> Tuple[Optional[float], str]:
    correct = total = 0
    students: Set[str] = set()
    for (student, quiz_id), attempt in first.items():
        quiz = quizzes_by_id.get(quiz_id)
        if quiz and concept_key(quiz["topic"]) == key:
            students.add(student)
            correct += attempt["correct"]
            total += attempt["total"]
    if len(students) < min_students or total == 0:
        return None, ""
    accuracy = correct / total
    return 1.0 - accuracy, f"{len(students)} students took the class quiz; they got {accuracy:.0%} of the questions right"


def _oral_signal(key: str, first_oral: Dict[Tuple[str, str], Any], min_students: int) -> Tuple[Optional[float], str]:
    levels = [a.understanding_level for (student, k), a in first_oral.items() if k == key]
    if len(levels) < min_students:
        return None, ""
    weak = sum(1 for level in levels if level != LEVEL_STRONG)
    return weak / len(levels), f"{weak} of {len(levels)} oral checks were not yet strong"


def _score(signals: Sequence[Signal], means: Dict[str, float], weights: Dict[str, float]) -> float:
    """Weighted average of the signals, with a missing one replaced by the class average for it."""
    used = [name for name in SIGNALS if name in means]
    total = sum(weights[name] for name in used)
    if total == 0:
        return 0.0
    by_name = {s.name: s for s in signals}
    acc = 0.0
    for name in used:
        value = by_name[name].value
        acc += weights[name] * (means[name] if value is None else value)
    return 100.0 * acc / total


score_signals = _score              # public names for other modules (rag_core/intervention.py)
normalise_weights = _normalised


def compute_confusion(
    rows: Sequence[Dict[str, Any]],
    quizzes: Iterable[Dict[str, Any]] = (),
    attempts: Iterable[Dict[str, Any]] = (),
    oral: Iterable[Any] = (),
    weights: Optional[Dict[str, float]] = None,
    min_students: int = MIN_COHORT,
) -> ConfusionReport:
    """
    Score every concept the class has asked about. `rows` are question-log rows (query_log.list_queries),
    `quizzes` / `attempts` are from learning_log (only class quizzes are used), `oral` are OralAssessment
    records. Everything is passed in, so this works the same on real data and on simulated classes.
    """
    weights = _normalised(weights)
    class_quizzes = {q["id"]: q for q in quizzes if q.get("scope", "class") == "class"}
    first = {k: a for k, a in first_attempts(attempts).items() if k[1] in class_quizzes}

    first_oral: Dict[Tuple[str, str], Any] = {}
    for assessment in sorted(oral, key=lambda a: a.id):
        if assessment.student_id:
            first_oral.setdefault((assessment.student_id, concept_key(assessment.topic)), assessment)

    identities = {i for i in map(_identity, rows) if i}
    identities |= {student for student, _quiz in first}
    identities |= {student for student, _key in first_oral}
    active = len(identities)

    by_concept: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    names: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        if row.get("retrieved"):
            raw = topic_title(row["retrieved"][0])
            by_concept[concept_key(raw)].append(row)
            names[concept_key(raw)][human_title(raw)] += 1
    # A concept with only quiz or oral evidence still counts.
    for quiz in class_quizzes.values():
        names[concept_key(quiz["topic"])][human_title(quiz["topic"])] += 0
        by_concept.setdefault(concept_key(quiz["topic"]), [])
    for (_student, key), assessment in first_oral.items():
        names[key][human_title(assessment.topic)] += 0
        by_concept.setdefault(key, [])

    drafts: List[ConceptConfusion] = []
    for key, concept_rows in by_concept.items():
        measured = _question_signals(concept_rows, active, min_students)
        measured["quiz"] = _quiz_signal(key, class_quizzes, first, min_students)
        measured["oral"] = _oral_signal(key, first_oral, min_students)
        signals = [Signal(name, SIGNALS[name][0], weights[name], *measured[name]) for name in SIGNALS]
        sources = int(any(measured[n][0] is not None for n in QUESTION_SIGNALS))
        sources += int(measured["quiz"][0] is not None) + int(measured["oral"][0] is not None)
        drafts.append(ConceptConfusion(
            key=key, concept=names[key].most_common(1)[0][0], questions=len(concept_rows),
            signals=signals, sources=sources,
        ))

    scoreable = [d for d in drafts if d.sources]
    unscored = sorted(((d.concept, d.questions) for d in drafts if not d.sources), key=lambda t: -t[1])

    means: Dict[str, float] = {}
    for name in SIGNALS:
        values = [s.value for d in scoreable for s in d.signals if s.name == name and s.value is not None]
        if values:
            means[name] = sum(values) / len(values)

    for d in scoreable:
        d.score = _score(d.signals, means, weights)
        d.confidence = {1: "Low", 2: "Medium"}.get(d.sources, "High")
        stand = [
            (s.weight * (s.value - means[s.name]), s) for s in d.signals
            if s.value is not None and s.name in means
        ]
        best = max(stand, key=lambda t: t[0], default=None)
        d.stands_out = best[1].evidence if best and best[0] > 0 else ""
    scoreable.sort(key=lambda d: (-d.score, d.concept))
    return ConfusionReport(scoreable, unscored, active, means, weights)


def rank_stability(
    report: ConfusionReport, k: int = 3, draws: int = 300, spread: float = 20.0, seed: int = 0
) -> Dict[str, Stability]:
    """
    How much does the ranking depend on the weights? Re-score with many randomly perturbed weight sets
    (centred on the current weights; a larger `spread` means smaller perturbations) and report, per concept,
    the share of re-weightings in which it stays in the top `k` and its best and worst rank.
    """
    concepts = report.concepts
    if not concepts:
        return {}
    rng = np.random.default_rng(seed)
    names = list(SIGNALS)
    base = np.array([report.weights[n] for n in names]) * spread + 1e-6
    in_top = Counter()
    ranks: Dict[str, List[int]] = defaultdict(list)
    for _ in range(draws):
        sampled = dict(zip(names, rng.dirichlet(base)))
        scored = sorted(concepts, key=lambda c: (-_score(c.signals, report.means, sampled), c.concept))
        for rank, concept in enumerate(scored, start=1):
            ranks[concept.key].append(rank)
            in_top[concept.key] += int(rank <= k)
    return {
        c.key: Stability(in_top[c.key] / draws, min(ranks[c.key]), max(ranks[c.key])) for c in concepts
    }


def load_and_compute(db_dir, weights: Optional[Dict[str, float]] = None) -> ConfusionReport:
    """Read the stored question log, class quizzes, attempts and oral checks, and score the concepts."""
    from rag_core.learning_log import list_attempts, list_quizzes
    from rag_core.oral_assessment import list_oral_assessments
    from rag_core.query_log import list_queries

    return compute_confusion(
        list_queries(db_dir=db_dir), list_quizzes(db_dir=db_dir), list_attempts(db_dir=db_dir),
        list_oral_assessments(db_dir=db_dir), weights,
    )
