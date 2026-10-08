"""
Intervention tracking: did a topic's confusion score change after the professor acted on it?

An "intervention" is something the professor did about one concept: an approved clarification note, a published
remedial quiz, or an action logged by hand (re-taught it in class, updated a slide). This module compares the
concept's confusion score before and after that moment.

Why not just subtract the two scores? Because many things change over time besides what the professor did: the
unit winds down, the exam gets closer, the class gets used to the material. A topic's score can fall for those
reasons alone. So the comparison is built to avoid the obvious traps:

  - Equal windows. The same number of days on each side of the action, so signals that grow with time (how many
    different students asked, how many different days it came up) are not biased towards the longer side.
  - Untouched topics as a yardstick. The same before/after change is computed for the other topics, and the
    effect reported is the topic's change MINUS the typical change of the untouched ones. Anything that moved the
    whole class cancels out. Only topics that started at a SIMILAR level of confusion are used (professors act on
    the highest-scoring topics, and those tend to fall back on their own; low-scoring topics do not), and topics
    that were themselves acted on in the same period are left out. If fewer than two such topics have data, the
    answer is "cannot tell", which is common in a course with few topics.
  - Like with like. Only signals available on BOTH sides are used, for the topic and for the yardstick, so a
    signal that appears or disappears (a quiz taken only after) cannot fake a change.
  - A range, not a number. The students are resampled many times to show how much the result could move by
    chance, and a result is only called a change if that range excludes zero.
  - Waiting is a valid answer. Right after an action there is no "after" yet, and the page says so.

What it cannot do: it cannot prove the action CAUSED a change. The students in the two windows are overlapping
but not identical, the action may coincide with a lecture, and a clarification answers questions directly, so
fewer follow-ups is partly built in. The wording of every verdict says "consistent with", never "caused by".
"""

import copy
import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from rag_core.config import DB_DIR, MIN_COHORT
from rag_core.confusion import (
    MIN_QUESTIONS,
    SIGNALS,
    ConfusionReport,
    compute_confusion,
    concept_key,
    normalise_weights,
    score_signals,
)
from rag_core.insights import topic_title
from rag_core.item_analysis import first_attempts

MIN_WINDOW_DAYS = 1.0       # at least this many days of data on each side of the action
MAX_WINDOW_DAYS = 14.0      # the windows never extend further than this from the action
MIN_COMPARABLE_SIGNALS = 2  # signals available on both sides before a change is claimed
MIN_CONTROLS = 2            # untouched topics (at a similar starting level) needed as a yardstick
CONTROL_BAND = 15.0         # points: a yardstick topic must have started within this of the topic acted on
BOOTSTRAP_DRAWS = 150
INTERVAL = 0.90
MIN_VALID_DRAW_SHARE = 0.7  # share of resamples that must be usable for a range to be reported


@dataclass
class Intervention:
    id: int
    concept_key: str
    concept: str
    kind: str
    title: str
    at: datetime


@dataclass
class SignalChange:
    label: str
    before: str
    after: str


@dataclass
class Effect:
    intervention: Intervention
    status: str                          # "waiting" or "measured"
    message: str = ""                    # why it is waiting, or the plain-language result
    window_days: float = 0.0
    before: Optional[float] = None       # score before (comparable signals only)
    after: Optional[float] = None
    change: Optional[float] = None       # after - before
    control_change: Optional[float] = None   # typical change of the untouched topics
    n_controls: int = 0
    relative: Optional[float] = None     # change - control_change: negative = fell more than the others
    interval: Optional[Tuple[float, float]] = None
    verdict: str = ""
    evidence_kinds: List[str] = field(default_factory=list)   # "questions", "quiz", "oral" actually compared
    signals_used: List[str] = field(default_factory=list)
    signal_changes: List[SignalChange] = field(default_factory=list)


def parse_time(value: Any) -> Optional[datetime]:
    """An ISO timestamp as an aware UTC datetime, or None if it cannot be read."""
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def interventions_from_log(db_dir=DB_DIR) -> List[Intervention]:
    """Every approved clarification, published remedial quiz and logged action, oldest first."""
    from rag_core.advice_log import STATUS_APPROVED, list_advice

    found = []
    for item in list_advice(status=STATUS_APPROVED, db_dir=db_dir):
        at = parse_time(item["decided_at"])
        if at is not None:
            found.append(Intervention(item["id"], item["concept_key"], item["concept"], item["kind"], item["title"], at))
    return sorted(found, key=lambda i: i.at)


def _time_of(item: Any, field_name: str) -> Optional[datetime]:
    value = item[field_name] if isinstance(item, dict) else getattr(item, field_name, None)
    return parse_time(value)


def _slice(rows, attempts, oral, start: datetime, end: datetime):
    def within(items, field_name):
        return [x for x in items if (t := _time_of(x, field_name)) is not None and start <= t < end]

    return within(rows, "asked_at"), within(attempts, "taken_at"), within(oral, "taken_at")


def window_days(at: datetime, rows, attempts, oral, as_of: datetime) -> float:
    """How many days can be compared on each side of `at`: limited by the data before it and the time since."""
    times = [t for t in (
        [_time_of(r, "asked_at") for r in rows] + [_time_of(a, "taken_at") for a in attempts]
        + [_time_of(o, "taken_at") for o in oral]
    ) if t is not None]
    if not times:
        return 0.0
    before = (at - min(times)).total_seconds() / 86400
    after = (as_of - at).total_seconds() / 86400
    return max(0.0, min(before, after, MAX_WINDOW_DAYS))


def _changes(
    before: ConfusionReport, after: ConfusionReport, treated: str, names: Sequence[str], weights: Dict[str, float],
    exclude: Iterable[str],
) -> Optional[Tuple[float, float, int, float]]:
    """(treated change, median control change, number of controls, treated score before) or None if not computable."""
    sub = {n: (weights[n] if n in names else 0.0) for n in SIGNALS}
    b = {c.key: c for c in before.concepts}
    a = {c.key: c for c in after.concepts}
    if treated not in b or treated not in a:
        return None
    skip = set(exclude) | {treated}
    t_before = score_signals(b[treated].signals, before.means, sub)
    change = score_signals(a[treated].signals, after.means, sub) - t_before
    # Only untouched topics that STARTED at a similar level are a fair yardstick. A professor acts on the topics
    # that scored highest, and high scores tend to fall back by themselves (regression to the mean) while low-scoring
    # topics barely move. Comparing with every untouched topic would credit the action with that fall.
    controls = []
    for k in b:
        if k in a and k not in skip:
            start = score_signals(b[k].signals, before.means, sub)
            if abs(start - t_before) <= CONTROL_BAND:
                controls.append(score_signals(a[k].signals, after.means, sub) - start)
    return change, (median(controls) if controls else float("nan")), len(controls), t_before


def _comparable(before: ConfusionReport, after: ConfusionReport, treated: str) -> Tuple[List[str], Dict[str, Any], Dict[str, Any]]:
    b = next((c for c in before.concepts if c.key == treated), None)
    a = next((c for c in after.concepts if c.key == treated), None)
    if b is None or a is None:
        return [], {}, {}
    bs, as_ = {s.name: s for s in b.signals}, {s.name: s for s in a.signals}
    return [n for n in SIGNALS if bs[n].value is not None and as_[n].value is not None and n in before.means
            and n in after.means], bs, as_


def _reading(signal) -> str:
    return f"{signal.value:.0%}" if signal.value is not None else "not enough data"


def _clone(item: Any, new_student: Optional[str]) -> Any:
    """A copy of a row/attempt/oral record belonging to a relabelled student (for resampling)."""
    if isinstance(item, dict):
        clone = dict(item)
        if "student_id" in clone:
            clone["student_id"] = new_student
        if clone.get("session_id") and new_student:
            clone["session_id"] = f"{clone['session_id']}|{new_student}"
        return clone
    if dataclasses.is_dataclass(item):
        return dataclasses.replace(item, student_id=new_student)
    clone = copy.copy(item)
    clone.student_id = new_student
    return clone


def _identity_of(item: Any) -> Optional[str]:
    if isinstance(item, dict):
        return item.get("student_id") or item.get("session_id")
    return getattr(item, "student_id", None)


def _resample(rows, attempts, oral, rng):
    """Draw students with replacement (anonymous questions are each their own unit); keep each draw distinct."""
    units: Dict[Any, Tuple[list, list, list]] = {}
    for i, r in enumerate(rows):
        units.setdefault(_identity_of(r) or ("anon", i), ([], [], []))[0].append(r)
    for a in attempts:
        units.setdefault(_identity_of(a), ([], [], []))[1].append(a)
    for o in oral:
        units.setdefault(_identity_of(o) or ("anon-oral", id(o)), ([], [], []))[2].append(o)
    keys = list(units)
    out_rows, out_attempts, out_oral = [], [], []
    for j, index in enumerate(rng.integers(0, len(keys), size=len(keys))):
        key = keys[index]
        label = None if isinstance(key, tuple) else f"{key}#{j}"
        r, a, o = units[key]
        out_rows += [_clone(x, label) for x in r]
        out_attempts += [_clone(x, label) for x in a]
        out_oral += [_clone(x, label) for x in o]
    return out_rows, out_attempts, out_oral


def measure(
    intervention: Intervention,
    rows: Sequence[Dict[str, Any]],
    quizzes: Sequence[Dict[str, Any]] = (),
    attempts: Sequence[Dict[str, Any]] = (),
    oral: Sequence[Any] = (),
    all_interventions: Iterable[Intervention] = (),
    as_of: Optional[datetime] = None,
    weights: Optional[Dict[str, float]] = None,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = 0,
    min_students: int = MIN_COHORT,
) -> Effect:
    """Compare the concept's score before and after `intervention`. See the module docstring for the method."""
    at, key = intervention.at, intervention.concept_key
    as_of = as_of or datetime.now(timezone.utc)
    effect = Effect(intervention, "waiting")

    # A student's FIRST attempt (and first oral check) is chosen over all time, before the data is split in two.
    # Otherwise a retake of an old quiz after the action would count as a fresh first attempt and carry memory.
    attempts = sorted(first_attempts(attempts).values(), key=lambda a: a["id"])
    seen_oral: set = set()
    first_oral = []
    for o in sorted(oral, key=lambda o: o.id):
        marker = (o.student_id, concept_key(o.topic))
        if o.student_id is None or marker not in seen_oral:
            seen_oral.add(marker)
            first_oral.append(o)
    oral = first_oral

    days = window_days(at, rows, attempts, oral, as_of)
    effect.window_days = days
    if days < MIN_WINDOW_DAYS:
        since = (as_of - at).total_seconds() / 86400
        effect.message = (
            f"Too early to tell: it needs at least {MIN_WINDOW_DAYS:.0f} day of data on each side of the action "
            f"(it was {since:.1f} day(s) ago)."
        ) if since < MIN_WINDOW_DAYS else "Too early to tell: there is not enough data from before the action."
        return effect

    start, end = at - timedelta(days=days), at + timedelta(days=days)
    rows_all, attempts_all, oral_all = _slice(rows, attempts, oral, start, end)

    def report(r, a, o, lo, hi) -> ConfusionReport:
        r, a, o = _slice(r, a, o, lo, hi)
        return compute_confusion(r, quizzes, a, o, weights, min_students)

    before = report(rows_all, attempts_all, oral_all, start, at)
    after = report(rows_all, attempts_all, oral_all, at, end)
    names, bs, as_ = _comparable(before, after, key)
    if len(names) < MIN_COMPARABLE_SIGNALS:
        in_after = [r for r in _slice(rows, attempts, oral, at, end)[0]
                    if r.get("retrieved") and concept_key_of(r) == key]
        effect.message = (
            f"Not enough evidence yet to compare. Since the action there are {len(in_after)} question(s) on this topic; "
            f"a signal needs at least {MIN_QUESTIONS} questions or {min_students} students (on a quiz or oral check) "
            f"in each window, and at least {MIN_COMPARABLE_SIGNALS} kinds of evidence must exist on both sides."
        )
        return effect

    w = normalise_weights(weights)
    others = [
        i.concept_key for i in all_interventions
        if i.id != intervention.id and i.concept_key != key and start <= i.at < end
    ]
    got = _changes(before, after, key, names, w, others)
    effect.status = "measured"
    effect.signals_used = [SIGNALS[n][0] for n in names]
    effect.evidence_kinds = [k for k in ("questions", "quiz", "oral") if any(_kind_of(n) == k for n in names)]
    effect.signal_changes = [SignalChange(SIGNALS[n][0], _reading(bs[n]), _reading(as_[n])) for n in SIGNALS]
    effect.change, control, effect.n_controls, effect.before = got[0], got[1], got[2], got[3]
    effect.after = effect.before + effect.change

    if effect.n_controls < MIN_CONTROLS:
        effect.message = (
            f"The score moved by {effect.change:+.0f}, but fewer than {MIN_CONTROLS} untouched topics that started at a "
            f"similar level (within {CONTROL_BAND:.0f} points) have enough data to compare with, so this cannot be "
            "separated from everything else that changed."
        )
        effect.verdict = "Cannot tell"
        return effect

    effect.control_change = control
    effect.relative = effect.change - control

    rng = np.random.default_rng(seed)
    results = []
    for _ in range(draws):
        r, a, o = _resample(rows_all, attempts_all, oral_all, rng)
        b_i, a_i = report(r, a, o, start, at), report(r, a, o, at, end)
        got_i = _changes(b_i, a_i, key, names, w, others)
        if got_i is not None and got_i[2] >= MIN_CONTROLS:
            results.append(got_i[0] - got_i[1])
    if len(results) >= MIN_VALID_DRAW_SHARE * draws:
        tail = (1 - INTERVAL) / 2 * 100
        effect.interval = (float(np.percentile(results, tail)), float(np.percentile(results, 100 - tail)))

    lo, hi = effect.interval if effect.interval else (None, None)
    if effect.interval is None:
        effect.verdict = "Cannot tell"
        effect.message = "Too few of the re-checks had enough data to give a reliable range."
    elif hi < 0:
        effect.verdict = "Fell more than other topics"
        effect.message = "Consistent with the action helping. It does not prove it."
    elif lo > 0:
        effect.verdict = "Rose more than other topics"
        effect.message = "The action did not help, or something else made this topic harder."
    else:
        effect.verdict = "No clear difference from other topics"
        effect.message = "The change is within what chance alone could produce."
    if effect.evidence_kinds == ["questions"] and effect.verdict != "Cannot tell":
        effect.message += (
            " Only question behaviour could be compared. A clarification changes that directly, because it answers "
            "questions on the spot, so a quiz taken before and after would be stronger evidence."
        )
    return effect


def _kind_of(signal_name: str) -> str:
    return signal_name if signal_name in ("quiz", "oral") else "questions"


def concept_key_of(row: Dict[str, Any]) -> str:
    return concept_key(topic_title(row["retrieved"][0]))


def measure_all(
    db_dir=DB_DIR,
    weights: Optional[Dict[str, float]] = None,
    as_of: Optional[datetime] = None,
    draws: int = BOOTSTRAP_DRAWS,
) -> List[Effect]:
    """Effect of every approved action, most recent first."""
    from rag_core.learning_log import list_attempts, list_quizzes
    from rag_core.oral_assessment import list_oral_assessments
    from rag_core.query_log import list_queries

    interventions = interventions_from_log(db_dir)
    if not interventions:
        return []
    rows, quizzes = list_queries(db_dir=db_dir), list_quizzes(db_dir=db_dir)
    attempts, oral = list_attempts(db_dir=db_dir), list_oral_assessments(db_dir=db_dir)
    return [
        measure(i, rows, quizzes, attempts, oral, interventions, as_of, weights, draws)
        for i in reversed(interventions)
    ]
