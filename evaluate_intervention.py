"""
evaluate_intervention.py
========================
When the professor acts on a topic, how often does each way of checking claim "it helped", and how often is
that claim right?

It simulates two-week classes (8 topics; 4 planted as confusing, 1 as merely popular) in which the professor acts
on one confusing topic at the start of day 8. Two things are varied:

  effect   how much the action really reduces that topic's confusion afterwards (0 = it does nothing)
  drift    how much EVERY topic's confusion falls over the two weeks anyway (the unit winds down, students
           get used to the material). With drift, a topic improves whether or not anyone did anything.

Three checks are compared on the same classes:

  naive before/after   subtract the topic's score before the action from its score after; claim "helped" if it fell
                       by 5 points or more. No other topics, no uncertainty.
  vs other topics      the same, but minus the change in untouched topics that started at a similar level (the
                       point estimate only).
  vs other topics + range
                       what the app reports: the effect against those topics, with a resampled range, and a
                       claim only when the whole range is below zero.

A check is good if it claims "helped" often when the action really worked (effect > 0) and rarely when it did
not (effect = 0), including when everything is drifting down. Cells show the share of classes in which the check
claimed "helped".

READ THIS BEFORE QUOTING ANY NUMBER: the students come from a model the project authors wrote. Confusion is made to
cause follow-ups, explanation requests, quiz misses and weak oral checks, and the drift and the size of the effect
are chosen by us. This shows how the checks behave under those assumptions. It does not show that a real action
has any effect, or that real classes drift this way.

Usage:  python evaluate_intervention.py [classes_per_cell] [resamples_per_class]
"""

import random
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

from rag_core.advice_log import KIND_ACTION
from rag_core.confusion import compute_confusion, concept_key
from rag_core.intervention import Intervention, measure
from rag_core.oral_assessment import LEVEL_DEVELOPING, LEVEL_STRONG

CONCEPTS = 8
UTC = timezone.utc
ACTION_AT = datetime(2026, 10, 8, tzinfo=UTC)       # start of day 8
AS_OF = datetime(2026, 10, 15, tzinfo=UTC)
CLAIM_AT = -5.0                                      # points: a fall of at least this counts as "helped"
NAMES = [f"Concept {j} Topic" for j in range(CONCEPTS)]


def simulate(size: int, effect: float, drift: float, seed: int):
    """One class. Topic 0 is confusing and gets the action; topics 1 to 3 are confusing too; topic 4 is merely popular."""
    rng = random.Random(seed)
    confusing = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
    popular = 4
    treated = 0

    def confusion(j: int, day: int) -> float:
        level = confusing.get(j, 0.0) * (1 - drift * (day - 1) / 13)
        if j == treated and day >= 8:
            level *= 1 - effect
        return max(level, 0.0)

    weights = [3.0 if j == popular else 1.5 if j in confusing else 1.0 for j in range(CONCEPTS)]
    rows, attempts, oral = [], [], []
    quizzes = [{"id": j + 1, "topic": NAMES[j], "scope": "class", "questions": []} for j in range(CONCEPTS)]
    for s in range(size):
        student = f"student-{s}"
        for day in range(1, 15):
            if rng.random() < 0.35:                                   # this student studies one topic today
                j = rng.choices(range(CONCEPTS), weights)[0]
                c = confusion(j, day)
                for k in range(1 + (rng.random() < 0.05 + 0.50 * c)):
                    rows.append({
                        "id": len(rows) + 1, "student_id": student, "session_id": f"{student}-d{day}",
                        "asked_at": f"2026-10-{day:02d}T10:{3 * k:02d}:00+00:00",
                        "agent": "Concept Explainer" if rng.random() < 0.05 + 0.30 * c else "Doubt Resolver",
                        "follow_up_of": 1 if k else None, "question": f"question about {NAMES[j]}", "top_score": 0.5,
                        "retrieved": [{"source": "notes.pdf", "page": j * 3, "section": NAMES[j]}],
                    })
        for j in range(CONCEPTS):
            day = rng.randint(1, 14)
            if rng.random() < 0.8:
                right = sum(rng.random() < 0.75 - 0.40 * confusion(j, day) for _ in range(4))
                attempts.append({"id": len(attempts) + 1, "student_id": student, "quiz_id": j + 1, "topic": NAMES[j],
                                 "correct": right, "total": 4, "answers": [],
                                 "taken_at": f"2026-10-{day:02d}T12:00:00+00:00"})
            day = rng.randint(1, 14)
            if rng.random() < 0.25:
                weak = rng.random() < 0.20 + 0.45 * confusion(j, day)
                oral.append(SimpleNamespace(
                    id=len(oral) + 1, student_id=student, topic=NAMES[j], taken_at=f"2026-10-{day:02d}T14:00:00+00:00",
                    understanding_level=LEVEL_DEVELOPING if weak else LEVEL_STRONG))
    return rows, quizzes, attempts, oral, Intervention(1, concept_key(NAMES[treated]), NAMES[treated], KIND_ACTION, "re-taught", ACTION_AT)


def naive_change(rows, quizzes, attempts, oral, key) -> float | None:
    """Score after minus score before for the topic alone: all signals, no other topics, no range."""
    def score(lo, hi):
        pick = lambda items, f: [x for x in items if lo <= datetime.fromisoformat(
            x[f] if isinstance(x, dict) else getattr(x, f)) < hi]
        report = compute_confusion(pick(rows, "asked_at"), quizzes, pick(attempts, "taken_at"), pick(oral, "taken_at"))
        return next((c.score for c in report.concepts if c.key == key), None)

    before, after = score(datetime(2026, 10, 1, tzinfo=UTC), ACTION_AT), score(ACTION_AT, AS_OF)
    return None if before is None or after is None else after - before


def evaluate(size: int, effect: float, drift: float, classes: int, draws: int) -> dict:
    claims = {"naive": 0, "point": 0, "range": 0}
    answered = 0
    for seed in range(classes):
        rows, quizzes, attempts, oral, iv = simulate(size, effect, drift, seed)
        change = naive_change(rows, quizzes, attempts, oral, iv.concept_key)
        result = measure(iv, rows, quizzes, attempts, oral, [iv], AS_OF, draws=draws, seed=seed)
        if change is not None and result.status == "measured" and result.relative is not None:
            answered += 1
            claims["naive"] += change <= CLAIM_AT
            claims["point"] += result.relative <= CLAIM_AT
            claims["range"] += result.verdict == "Fell more than other topics"
    return {"answered": answered, **{k: (v / answered if answered else float("nan")) for k, v in claims.items()}}


def main() -> None:
    classes = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    draws = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    size = 48
    print(f"{classes} simulated classes of {size} per cell, {draws} resamples each. "
          "Cells: share of classes in which the check claimed the action helped.\n")
    print(f"{'real effect':>12} {'drift':>6} | {'naive before/after':>19} {'vs other topics':>16} {'+ range (app)':>14} | classes judged")
    for drift in (0.0, 0.6):
        for effect in (0.0, 0.4, 0.8):
            r = evaluate(size, effect, drift, classes, draws)
            print(f"{effect:>12.1f} {drift:>6.1f} | {r['naive']:>19.0%} {r['point']:>16.0%} {r['range']:>14.0%} | {r['answered']}/{classes}")
        print()


if __name__ == "__main__":
    main()
