"""
Diagnosis: from the evidence about a concept to specific things a professor could do about it.

This is the part of the Content Advisor that needs no language model. Every finding is produced by a plain
rule from numbers the dashboard already shows, and it names the evidence it came from, so a professor can
see WHY it was suggested and disagree. The rules are:

  check_question    a quiz question that strong students tend to get wrong: fix the question before re-teaching
  misconception     one wrong quiz answer drew a large share of the class: address that specific mistake
  prerequisite      an earlier topic in the same file also scores high: students may be missing it
  coming_back       students keep returning or asking for explanations: add an example or simpler wording
  recognise_not_explain
                    students pick the right quiz answer but their oral checks are weak
  wording           students' wording matches the slide poorly: add the terms they use
  thin_evidence     only one kind of evidence backs the score: collect more before changing anything

These are suggestions drawn from patterns in a small class. They say what the data points at, not what
caused it, and every wording below is deliberately hedged.
"""

from collections import Counter
from dataclasses import dataclass, field
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from rag_core.config import FACULTY_SOURCE_NAME, RELEVANCE_SCORE_THRESHOLD
from rag_core.confusion import MIN_QUESTIONS, ConceptConfusion, ConfusionReport, concept_key
from rag_core.insights import cluster_questions, slide_label, topic_title
from rag_core.item_analysis import (
    FLAG_CHECK_KEY,
    QuestionStat,
    analyze_quizzes,
    likely_misconceptions,
)

COMING_BACK_AT = 0.5        # follow-up or explain signal at or above this = students keep returning
RECOGNISE_ORAL_AT = 0.6     # oral weak share at or above this ...
RECOGNISE_QUIZ_AT = 0.4     # ... while the quiz miss rate is at or below this
WORDING_MATCH_FRACTION = 0.8  # mean retrieval distance this close to the relevance cut-off = poor match
MAX_SUB_QUESTIONS = 3
MAX_LANDING = 3


@dataclass
class Finding:
    kind: str
    priority: int       # 1 = look at this first
    title: str
    detail: str         # what the data shows
    action: str         # what a professor could do


@dataclass
class Advice:
    key: str
    concept: str
    score: float
    confidence: str
    findings: List[Finding] = field(default_factory=list)
    landing: List[Tuple[str, int]] = field(default_factory=list)       # (slide label, questions)
    sub_questions: List[Tuple[str, int]] = field(default_factory=list)  # (representative question, group size)
    misconceptions: List[Dict[str, Any]] = field(default_factory=list)  # question, wrong, right, share
    source_file: Optional[str] = None
    first_page: Optional[int] = None


def _location(rows: Sequence[Dict[str, Any]]) -> Tuple[Optional[str], Optional[int]]:
    """The file this concept mostly lives in and the first page of it that students land on."""
    chunks = [
        r["retrieved"][0] for r in rows
        if r["retrieved"] and r["retrieved"][0].get("source") not in (None, FACULTY_SOURCE_NAME)
        and r["retrieved"][0].get("page") is not None
    ]
    if not chunks:
        return None, None
    source = Counter(c["source"] for c in chunks).most_common(1)[0][0]
    return source, min(int(c["page"]) for c in chunks if c["source"] == source)


def _sub_questions(rows: Sequence[Dict[str, Any]], embeddings) -> List[Tuple[str, int]]:
    """What students actually ask, grouped by meaning when an embedding model is available."""
    counts = Counter(" ".join(r["question"].split()) for r in rows)
    texts = list(counts)
    if not texts:
        return []
    if embeddings is None or len(texts) < 2:
        return counts.most_common(MAX_SUB_QUESTIONS)
    groups = cluster_questions(texts, embeddings)
    out = [(texts[g[0]], sum(counts[texts[i]] for i in g)) for g in groups]
    return sorted(out, key=lambda t: -t[1])[:MAX_SUB_QUESTIONS]


def _signal(concept: ConceptConfusion, name: str) -> Optional[float]:
    return next((s.value for s in concept.signals if s.name == name), None)


def _misconception_dict(q: QuestionStat) -> Dict[str, Any]:
    return {
        "question": q.question, "wrong": q.misconception.text, "right": q.correct_option.text,
        "share": q.misconception.share, "students": q.students, "count": q.misconception.count,
        "source": q.source,
    }


def diagnose(
    report: ConfusionReport,
    rows: Sequence[Dict[str, Any]],
    quizzes: Iterable[Dict[str, Any]] = (),
    attempts: Iterable[Dict[str, Any]] = (),
    embeddings=None,
    top_n: int = 5,
) -> List[Advice]:
    """Advice for the `top_n` highest-scoring concepts in `report`, highest first."""
    class_quizzes = [q for q in quizzes if q.get("scope", "class") == "class"]
    analyses = analyze_quizzes(class_quizzes, attempts)
    misconceptions = likely_misconceptions(analyses)
    suspect = [q for a in analyses for q in a.questions if FLAG_CHECK_KEY in q.flags]

    rows_by_key: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("retrieved"):
            rows_by_key.setdefault(concept_key(topic_title(row["retrieved"][0])), []).append(row)

    where = {c.key: _location(rows_by_key.get(c.key, [])) for c in report.concepts}
    cut = median(c.score for c in report.concepts) if report.concepts else 0.0

    out: List[Advice] = []
    for concept in report.concepts[:top_n]:
        rs = rows_by_key.get(concept.key, [])
        source, page = where[concept.key]
        landing = Counter(slide_label(r["retrieved"][0]) for r in rs).most_common(MAX_LANDING)
        advice = Advice(
            key=concept.key, concept=concept.concept, score=concept.score, confidence=concept.confidence,
            landing=landing, sub_questions=_sub_questions(rs, embeddings), source_file=source, first_page=page,
            misconceptions=[_misconception_dict(q) for q in misconceptions if concept_key(q.topic) == concept.key][:2],
        )
        f = advice.findings

        for q in [q for q in suspect if concept_key(q.topic) == concept.key][:2]:
            f.append(Finding(
                "check_question", 1, f"Check quiz question {q.number} before re-teaching",
                f"Students who did well on the other questions tended to get \"{q.question}\" wrong "
                "(a clear reversal, not just a low score). The low result may come from the question, not the topic.",
                "Re-read the question and its marked answer. Fix or remove it, then look at the score again.",
            ))

        for m in advice.misconceptions:
            f.append(Finding(
                "misconception", 2, "Many students share one mistake",
                f"{m['share']:.0%} of the class ({m['count']} of {m['students']}) chose \"{m['wrong']}\" instead of "
                f"\"{m['right']}\" on \"{m['question']}\".",
                "Add an example or a side-by-side contrast of the two ideas. If that wrong option is really "
                "defensible, the question needs rewording instead.",
            ))

        if source is not None and page is not None:
            earlier = [
                (where[o.key][1], o) for o in report.concepts
                if o.key != concept.key and where[o.key][0] == source and where[o.key][1] is not None
                and where[o.key][1] < page and o.score >= cut
            ]
            if earlier:
                _, prev = max(earlier, key=lambda t: t[0])
                f.append(Finding(
                    "prerequisite", 3, "Students may be missing an earlier idea",
                    f"\"{prev.concept}\" comes earlier in {source} and also scores high on confusion "
                    f"({round(prev.score)}). This is a guess from slide order, not a measured dependency.",
                    f"Consider recapping \"{prev.concept}\" before this topic.",
                ))

        back = max(_signal(concept, "follow_up") or 0.0, _signal(concept, "explain") or 0.0)
        if back >= COMING_BACK_AT and landing:
            top_label, top_n_q = landing[0]
            f.append(Finding(
                "coming_back", 4, "Students keep returning to this topic",
                f"{top_n_q} question(s) landed on {top_label}, and students came back with follow-ups or asked "
                "for it to be explained.",
                "Consider a worked example or simpler wording on that slide. A clarification note can cover "
                "the gap until the slide is updated.",
            ))

        oral, quiz = _signal(concept, "oral"), _signal(concept, "quiz")
        if oral is not None and quiz is not None and oral >= RECOGNISE_ORAL_AT and quiz <= RECOGNISE_QUIZ_AT:
            f.append(Finding(
                "recognise_not_explain", 5, "Students recognise the answer but struggle to explain it",
                f"The quiz miss rate is only {quiz:.0%}, but {oral:.0%} of oral checks were not yet strong.",
                "Add short 'explain it in your own words' practice rather than more multiple-choice.",
            ))

        distances = [r["top_score"] for r in rs if r.get("top_score") is not None]
        if len(distances) >= MIN_QUESTIONS and mean(distances) >= WORDING_MATCH_FRACTION * RELEVANCE_SCORE_THRESHOLD:
            examples = "; ".join(f"\"{q}\"" for q, _n in advice.sub_questions[:3])
            f.append(Finding(
                "wording", 6, "Students' wording matches the slide poorly",
                f"Their questions sit close to the point where the notes stop counting as a match "
                f"(average distance {mean(distances):.2f}, cut-off {RELEVANCE_SCORE_THRESHOLD}). They ask: {examples}.",
                "Consider using the terms students use on the slide or in a clarification note.",
            ))

        if concept.confidence == "Low":
            f.append(Finding(
                "thin_evidence", 7, "Only one kind of evidence backs this score",
                "The score rests on a single source (questions, quizzes or oral checks only).",
                "Publish a class quiz on this topic, or wait for more questions, before changing the material.",
            ))
        if not f:
            f.append(Finding(
                "no_cause", 8, "No specific cause stands out",
                "The score is above average but none of the checks found a clear pattern.",
                "Treat it as a topic to watch rather than one to rewrite.",
            ))
        f.sort(key=lambda x: x.priority)
        out.append(advice)
    return out
