"""
Progress Tracker: per-student view of strong and weak topics.

It combines two signals on the same unit (a slide topic):
  - what the student asked about (repeated questions on one topic = struggling)
  - how they scored on quizzes for that topic

There is no LLM call here: the status rules are plain, inspectable code, so a
student or professor can see exactly why a topic is marked weak.
"""

from collections import Counter, defaultdict
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.base import Agent, AgentContext, AgentResult
from agents.quiz_generator import grade_answers
from rag_core.insights import topic_title
from agents.note_generator import group_material
from rag_core.config import FACULTY_SOURCE_NAME
from rag_core.learning_log import get_quiz, list_attempts, record_attempt
from rag_core.normalize import human_title
from rag_core.oral_assessment import list_oral_assessments
from rag_core.query_log import list_queries

STRONG_AT = 0.8      # quiz accuracy at or above this = Strong
DEVELOPING_AT = 0.5  # at or above this (but below STRONG_AT) = Developing

# Higher = study this first. "Strong" never gets recommended.
_PRIORITY = {"Weak": 3, "Needs practice": 2, "Developing": 1, "Not yet tested": 0.5, "Strong": 0}


@dataclass
class TopicProgress:
    topic: str
    questions: int
    quiz_correct: int
    quiz_total: int
    status: str
    reason: str
    asked: List[str] = field(default_factory=list)   # the questions asked, newest last
    oral: str = ""                                   # latest oral check result for this topic
    oral_question: str = ""                          # what the oral check asked
    file: str = ""                                   # the uploaded material it belongs to


def _status(questions: int, correct: int, total: int, oral_scores: List[float] | None = None) -> tuple[str, str]:
    if total:
        accuracy = correct / total
        reason = f"{correct}/{total} correct on quizzes"
        if accuracy >= STRONG_AT:
            return "Strong", reason
        if accuracy >= DEVELOPING_AT:
            return "Developing", reason
        return "Weak", reason
    if oral_scores:
        average = sum(oral_scores) / len(oral_scores)
        reason = f"{average:.0%} average on {len(oral_scores)} oral check(s)"
        if average >= STRONG_AT:
            return "Strong", reason
        if average >= DEVELOPING_AT:
            return "Developing", reason
        return "Weak", reason
    if questions >= 2:
        return "Needs practice", f"asked {questions} questions on this, no quiz taken yet"
    return "Not yet tested", f"asked {questions} question(s), no quiz taken yet"


class ProgressTracker(Agent):
    name = "Progress Tracker"
    job = "Combine a student's questions and quiz scores into strong/weak topics."

    def run(self, request: str, ctx: AgentContext, **_) -> AgentResult:
        if not ctx.student_id:
            return AgentResult(
                agent=self.name, kind="progress", grounded=False,
                text="Enter a Student ID in the sidebar to track your progress.",
                data={"topics": [], "recommended": None},
            )

        asked: Counter = Counter()
        questions: Dict[str, List[str]] = defaultdict(list)
        unanswered = 0
        for row in list_queries(db_dir=ctx.db_dir, student_id=ctx.student_id):
            if row["retrieved"]:
                topic = topic_title(row["retrieved"][0])
                asked[topic] += 1
                if row["question"] not in questions[topic]:
                    questions[topic].append(row["question"])
            else:
                unanswered += 1

        scores: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
        for attempt in list_attempts(ctx.student_id, ctx.db_dir):
            scores[attempt["topic"]][0] += attempt["correct"]
            scores[attempt["topic"]][1] += attempt["total"]

        oral_scores: Dict[str, List[float]] = defaultdict(list)
        oral_latest: Dict[str, Any] = {}
        for assessment in list_oral_assessments(student_id=ctx.student_id, db_dir=ctx.db_dir):
            oral_scores[assessment.topic].append(float(assessment.content_score))
            oral_latest[assessment.topic] = assessment  # oldest first, so the last one wins

        topics: List[TopicProgress] = []
        for topic in set(asked) | set(scores) | set(oral_scores):
            correct, total = scores[topic] if topic in scores else (0, 0)
            status, reason = _status(asked[topic], correct, total, oral_scores.get(topic))
            latest = oral_latest.get(topic)
            topics.append(TopicProgress(
                topic, asked[topic], correct, total, status, reason,
                asked=questions.get(topic, []),
                oral=f"{latest.understanding_level} ({float(latest.content_score):.0%})" if latest else "",
                oral_question=latest.prompt if latest else "",
            ))
        topics.sort(key=lambda t: (-_PRIORITY[t.status], -t.questions, t.topic))

        materials = self._by_material(ctx)
        recommended = next((t for t in topics if _PRIORITY[t.status] > 0.5), None)
        if not topics:
            text = "No activity yet. Ask a question or take a quiz and your progress will appear here."
        elif recommended:
            text = f"Focus next on: {recommended.topic} ({recommended.status.lower()}: {recommended.reason})."
        else:
            text = "No weak spots found so far."
        return AgentResult(
            agent=self.name,
            kind="progress",
            text=text,
            data={"topics": topics, "recommended": recommended, "unanswered": unanswered, "materials": materials},
        )

    def _by_material(self, ctx: AgentContext) -> List[Dict[str, Any]]:
        """The same progress, split by the uploaded file it belongs to: quizzes taken and topic cards."""
        student = ctx.student_id
        mats: Dict[str, Dict[str, Any]] = {}
        # Only files still in the library get a section; records of deleted files are ignored, never shown as "Other".
        library = {
            Path(str(d.metadata.get("source", ""))).name for d in ctx.vectorstore.docstore._dict.values()
        } - {FACULTY_SOURCE_NAME}

        def mat(file: str) -> Optional[Dict[str, Any]]:
            if file not in library:
                return None
            return mats.setdefault(file, {"file": file, "quizzes": [], "asked": defaultdict(list), "scores": {}, "oral": {}})

        for row in list_queries(db_dir=ctx.db_dir, student_id=student):
            if row["retrieved"]:
                first = row["retrieved"][0]
                file = Path(first.get("source", "") or "Other").name
                topic = topic_title(first)
                m = mat(file)
                if m is not None and row["question"] not in m["asked"][topic]:
                    m["asked"][topic].append(row["question"])

        for attempt in list_attempts(student, ctx.db_dir):
            quiz = get_quiz(attempt["quiz_id"], ctx.db_dir)
            files = Counter(
                Path(q.get("source", {}).get("file", "")).name for q in (quiz or {}).get("questions", [])
                if q.get("source", {}).get("file")
            )
            m = mat(files.most_common(1)[0][0]) if files else None
            if m is None:
                continue
            targeted = bool(quiz and str(quiz.get("prompt", "")).startswith("Quiz on "))
            accuracy = attempt["correct"] / max(attempt["total"], 1)
            m["quizzes"].append({
                "label": f"Quiz {len(m['quizzes']) + 1}",
                "topic": attempt["topic"] if targeted else "",
                "correct": attempt["correct"], "total": attempt["total"],
                "status": "Strong" if accuracy >= STRONG_AT else "Developing" if accuracy >= DEVELOPING_AT else "Weak",
            })
            if targeted:
                got = m["scores"].setdefault(attempt["topic"], [0, 0])
                got[0] += attempt["correct"]
                got[1] += attempt["total"]

        heading_file = {}
        try:
            everything = AgentContext(vectorstore=ctx.vectorstore, db_dir=ctx.db_dir)
            for group in group_material(everything):
                if group["title"]:
                    heading_file.setdefault(human_title(group["title"]), group["file"])
        except Exception:
            pass
        for assessment in list_oral_assessments(student_id=student, db_dir=ctx.db_dir):
            m = mat(heading_file.get(assessment.topic, ""))
            if m is not None:
                m["oral"][assessment.topic] = assessment

        out = []
        for file in sorted(mats):
            m = mats[file]
            names = list(dict.fromkeys([*m["asked"], *m["scores"], *m["oral"]]))
            cards = []
            for topic in names:
                correct, total = m["scores"].get(topic, (0, 0))
                oral = m["oral"].get(topic)
                status, reason = _status(
                    len(m["asked"].get(topic, [])), correct, total,
                    [float(oral.content_score)] if oral else None,
                )
                cards.append(TopicProgress(
                    topic, len(m["asked"].get(topic, [])), correct, total, status, reason,
                    asked=m["asked"].get(topic, []),
                    oral=f"{oral.understanding_level} ({float(oral.content_score):.0%})" if oral else "",
                    oral_question=oral.prompt if oral else "", file=file,
                ))
            cards.sort(key=lambda t: (-_PRIORITY[t.status], t.topic))
            weakest = next((t for t in cards if _PRIORITY[t.status] > 0.5), None)
            weak_quiz = next((q for q in sorted(m["quizzes"], key=lambda q: q["correct"] / max(q["total"], 1))
                              if q["status"] != "Strong"), None)
            if weakest:
                advice = f"Focus next on: {human_title(weakest.topic)} ({weakest.status.lower()}: {weakest.reason})."
            elif weak_quiz:
                advice = (f"Focus next on: revising this material. {weak_quiz['label']} was "
                          f"{weak_quiz['correct']}/{weak_quiz['total']}; take a new quiz when you are ready.")
            elif cards or m["quizzes"]:
                advice = "No weak spots found in this material so far."
            else:
                advice = "No activity on this material yet."
            out.append({"file": file, "topics": cards, "quizzes": m["quizzes"], "advice": advice})
        return out

    def record_attempt(
        self, quiz: Dict[str, Any], answers: List[Optional[int]], ctx: AgentContext
    ) -> Dict[str, Any]:
        """Grade a quiz attempt and, for an identified student, save it."""
        graded = grade_answers(quiz["questions"], answers)
        saved = bool(ctx.student_id)
        if saved:
            record_attempt(
                ctx.student_id, quiz["id"], quiz["topic"],
                graded["correct"], graded["total"], answers, ctx.db_dir,
            )
        return {**graded, "saved": saved, "topic": quiz["topic"]}
