"""
Progress Tracker: per-student view of strong and weak topics.

It combines two signals on the same unit (a slide topic):
  - what the student asked about (repeated questions on one topic = struggling)
  - how they scored on quizzes for that topic

There is no LLM call here: the status rules are plain, inspectable code, so a
student or professor can see exactly why a topic is marked weak.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agents.base import Agent, AgentContext, AgentResult
from agents.quiz_generator import grade_answers
from rag_core.insights import topic_title
from rag_core.learning_log import list_attempts, record_attempt
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


def _status(questions: int, correct: int, total: int) -> tuple[str, str]:
    if total:
        accuracy = correct / total
        reason = f"{correct}/{total} correct on quizzes"
        if accuracy >= STRONG_AT:
            return "Strong", reason
        if accuracy >= DEVELOPING_AT:
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
        unanswered = 0
        for row in list_queries(db_dir=ctx.db_dir, student_id=ctx.student_id):
            if row["retrieved"]:
                asked[topic_title(row["retrieved"][0])] += 1
            else:
                unanswered += 1

        scores: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
        for attempt in list_attempts(ctx.student_id, ctx.db_dir):
            scores[attempt["topic"]][0] += attempt["correct"]
            scores[attempt["topic"]][1] += attempt["total"]

        topics: List[TopicProgress] = []
        for topic in set(asked) | set(scores):
            correct, total = scores[topic] if topic in scores else (0, 0)
            status, reason = _status(asked[topic], correct, total)
            topics.append(TopicProgress(topic, asked[topic], correct, total, status, reason))
        topics.sort(key=lambda t: (-_PRIORITY[t.status], -t.questions, t.topic))

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
            data={"topics": topics, "recommended": recommended, "unanswered": unanswered},
        )

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
