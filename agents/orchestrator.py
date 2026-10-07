"""
Orchestrator: routes a request to the right agent and hands work between them.

Routing (agents/routing.py) is done by a hybrid: a classifier trained on labelled
example messages decides from the MEANING of the message, and hand-written
rules take over only when the classifier is unsure. No LLM call is involved, so
it is instant. Students can flag a wrong route; those corrections are stored
and the classifier is retrained on them.

Hand-offs implemented here (each is recorded in the returned `trace`):
  Doubt Resolver / Concept Explainer -> Gap Handler       notes do not cover it
  Progress Tracker -> Quiz Generator                      "quiz me" with no topic
  Progress Tracker -> Concept Explainer                   "what should I study next"
  Faculty Insight  -> Quiz Generator                      class quiz from class confusion
"""

from dataclasses import dataclass
from typing import List

from agents.base import AgentContext, AgentResult
from agents.concept_explainer import ConceptExplainer
from agents.doubt_resolver import DoubtResolver
from agents.faculty_insight import FacultyInsight
from agents.gap_handler import GapHandler
from agents.note_generator import NoteGenerator
from agents.oral_assessor import OralAssessor
from agents.progress_tracker import ProgressTracker
from agents.quiz_generator import QuizGenerator
from agents.routing import Route, default_router, route_message
from rag_core.learning_log import SCOPE_CLASS, SCOPE_PERSONAL, list_quizzes


@dataclass
class OrchestratorResult:
    route: Route
    results: List[AgentResult]
    trace: List[str]
    message: str = ""

    @property
    def primary(self) -> AgentResult:
        """The result the student should see first (the last non-gap agent)."""
        shown = [r for r in self.results if r.kind != "gap"]
        return (shown or self.results)[-1]


class Orchestrator:
    def __init__(self, router=None) -> None:
        # Any object with route(message, db_dir) -> Route. Tests inject a RuleRouter.
        self.router = router if router is not None else default_router()
        self.doubt_resolver = DoubtResolver()
        self.concept_explainer = ConceptExplainer()
        self.quiz_generator = QuizGenerator()
        self.note_generator = NoteGenerator()
        self.oral_assessor = OralAssessor()
        self.progress_tracker = ProgressTracker()
        self.faculty_insight = FacultyInsight()
        self.gap_handler = GapHandler()

    # ------------------------------------------------------------------ #
    # Student requests                                                   #
    # ------------------------------------------------------------------ #
    def handle(self, message: str, ctx: AgentContext) -> OrchestratorResult:
        route = self.router.route(message, ctx.db_dir)
        how = route.source if route.confidence is None else f"{route.source}, {route.confidence:.0%} sure"
        trace = [f"Router -> {route.name} ({how}): {route.reason}"]
        results: List[AgentResult] = []

        if route.name == "doubt":
            results.append(self.doubt_resolver.run(message, ctx))
        elif route.name == "explain":
            results.append(self.concept_explainer.run(message, ctx, level=route.params["level"]))
        elif route.name == "notes":
            topics = route.params.get("topics") or []
            if topics:
                results.append(self.note_generator.run(message, ctx, topics=topics))
            else:
                results.append(AgentResult(
                    agent="Orchestrator", kind="clarify", grounded=False,
                    text="Which topic? For example: 'make notes on ACID and BASE'.",
                ))
        elif route.name == "progress":
            results.append(self.progress_tracker.run(message, ctx))
        elif route.name == "quiz":
            self._quiz(route, message, ctx, results, trace)
        elif route.name == "study_next":
            self._study_next(ctx, results, trace)

        # Hand-off: a student-facing agent could not ground its response.
        last = results[-1] if results else None
        if last and last.kind in ("answer", "explanation", "notes") and not last.grounded:
            trace.append(f"{last.agent} -> Gap Handler: the notes do not cover this")
            results.append(
                self.gap_handler.run(message, ctx, log_id=last.data.get("log_id"))
            )
        return OrchestratorResult(route, results, trace, message)

    def _quiz(self, route, message, ctx, results, trace) -> None:
        topic = route.params["topic"]
        if not topic:
            progress = self.progress_tracker.run(message, ctx)
            results.append(progress)
            weakest = progress.data.get("recommended")
            if weakest is None:
                results.append(AgentResult(
                    agent="Orchestrator", kind="clarify", grounded=False,
                    text="Which topic? For example: 'quiz me on RDS read replicas'.",
                ))
                trace.append("Progress Tracker: no weak topic to pick, asking the student for one")
                return
            trace.append(f"Progress Tracker -> Quiz Generator: weakest topic is '{weakest.topic}'")
            topic = weakest.topic
        results.append(self.quiz_generator.run(topic, ctx, scope=SCOPE_PERSONAL))

    def _study_next(self, ctx, results, trace) -> None:
        progress = self.progress_tracker.run("", ctx)
        results.append(progress)
        weakest = progress.data.get("recommended")
        if weakest is None:
            return
        trace.append(f"Progress Tracker -> Concept Explainer: weakest topic is '{weakest.topic}'")
        # log=False: this request was generated by the system, not asked by the student.
        results.append(self.concept_explainer.run(
            f"Explain {weakest.topic} step by step", ctx, level="beginner", log=False
        ))

    # ------------------------------------------------------------------ #
    # Professor pipeline                                                 #
    # ------------------------------------------------------------------ #
    def publish_class_quizzes(
        self, ctx: AgentContext, top_n: int = 3, n_questions: int = 3
    ) -> OrchestratorResult:
        """Faculty Insight finds what the class is confused about; Quiz Generator quizzes it."""
        trace = ["Professor action: publish quizzes from class confusion"]
        insight = self.faculty_insight.run("", ctx, top_n=top_n)
        results: List[AgentResult] = [insight]
        topics = insight.data.get("confusion_topics", [])
        trace.append(f"Faculty Insight: found {len(topics)} confusion topic(s)")

        existing = {q["topic"] for q in list_quizzes(SCOPE_CLASS, ctx.db_dir)}
        for item in topics:
            if item.topic in existing:
                trace.append(f"Faculty Insight -> Quiz Generator: skipped '{item.topic}' (quiz already published)")
                continue
            trace.append(
                f"Faculty Insight -> Quiz Generator: '{item.topic}' ({item.size} student questions)"
            )
            result = self.quiz_generator.run(
                item.query, ctx, n_questions=n_questions, scope=SCOPE_CLASS, topic=item.topic
            )
            results.append(result)
            if result.data.get("error"):
                trace.append(f"Quiz Generator: no quiz for '{item.topic}' ({result.data['error']})")
        return OrchestratorResult(Route("publish_class_quizzes", "professor pipeline"), results, trace)
