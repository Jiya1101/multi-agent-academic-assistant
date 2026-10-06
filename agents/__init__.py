"""
agents
======
The multi-agent layer. Each agent has one job and its own tools; the
Orchestrator routes requests and hands work between agents. All of them sit
on the shared retrieval layer in `rag_core`.
"""

from agents.base import Agent, AgentContext, AgentResult
from agents.concept_explainer import ConceptExplainer
from agents.doubt_resolver import DoubtResolver
from agents.faculty_insight import ConfusionTopic, FacultyInsight
from agents.gap_handler import GapHandler
from agents.orchestrator import Orchestrator, OrchestratorResult
from agents.routing import HybridRouter, LearnedRouter, Route, RuleRouter, route_message
from agents.note_generator import NoteGenerator
from agents.progress_tracker import ProgressTracker, TopicProgress
from agents.quiz_generator import QuizGenerator, grade_answers

__all__ = [
    "Agent", "AgentContext", "AgentResult",
    "ConceptExplainer", "DoubtResolver", "FacultyInsight", "ConfusionTopic",
    "GapHandler", "ProgressTracker", "TopicProgress", "QuizGenerator", "grade_answers", "NoteGenerator",
    "Orchestrator", "OrchestratorResult", "Route", "route_message",
    "HybridRouter", "LearnedRouter", "RuleRouter",
]
