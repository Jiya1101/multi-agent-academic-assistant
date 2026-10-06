"""
Shared contract for every agent.

An agent is a unit with one job, its own tools (retrieval, the question log,
the quiz tables, an LLM) and a uniform way to be called and to report back.
Agents do not call each other directly. The Orchestrator decides who runs
next, and hands one agent's output to another, so the flow is visible and
testable in one place.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from rag_core.config import DB_DIR


@dataclass
class AgentContext:
    """Everything an agent may need for one request."""

    vectorstore: FAISS
    db_dir: Path = DB_DIR
    student_id: Optional[str] = None            # None = anonymous
    source_filenames: Optional[Sequence[str]] = None  # None = search all files
    llm: Any = None                             # chat-model override (used by tests)
    embeddings: Any = None                      # embedding-model override


@dataclass
class AgentResult:
    """What an agent hands back to the Orchestrator and, through it, the UI."""

    agent: str
    kind: str  # answer | explanation | quiz | progress | insight | gap | clarify
    text: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    sources: List[Document] = field(default_factory=list)
    grounded: bool = True  # False when the notes could not support a response


class Agent(ABC):
    name: str = "Agent"
    job: str = ""

    @abstractmethod
    def run(self, request: str, ctx: AgentContext, **kwargs: Any) -> AgentResult:
        """Do this agent's job for `request`."""
