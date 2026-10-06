"""
Gap Handler: takes over when the notes could not answer a question.

It does not answer. It places the question in the professor's queue (the
Doubt Resolver already logged it as a gap) and checks how many other open
questions mean the same thing, so a topic many students are stuck on is
flagged as a priority instead of looking like a one-off.
"""

import numpy as np

from agents.base import Agent, AgentContext, AgentResult
from rag_core.config import CLUSTER_DISTANCE_THRESHOLD
from rag_core.embeddings import get_embeddings
from rag_core.query_log import STATUS_GAP, list_queries


class GapHandler(Agent):
    name = "Gap Handler"
    job = "Queue an unanswered question for the professor and spot repeated gaps."

    def run(self, request: str, ctx: AgentContext, log_id: int | None = None, **_) -> AgentResult:
        others = [
            row
            for row in list_queries(status=STATUS_GAP, db_dir=ctx.db_dir)
            if row["id"] != log_id
        ]
        similar = 0
        if others:
            embeddings = ctx.embeddings or get_embeddings()
            vectors = np.array(embeddings.embed_documents([request] + [r["question"] for r in others]))
            cosine_distance = 1.0 - vectors[1:] @ vectors[0]
            similar = int((cosine_distance < CLUSTER_DISTANCE_THRESHOLD).sum())

        if similar:
            text = (
                f"{similar} other student question(s) on this are already waiting. "
                "It has been added to the same topic in the professor's queue."
            )
        else:
            text = "Your question has been sent to the professor."
        return AgentResult(
            agent=self.name,
            kind="gap",
            text=text,
            data={"similar_open_gaps": similar, "log_id": log_id},
        )
