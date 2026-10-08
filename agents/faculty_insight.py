"""
Faculty Insight: reads the whole class's question log and reports where the
class is confused.

Its key output for the rest of the system is `confusion_topics`: topics many
students asked about that the notes DID cover, i.e. material that is in the
notes but is not landing. The Orchestrator hands these to the Quiz Generator.
Topics the notes did not cover are reported separately as gaps.
"""

from dataclasses import dataclass
from typing import List

from agents.base import Agent, AgentContext, AgentResult
from rag_core.confusion import load_and_compute
from rag_core.embeddings import get_embeddings
from rag_core.insights import QuestionCluster, summarize_clusters
from rag_core.query_log import list_queries


@dataclass
class ConfusionTopic:
    topic: str   # slide title the questions kept landing on
    query: str   # a representative student question, used to retrieve for a quiz
    size: int    # how many questions in the cluster


class FacultyInsight(Agent):
    name = "Faculty Insight"
    job = "Cluster the class's questions and find topics the class keeps asking about."

    def concept_report(self, ctx: AgentContext, weights=None):
        """Score every concept from all the evidence (no embeddings needed). The Content Advisor builds on this."""
        return load_and_compute(ctx.db_dir, weights)

    def run(
        self,
        request: str,
        ctx: AgentContext,
        top_n: int = 3,
        min_size: int = 2,
        **_,
    ) -> AgentResult:
        rows = list_queries(db_dir=ctx.db_dir)
        if not rows:
            return AgentResult(
                agent=self.name, kind="insight", text="No questions logged yet.",
                data={"clusters": [], "confusion_topics": [], "gap_clusters": [], "rows": rows, "concept_report": None},
            )

        clusters: List[QuestionCluster] = summarize_clusters(rows, ctx.embeddings or get_embeddings())

        confusion: List[ConfusionTopic] = []
        seen = set()
        for cluster in clusters:  # already largest first
            if cluster.gap_count or cluster.size < min_size or not cluster.topic_titles:
                continue
            topic = cluster.topic_titles[0]
            if topic in seen:
                continue
            seen.add(topic)
            confusion.append(ConfusionTopic(topic=topic, query=cluster.label, size=cluster.size))
            if len(confusion) == top_n:
                break

        gap_clusters = [c for c in clusters if c.gap_count]
        concept_report = load_and_compute(ctx.db_dir)
        text = (
            f"{len(rows)} questions in {len(clusters)} topics. "
            f"{len(confusion)} topic(s) the notes cover but the class keeps asking about; "
            f"{len(gap_clusters)} topic(s) the notes do not cover."
        )
        return AgentResult(
            agent=self.name,
            kind="insight",
            text=text,
            data={
                "clusters": clusters,
                "confusion_topics": confusion,
                "gap_clusters": gap_clusters,
                "rows": rows,
                # Several kinds of evidence combined per concept (rag_core/confusion.py). The quiz hand-off above
                # still uses cluster size; switching it to this score is a separate, deliberate change.
                "concept_report": concept_report,
            },
        )
