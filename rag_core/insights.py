"""
Faculty insight engine: groups similar student questions together.

Students phrase the same doubt in many different ways, so counting exact
question text tells a professor nothing. Here every logged question is
embedded with the same model used for retrieval, and questions whose
embeddings are close are grouped into one cluster ("topic of confusion").

Clustering is agglomerative with a distance threshold, so the number of
clusters is not fixed in advance -- it depends on what students actually ask.
"""

from collections import Counter
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.cluster import AgglomerativeClustering

from rag_core.config import CLUSTER_DISTANCE_THRESHOLD
from rag_core.embeddings import get_embeddings
from rag_core.normalize import human_title, short_section_title


@dataclass
class QuestionCluster:
    """One group of near-duplicate / same-topic questions."""

    label: str                      # the most representative question
    indices: List[int]              # positions in the input list
    size: int = 0
    gap_count: int = 0              # how many of these the notes could not answer
    top_slides: List[str] = field(default_factory=list)  # most-retrieved slides/sections
    topic_titles: List[str] = field(default_factory=list)  # short slide titles, most common first


def cluster_questions(
    questions: Sequence[str],
    embeddings: HuggingFaceEmbeddings | None = None,
    distance_threshold: float = CLUSTER_DISTANCE_THRESHOLD,
) -> List[List[int]]:
    """
    Group `questions` by meaning. Returns a list of index groups, largest
    first. Each group's first index is its most representative question
    (the one closest to the group's centroid).
    """
    if not questions:
        return []
    if len(questions) == 1:
        return [[0]]

    embeddings = embeddings or get_embeddings()
    vectors = np.array(embeddings.embed_documents(list(questions)))

    labels = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
    ).fit_predict(vectors)

    groups: Dict[int, List[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(int(label), []).append(index)

    ordered: List[List[int]] = []
    for members in groups.values():
        centroid = vectors[members].mean(axis=0)
        # Vectors are normalized, so the dot product ranks by cosine similarity.
        closeness = vectors[members] @ centroid
        best = members[int(np.argmax(closeness))]
        ordered.append([best] + [m for m in members if m != best])

    ordered.sort(key=len, reverse=True)
    return ordered


def slide_label(chunk: Dict[str, Any]) -> str:
    page = chunk.get("page")
    label = chunk.get("source", "unknown")
    if page is not None:
        label += f" p.{int(page) + 1}"
    if chunk.get("section"):
        label += f" - {short_section_title(chunk['section'])}"
    return label


def topic_title(chunk: Dict[str, Any]) -> str:
    """Short, file-qualified slide title used as the unit of a 'topic'."""
    if chunk.get("section"):
        return short_section_title(chunk["section"])
    return human_title(Path(chunk.get("source", "unknown")).name)


def summarize_clusters(
    rows: Sequence[Dict[str, Any]],
    embeddings: HuggingFaceEmbeddings | None = None,
    top_slides: int = 3,
) -> List[QuestionCluster]:
    """
    Cluster logged query rows (from `query_log.list_queries`) and attach, per
    cluster, how many were unanswerable and which slides were retrieved most
    often -- i.e. where in the material students keep landing.
    """
    groups = cluster_questions([r["question"] for r in rows], embeddings)

    clusters: List[QuestionCluster] = []
    for group in groups:
        members = [rows[i] for i in group]
        slide_counts = Counter(
            slide_label(chunk) for row in members for chunk in row["retrieved"][:1]
        )
        title_counts = Counter(
            topic_title(chunk) for row in members for chunk in row["retrieved"][:1]
        )
        clusters.append(
            QuestionCluster(
                label=rows[group[0]]["question"],
                indices=group,
                size=len(group),
                gap_count=sum(1 for r in members if not r["grounded"]),
                top_slides=[s for s, _ in slide_counts.most_common(top_slides)],
                topic_titles=[t for t, _ in title_counts.most_common(top_slides)],
            )
        )
    return clusters
