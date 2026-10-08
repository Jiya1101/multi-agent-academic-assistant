"""
Oral Assessor: asks a short spoken-answer question and scores the response.

This is intentionally lighter than training a speech-understanding model. The
agent uses existing course retrieval for grounding, the local LLM for question
generation and content scoring, and `rag_core.audio_metrics` for pause/fluency
signals. The final understanding level is stored in SQLite.
"""

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agents.base import Agent, AgentContext, AgentResult
from agents.note_generator import group_material, major_headings
from rag_core.audio_metrics import analyze_audio_response
from rag_core.chain import NOT_COVERED_MESSAGE, format_context, format_type_definitions, retrieve, type_definitions
from rag_core.config import FACULTY_SOURCE_NAME
from rag_core.insights import topic_title
from rag_core.llm import get_json_llm
from rag_core.normalize import bullet_points, human_title, is_title_like, readable_sentences
from rag_core.oral_assessment import (
    infer_understanding_level,
    record_oral_assessment,
)

_QUESTION_SYSTEM = """You write oral check questions for a university student.

Use ONLY the course-note context below.
Write exactly one question about: {topic}

The question must:
- ask the student to explain in their own words;
- be answerable from the context alone;
- be short enough to answer in 30-60 seconds;
- not be multiple choice.

Return ONLY JSON in this shape:
{{"question": "..."}}

--- CONTEXT ---
{context}
"""

_SCORE_SYSTEM = """You score a student's spoken answer to an oral check.

Use ONLY the course-note context below and the expected question.
Do not reward outside knowledge unless the context supports it.

Return ONLY JSON in this shape:
{{
  "content_score": 0.0,
  "feedback": "one or two short sentences naming what was correct and what was missing"
}}

Scoring guide:
- 0.85-1.00: accurate and covers the key ideas from the context.
- 0.60-0.84: mostly correct but misses an important point or has a small confusion.
- 0.35-0.59: partly related but incomplete or vague.
- 0.00-0.34: incorrect, off-topic, or too little to judge.

--- CONTEXT ---
{context}
"""

_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _QUESTION_SYSTEM), ("human", "Write the oral check question.")]
)
_SCORE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SCORE_SYSTEM),
        ("human", "--- QUESTION ---\n{question}\n\n--- STUDENT ANSWER ---\n{answer}"),
    ]
)


def _json_object(raw: str) -> Dict[str, Any]:
    """Parse a model JSON object, tolerating a little accidental wrapper text."""
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", raw or "", re.S)
        if not match:
            return {}
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return payload if isinstance(payload, dict) else {}


_QUESTION_TEMPLATES = (
    "Explain {topic} in your own words, including the main idea and one important detail.",
    "In your own words, what is {topic} and why does it matter?",
    "Describe {topic} as you would explain it to a classmate.",
    "What are the key points a student should know about {topic}?",
    "Give a short explanation of {topic} and one example from the material.",
)


def _fallback_question(topic: str, rng: Optional[random.Random] = None) -> str:
    if topic.rstrip().endswith("?"):  # the heading is already a question ("What is Multimedia?")
        return f"{topic.strip()} Answer in your own words, with the main idea and one important detail."
    return (rng or random).choice(_QUESTION_TEMPLATES).format(topic=topic)


def _selected_chunks(ctx: AgentContext, limit: int = 4, exclude_topics=(), rng: Optional[random.Random] = None) -> list:
    """A random stretch of the selected material, avoiding topics already asked about."""
    rng = rng or random
    selected = set(ctx.source_filenames or [])
    chunks = []
    for doc in ctx.vectorstore.docstore._dict.values():
        source = Path(doc.metadata.get("source", "")).name
        if source == FACULTY_SOURCE_NAME:
            continue
        if selected and source not in selected:
            continue
        text = " ".join(doc.page_content.split())
        if len(text) >= 80:
            chunks.append(doc)
    chunks.sort(key=lambda d: (Path(d.metadata.get("source", "")).name, d.metadata.get("page") or 0))
    if not chunks:
        return []
    asked = {t.lower() for t in exclude_topics}
    starts = [
        i for i, d in enumerate(chunks)
        if len(" ".join(d.page_content.split())) >= 200
        and is_title_like(d.metadata.get("section") or "")
        and topic_title(d.metadata).lower() not in asked
        and not topic_title(d.metadata).endswith("...")  # a cut-off heading is not a clean topic
    ]
    if not starts:  # everything has been asked (or nothing looks like a topic): start over
        starts = [i for i, d in enumerate(chunks) if is_title_like(d.metadata.get("section") or "")] or list(range(len(chunks)))
    start = rng.choice(starts)
    return chunks[start:start + limit]


MAX_MODEL_POINTS = 6        # a model answer for a 30 to 60 second spoken reply


def _bullet_headings(ctx: AgentContext, min_points: int = 3) -> list:
    """Headings with enough bullet points under them to ask about (same shape as `major_headings`)."""
    out = []
    for group in group_material(ctx):
        if not group["title"]:
            continue
        points = bullet_points(group["text"], group["section"])
        if len(points) >= min_points:
            out.append({
                "heading": human_title(group["title"]), "sentences": points, "bullets": True,
                "file": group["file"], "pages": group["pages"], "text": group["text"],
            })
    return out


NO_HEADING_MESSAGE = (
    "There is nothing in the selected material to ask an oral question about yet: no heading has enough readable "
    "text under it. Choose other material, or use the Ask or Quiz tab for this file."
)


def _best_answer(context: str, topic: str, question: str, ctx: AgentContext) -> str:
    """Model answer: the full list when the topic is "types of X", else the best points of the context."""
    types = type_definitions(f"types of {topic}" if "types of" not in topic.lower() else topic,
                             ctx.vectorstore, ctx.source_filenames)
    if types:
        return "\n".join(format_type_definitions(types))
    return _best_answer_from_context(context, question)


def _best_answer_from_context(context: str, question: str = "") -> str:
    """A model answer as short bullet points ("- ...") built from the retrieved context."""
    text = re.sub(r"\[Chunk \d+[^\]]*\]", " ", context or "")
    sentences = readable_sentences(text, max_len=240)
    if not sentences:
        flat = " ".join(text.split())
        if len(flat) < 45:
            return "The selected material does not contain enough readable text to form a model answer."
        return "- " + (flat[:260].rsplit(" ", 1)[0] if len(flat) > 260 else flat).rstrip(" ,;:—-") + "."
    terms = set(_content_terms(question))
    if terms:  # points about the question's own topic first, the rest in document order
        sentences.sort(key=lambda sentence: -len(terms & set(_content_terms(sentence))))
    return "\n".join(f"- {sentence}" for sentence in sentences[:5])


_SCORING_STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "about", "into", "onto", "your", "you",
    "are", "was", "were", "has", "have", "had", "can", "could", "would", "should", "will",
    "been", "being", "they", "their", "there", "which", "what", "when", "where", "why", "how",
    "used", "uses", "use", "using", "also", "then", "than", "such", "like", "some", "very",
    "chunk", "source", "page", "section", "student", "answer", "question", "explain", "main",
    "idea", "important", "detail", "own", "words",
}


def _content_terms(text: str) -> list[str]:
    terms = []
    for token in re.findall(r"[a-z0-9][a-z0-9\-]{2,}", text.lower()):
        if token not in _SCORING_STOPWORDS and not token.isdigit():
            terms.append(token)
    return terms


def _fallback_score(answer: str, context: str, question: str = "") -> tuple[float, str]:
    """
    Lexical fallback when the local LLM is unavailable or returns invalid JSON.

    This is not the preferred scorer; it lets the feature degrade gracefully for
    demos and tests.
    """
    answer_terms = set(_content_terms(answer))
    if not answer_terms:
        return 0.0, "No usable spoken answer was provided."

    context_tokens = _content_terms(context)
    context_terms = set(context_tokens)
    if not context_terms:
        return 0.2, "The system could not find enough retrieved material to score the answer confidently."

    frequencies: Dict[str, int] = {}
    for token in context_tokens:
        frequencies[token] = frequencies.get(token, 0) + 1
    question_terms = set(_content_terms(question))
    # Frequent source terms plus terms named in the question approximate the
    # "expected points" better than raw overlap against every word the student said.
    ranked = sorted(context_terms, key=lambda term: (term not in question_terms, -frequencies[term], term))
    key_terms = set(ranked[: min(24, max(8, len(ranked)))])

    key_coverage = len(answer_terms & key_terms) / max(1, len(key_terms))
    source_precision = len(answer_terms & context_terms) / max(1, len(answer_terms))
    topic_match = 1.0 if question_terms and answer_terms & question_terms else 0.0
    length_bonus = 0.08 if len(answer_terms) >= 20 else 0.0

    score = (0.65 * key_coverage) + (0.25 * source_precision) + (0.10 * topic_match) + length_bonus
    score = max(0.0, min(0.88, score))
    return score, ""


_FILLER_BEFORE_HEAD = frozenset(
    "a an the of and or are is as to in on types type six main these those other each one two three "
    "different all some any this that its their same".split()
)


def _score_named_items(answer: str, expected: list) -> tuple[float, str]:
    """Score an answer that should name a fixed list (e.g. the six types of medium)."""
    spoken = " ".join(re.findall(r"[a-z]+", answer.lower()))
    head = expected[0].split()[-1].lower()  # "Medium"
    keys = {name: " ".join(name.lower().split()[:-1]) for name in expected}  # "information exchange"
    named = [name for name, key in keys.items() if re.search(rf"\b{re.escape(key)}\b", spoken)]
    missing = [name for name in expected if name not in named]
    known_words = {w for key in keys.values() for w in key.split()}
    wrong = sorted({
        f"{m.group(1)} {head}" for m in re.finditer(rf"\b([a-z]+) {re.escape(head)}\b", spoken)
        if m.group(1) not in _FILLER_BEFORE_HEAD and m.group(1) not in known_words
    })
    score = max(0.0, len(named) / len(expected) - 0.1 * len(wrong))
    parts = [f"You named {len(named)} of {len(expected)} correctly."]
    if missing:
        parts.append("Missing: " + ", ".join(missing) + ".")
    if wrong:
        parts.append("Not in the material: " + ", ".join(wrong) + ".")
    return min(score, 0.9), " ".join(parts)


def _missed_points_feedback(answer: str, best_answer: str, limit: int = 4) -> str:
    """Which key points of the model answer the spoken answer did not touch (empty if there are none to compare)."""
    answer_terms = set(_content_terms(answer))
    points = [re.sub(r"\*\*", "", line.lstrip("- ").strip()) for line in (best_answer or "").splitlines() if line.startswith("-")]
    scored = []
    for point in points:
        terms = set(_content_terms(point))
        if len(terms) >= 3:
            scored.append((point, len(terms & answer_terms) / len(terms)))
    if not scored:
        return ""
    missed = [point for point, share in scored if share < 0.35]
    covered = len(scored) - len(missed)
    if not missed:
        return f"You covered all {len(scored)} key points."
    lines = []
    for point in missed[:limit]:
        lines.append(f"- {point}")
    more = f"\n- ...and {len(missed) - limit} more." if len(missed) > limit else ""
    return f"You covered {covered} of {len(scored)} key points. Points you missed:\n" + "\n".join(lines) + more


class OralAssessor(Agent):
    name = "Oral Assessor"
    job = "Ask a spoken-answer question, score content, and combine it with pause/fluency metrics."

    def generate_question(
        self,
        topic: str,
        ctx: AgentContext,
        question_llm=None,
    ) -> AgentResult:
        retrieval = retrieve(topic, ctx.vectorstore, source_filenames=ctx.source_filenames)
        if not retrieval.relevant:
            return AgentResult(
                agent=self.name,
                kind="oral_question",
                grounded=False,
                text=NOT_COVERED_MESSAGE,
                data={"topic": topic, "error": "not_covered"},
            )

        chunks = retrieval.chunks
        resolved_topic = topic_title(chunks[0].metadata) if topic.strip() == "" else topic.strip()
        context = format_context(chunks)
        chain = _QUESTION_PROMPT | (question_llm or ctx.llm or get_json_llm()) | StrOutputParser()
        question = ""
        try:
            payload = _json_object(chain.invoke({"topic": resolved_topic, "context": context}))
            question = payload.get("question", "").strip() if isinstance(payload.get("question"), str) else ""
        except Exception:
            question = ""
        if not question:
            question = _fallback_question(resolved_topic)

        return AgentResult(
            agent=self.name,
            kind="oral_question",
            text=question,
            sources=chunks,
            data={
                "topic": resolved_topic,
                "question": question,
                "context": context,
                "best_answer": _best_answer(context, resolved_topic, question, ctx),
            },
        )

    def generate_question_from_material(
        self, ctx: AgentContext, question_llm=None, exclude_topics=(), rng: Optional[random.Random] = None
    ) -> AgentResult:
        """Ask about one of the material's major headings (the same ones the study notes use)."""
        rng = rng or random
        headings = major_headings(ctx) or _bullet_headings(ctx)   # a deck written as bullets has no full sentences
        if not headings:
            return AgentResult(
                agent=self.name,
                kind="oral_question",
                grounded=False,
                text=NO_HEADING_MESSAGE,
                data={"topic": "Selected material", "error": "no_headings"},
            )
        asked = {t.lower() for t in exclude_topics}
        fresh = [h for h in headings if h["heading"].lower() not in asked] or headings  # all asked: start over
        chosen = rng.choice(fresh)
        topic = chosen["heading"]
        context = f"[Chunk 1 | source: {chosen['file']} | page: {chosen['pages'][0]} | section: {topic}]\n{chosen['text'][:3500]}"
        chain = _QUESTION_PROMPT | (question_llm or ctx.llm or get_json_llm()) | StrOutputParser()
        question = ""
        try:
            payload = _json_object(chain.invoke({"topic": topic, "context": context}))
            question = payload.get("question", "").strip() if isinstance(payload.get("question"), str) else ""
        except Exception:
            question = ""
        if not question:
            question = _fallback_question(topic, rng)
        types = type_definitions(topic, ctx.vectorstore, ctx.source_filenames)
        points = chosen["sentences"]
        if chosen.get("bullets") and len(points) > MAX_MODEL_POINTS:
            # A bullet slide can hold dozens of points under one heading: keep those closest to the question.
            terms = set(_content_terms(f"{question} {topic}"))
            keep = sorted(range(len(points)), key=lambda i: (-len(terms & set(_content_terms(points[i]))), i))
            points = [points[i] for i in sorted(keep[:MAX_MODEL_POINTS])]
        best = "\n".join(format_type_definitions(types)) if types else "\n".join(f"- {s}" for s in points)
        return AgentResult(
            agent=self.name,
            kind="oral_question",
            text=question,
            data={"topic": topic, "question": question, "context": context, "best_answer": best},
        )

    def assess_response(
        self,
        topic: str,
        question: str,
        transcript: str,
        audio_data: bytes,
        ctx: AgentContext,
        context: Optional[str] = None,
        score_llm=None,
        save: bool = True,
    ) -> AgentResult:
        metrics = analyze_audio_response(audio_data, transcript)

        if context is None:
            retrieval = retrieve(topic or question, ctx.vectorstore, source_filenames=ctx.source_filenames)
            context = format_context(retrieval.chunks) if retrieval.relevant else ""

        types = type_definitions(topic if "types of" in topic.lower() else f"types of {topic}",
                                 ctx.vectorstore, ctx.source_filenames)
        best_answer = _best_answer(context, topic, question, ctx)
        content_score, feedback = self._score_content(
            question, transcript, context, score_llm or ctx.llm, expected=[t for t, _ in types],
            best_answer=best_answer,
        )
        level = infer_understanding_level(content_score, metrics)
        assessment_id = None
        if save:
            assessment_id = record_oral_assessment(
                topic=topic,
                prompt=question,
                transcript=transcript,
                content_score=content_score,
                metrics=metrics,
                feedback=feedback,
                student_id=ctx.student_id,
                understanding_level=level,
                db_dir=ctx.db_dir,
            )

        text = (
            f"Understanding level: **{level}**\n\n"
            f"Content score: {content_score:.0%}"
            + (f"\n\n{feedback}" if feedback else "")
        )
        return AgentResult(
            agent=self.name,
            kind="oral_assessment",
            text=text,
            data={
                "assessment_id": assessment_id,
                "topic": topic,
                "question": question,
                "transcript": transcript,
                "content_score": content_score,
                "understanding_level": level,
                "feedback": feedback,
                "best_answer": best_answer,
                "metrics": metrics.to_dict(),
            },
        )

    def _score_content(
        self, question: str, transcript: str, context: str, llm=None, expected=None, best_answer: str = ""
    ) -> tuple[float, str]:
        if not transcript.strip():
            return 0.0, "No spoken answer was transcribed."
        if not context.strip():
            score, _ = _fallback_score(transcript, "", question)
            return score, _missed_points_feedback(transcript, best_answer)
        if expected:
            context = context + "\n\nThe complete list in the material: " + "; ".join(expected) + "."

        chain = _SCORE_PROMPT | (llm or get_json_llm()) | StrOutputParser()
        try:
            payload = _json_object(chain.invoke({"context": context, "question": question, "answer": transcript}))
            score = float(payload.get("content_score", 0.0))
            feedback = payload.get("feedback", "")
            if not isinstance(feedback, str) or not feedback.strip():
                feedback = _missed_points_feedback(transcript, best_answer)
            return max(0.0, min(1.0, score)), feedback.strip()
        except Exception:
            if expected:
                return _score_named_items(transcript, expected)
            score, _ = _fallback_score(transcript, context, question)
            return score, _missed_points_feedback(transcript, best_answer)

    def run(self, request: str, ctx: AgentContext, **kwargs: Any) -> AgentResult:
        return self.generate_question(request, ctx, **kwargs)
