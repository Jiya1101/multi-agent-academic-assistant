"""
Content Advisor: turns "the class is confused about X" into specific things a professor can do about it.

It works in two steps, and only the second uses the language model:

  1. diagnose (no model): for the highest-scoring concepts, rule-based findings that cite their evidence
     (a faulty-looking quiz question, a shared wrong answer, a missing prerequisite, wording that does not
     match the slides, ...). See rag_core/advice.py.
  2. draft (model, only when the professor asks): a short clarification note for students, written ONLY from
     the course material and screened sentence by sentence against it, the same way study notes are.

Nothing here reaches students by itself. A draft is saved for review; a professor reads it, edits it, and
approves it before it is added to the course material. Remedial quizzes are written by the Quiz Generator
(the Orchestrator hands the request over) and stay hidden until approved.
"""

import re
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agents.base import Agent, AgentContext, AgentResult
from agents.note_generator import CHECK_NOVEL_WORDS, KEY_POINT_MIN_SUPPORT, _stems, novel_words, support
from rag_core.advice import Advice, diagnose
from rag_core.chain import NOT_COVERED_MESSAGE, format_context, retrieve
from rag_core.config import FACULTY_SOURCE_NAME
from rag_core.confusion import compute_confusion
from rag_core.embeddings import get_embeddings
from rag_core.learning_log import list_attempts, list_quizzes
from rag_core.llm import get_llm
from rag_core.oral_assessment import list_oral_assessments
from rag_core.query_log import list_queries

CLARIFY_K = 6               # chunks of course material given to the model
MIN_SENTENCES = 2           # fewer supported sentences than this is not worth saving

_SYSTEM = """You help a professor write a short clarification for students about a topic many of them find confusing.

Use ONLY the course material below. Do not add examples, numbers or facts that are not in it.
Write 3 to 6 short sentences in simple language, as plain text: no headings, no bullet points, no citations.
Begin directly with the explanation: no introduction such as "Here is" or "Sure".
State the correct idea first. Never present a wrong idea as true; mention one only to say it is wrong, and say
how it differs from the correct idea.

--- WHAT STUDENTS ARE CONFUSED ABOUT ---
{confusion}

--- COURSE MATERIAL ---
{context}
"""
_PROMPT = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", "{task}")])


def _task(advice: Advice) -> str:
    """The instruction itself carries the focus: a small model follows it far better than background notes."""
    if advice.misconceptions:
        m = advice.misconceptions[0]
        return (
            f"Write the clarification. Explain the correct answer to \"{m['question']}\", which is \"{m['right']}\", "
            f"and explain why \"{m['wrong']}\" is not correct. Topic: {advice.concept}."
        )
    if advice.sub_questions:
        return (
            f"Write the clarification. Answer this question students keep asking, in simple words: "
            f"\"{advice.sub_questions[0][0]}\" Topic: {advice.concept}."
        )
    return f"Write the clarification about: {advice.concept}"


def _confusion_brief(advice: Advice) -> str:
    lines = []
    for m in advice.misconceptions:
        lines.append(
            f"- Asked \"{m['question']}\", {m['share']:.0%} of the class wrongly answered \"{m['wrong']}\". "
            f"The correct answer is \"{m['right']}\"."
        )
    for question, size in advice.sub_questions:
        lines.append(f"- Students ask (about {size} time(s)): \"{question}\"")
    return "\n".join(lines) or f"- Many students asked about {advice.concept}."


def _gather_context(advice: Advice, ctx: AgentContext) -> tuple:
    """
    Course material for the draft: what the concept's slides say AND what the confusing quiz question is about.
    (The question can concern a neighbouring idea: searching only the concept title once missed the slides the
    mistake was actually about.) Returns (chunks, relevant_any).
    """
    queries = [m["question"] for m in advice.misconceptions] + [advice.concept]
    if advice.sub_questions:
        queries.append(advice.sub_questions[0][0])
    chunks, seen, relevant = [], set(), False
    for query in queries:
        retrieval = retrieve(query, ctx.vectorstore, k=4, source_filenames=ctx.source_filenames)
        if not retrieval.relevant:
            continue
        relevant = True
        for doc in retrieval.chunks:
            ident = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:80])
            if doc.metadata.get("source") != FACULTY_SOURCE_NAME and ident not in seen:
                seen.add(ident)
                chunks.append(doc)
    return chunks[:CLARIFY_K], relevant


def _strip_preamble(text: str) -> str:
    """Remove model chatter before the explanation ("Sure, here is a clarification about X:")."""
    text = re.sub(r"^\s*(?:sure|certainly|of course|okay)\b[^.!?:\n]{0,40}[,!.:]\s*", "", text or "", flags=re.I)
    return re.sub(r"^\s*here(?:'s| is| are)\b[^:\n]{0,150}:\s*", "", text, flags=re.I)


def _split_sentences(text: str) -> List[str]:
    text = _strip_preamble(text)
    text = re.sub(r"\[Chunk\s*\d+\]", "", text or "")                  # citations are not wanted in a note
    text = re.sub(r"^\s*[-*•]\s*", "", text, flags=re.M)            # stray bullets
    parts = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    return [p.strip() for p in parts if len(p.strip()) >= 25]


class ContentAdvisor(Agent):
    name = "Content Advisor"
    job = "Suggest what to change for concepts the class is confused about, and draft a clarification."

    def run(
        self,
        request: str,
        ctx: AgentContext,
        top_n: int = 3,
        report=None,
        weights: Optional[Dict[str, float]] = None,
        **_,
    ) -> AgentResult:
        """Diagnose the top concepts. `report` (from Faculty Insight) is used if given, else it is computed."""
        rows = list_queries(db_dir=ctx.db_dir)
        quizzes = list_quizzes(db_dir=ctx.db_dir)
        attempts = list_attempts(db_dir=ctx.db_dir)
        if report is None:
            report = compute_confusion(rows, quizzes, attempts, list_oral_assessments(db_dir=ctx.db_dir), weights)
        if not report.concepts:
            return AgentResult(
                agent=self.name, kind="advice", grounded=False,
                text="Not enough evidence yet to suggest anything.", data={"advice": [], "report": report},
            )
        advice = diagnose(report, rows, quizzes, attempts, ctx.embeddings or get_embeddings(), top_n)
        return AgentResult(
            agent=self.name, kind="advice", text=f"Suggestions for {len(advice)} concept(s).",
            data={"advice": advice, "report": report},
        )

    def review_text(self, text: str, advice: Advice, ctx: AgentContext) -> List[str]:
        """Sentences of `text` (e.g. after the professor edits it) that the course material does not clearly support."""
        chunks, _relevant = _gather_context(advice, ctx)
        stems = _stems(" ".join(d.page_content for d in chunks))
        return [
            s for s in _split_sentences(text)
            if support(s, stems) < KEY_POINT_MIN_SUPPORT or novel_words(s, stems) >= CHECK_NOVEL_WORDS
        ]

    def draft_clarification(self, advice: Advice, ctx: AgentContext) -> AgentResult:
        """
        A short note for students on `advice.concept`, grounded in the course material.

        Sentences with too little support in the material are dropped; sentences that still use words absent
        from the material are returned in `flagged` so the professor can check them. The note is a DRAFT.
        """
        chunks, relevant = _gather_context(advice, ctx)
        if not relevant or not chunks:
            return AgentResult(agent=self.name, kind="clarification", grounded=False, text=NOT_COVERED_MESSAGE,
                               data={"error": "not_covered"})

        context = format_context(chunks)
        chain = _PROMPT | (ctx.llm if ctx.llm is not None else get_llm()) | StrOutputParser()
        try:
            raw = chain.invoke({"task": _task(advice), "confusion": _confusion_brief(advice), "context": context})
        except Exception:
            return AgentResult(agent=self.name, kind="clarification", grounded=False,
                               text="The local language model is not available right now.",
                               data={"error": "llm_unavailable"})

        stems = _stems(" ".join(d.page_content for d in chunks))
        kept, dropped, flagged = [], 0, []
        for sentence in _split_sentences(raw):
            if support(sentence, stems) < KEY_POINT_MIN_SUPPORT:
                dropped += 1
                continue
            kept.append(sentence)
            if novel_words(sentence, stems) >= CHECK_NOVEL_WORDS:
                flagged.append(sentence)
        if len(kept) < MIN_SENTENCES:
            return AgentResult(agent=self.name, kind="clarification", grounded=False,
                               text="The model's draft was not supported by the course material. Try again.",
                               data={"error": "unsupported", "dropped": dropped})
        return AgentResult(
            agent=self.name, kind="clarification", sources=chunks,
            text=" ".join(kept),
            data={"flagged": flagged, "dropped": dropped, "concept": advice.concept, "key": advice.key},
        )
