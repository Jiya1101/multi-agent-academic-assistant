"""
Concept Explainer: teaches one concept step by step at a chosen level.

Different job from the Doubt Resolver: a doubt wants a direct answer, an
explanation wants a structured walk-through. Same retrieval and relevance
gate, so it still refuses to explain anything the notes do not contain.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agents.base import Agent, AgentContext, AgentResult
from rag_core.chain import NOT_COVERED_MESSAGE, format_context, retrieve
from rag_core.llm import get_llm
from rag_core.normalize import chunk_sentences
from rag_core.query_log import log_query

_LEVELS = {
    "beginner": (
        "Use plain language and short sentences, and define jargon the first time it appears. "
        "You may add ONE short everyday analogy, labelled exactly 'Analogy (not from the notes):'. "
        "Never present an analogy as if it came from the notes."
    ),
    "detailed": (
        "Be precise and complete. Include trade-offs, comparisons and conditions that appear in the "
        "context. Do not use analogies."
    ),
}

_SYSTEM = """You are a patient teaching assistant. Explain the concept the student asked about \
using ONLY the course-note context below.

Level: {level_rules}

Rules:
- Every factual statement must come from the context. Cite it as [Chunk N].
- If the context does not cover part of the concept, say what is missing instead of filling the gap.
- Keep it concise. Structure: one-sentence definition, then at most 5 short numbered steps or \
bullets (one idea each), then a line starting "Check yourself:" with one question the student can \
answer from the notes.

--- CONTEXT ---
{context}
"""

_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _SYSTEM), ("human", "Concept to explain: {request}")]
)


class ConceptExplainer(Agent):
    name = "Concept Explainer"
    job = "Explain a concept step by step, at beginner or detailed level, from the notes."

    def run(
        self,
        request: str,
        ctx: AgentContext,
        level: str = "beginner",
        log: bool = True,
        **_,
    ) -> AgentResult:
        retrieval = retrieve(request, ctx.vectorstore, source_filenames=ctx.source_filenames)

        if retrieval.relevant:
            chain = _PROMPT | (ctx.llm if ctx.llm is not None else get_llm()) | StrOutputParser()
            try:
                text = chain.invoke(
                    {
                        "level_rules": _LEVELS.get(level, _LEVELS["beginner"]),
                        "context": format_context(retrieval.chunks),
                        "request": request,
                    }
                )
            except Exception:
                excerpts = []
                seen = set()
                for i, doc in enumerate(retrieval.chunks[:4], start=1):
                    for sentence in chunk_sentences(doc, max_len=260)[:2]:
                        if sentence.lower()[:80] not in seen:
                            seen.add(sentence.lower()[:80])
                            excerpts.append(f"- {sentence} [Chunk {i}]")
                text = (
                    "I found relevant material, but the local LLM is not available right now. "
                    "Use these course excerpts as a grounded explanation:\n\n"
                    + "\n".join(excerpts[:5])
                )
        else:
            text = NOT_COVERED_MESSAGE

        log_id = None
        if log:
            # Logged like any question, so an explanation request the notes
            # cannot support still reaches the professor as a gap.
            log_id = log_query(
                question=request,
                grounded=retrieval.relevant,
                top_score=retrieval.top_score,
                source_documents=retrieval.chunks,
                scope=ctx.source_filenames,
                db_dir=ctx.db_dir,
                student_id=ctx.student_id,
                agent=self.name,
                session_id=ctx.session_id,
            )
        return AgentResult(
            agent=self.name,
            kind="explanation",
            text=text,
            sources=retrieval.chunks,
            grounded=retrieval.relevant,
            data={"level": level, "log_id": log_id, "question": request},
        )
