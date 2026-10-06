"""
Note Generator: turns uploaded material into short, simple study notes.

For each topic it retrieves the most relevant chunks of the notes and asks the
model for a fixed structure: a heading, a definition, key points (bullets) and a
use case. Because a small local model can drift from the source, nothing it
writes is trusted blindly:

  - the output must be valid JSON in the expected shape (one retry otherwise);
  - every key point is checked against the retrieved text, and a point that
    shares too few words with it is dropped and counted, not shown;
  - a use case with no support in the text is replaced by "The notes do not
    give an example";
  - each section records the slides it was written from.

A topic the notes do not cover is skipped (and logged as a gap for the
professor) instead of being written from the model's own knowledge.
"""

import json
import re
from typing import Any, Callable, Dict, List, Optional, Set

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agents.base import Agent, AgentContext, AgentResult
from rag_core.chain import NOT_COVERED_MESSAGE, STOPWORDS as _STOPWORDS, format_context, retrieve
from rag_core.insights import topic_title
from rag_core.llm import get_json_llm
from rag_core.query_log import log_query

NO_EXAMPLE = "The notes do not give an example."
NOTES_K = 5                 # chunks of context per topic
KEY_POINT_MIN_SUPPORT = 0.5  # share of a bullet's content words found in the context
USE_CASE_MIN_SUPPORT = 0.4
MAX_KEY_POINTS = 6
# A statement that survives the support filter but still contains this many words
# absent from the source is marked "check this". Word overlap cannot tell a fair
# paraphrase from an invented detail, so these are flagged for the reader, not removed.
CHECK_NOVEL_WORDS = 2

_SYSTEM = """You turn course material into short revision notes for a student. Use ONLY the \
context below. Write in simple language: short sentences, explain any jargon, no filler.

Topic: {topic}

Return ONLY JSON in exactly this shape:
{{"heading": "short title", "definition": "1 or 2 sentences", "key_points": ["short point", "short point"], "use_case": "one short real-world use or example", "chunks": [1]}}

Rules:
- Every statement must be supported by the context. Do not add outside facts.
- Do not add examples, industries, products or uses that the context does not name (for example, do \
not say "such as banking" or "for business intelligence" unless the context says so).
- "key_points": 3 to 6 bullets, one short sentence each. Write fewer if the context is thin.
- "use_case": a use or example that appears in the context. If there is none, use exactly: "{no_example}"
- "chunks": the numbers N of the [Chunk N] you used.

--- CONTEXT ---
{context}
"""

_PROMPT = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", "Write the notes for: {topic}")])

_EXTRA_STOP = frozenset("also have been into than then them they their there these those will would should "
                        "which while with from that this such some more most very each both only other".split())


def _stems(text: str) -> Set[str]:
    """Content-word stems (first five letters), so 'replicas' matches 'replica'."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w[:5] for w in words if len(w) >= 4 and w not in _STOPWORDS and w not in _EXTRA_STOP}


def support(text: str, context_stems: Set[str]) -> float:
    """Share of `text`'s content words that also appear in the context (1.0 if it has none)."""
    stems = _stems(text)
    if not stems:
        return 1.0
    return len(stems & context_stems) / len(stems)


_SMALL_WORDS = frozenset({"vs", "and", "of", "the", "in", "for", "to", "on", "a", "an", "or"})


def prettify_heading(heading: str) -> str:
    """'OLTP VS OLAP SYSTEMS' -> 'OLTP vs OLAP Systems'. Only touches ALL-CAPS headings;
    words of up to four letters are kept as acronyms (a long acronym such as NOSQL becomes 'Nosql')."""
    if not heading.isupper():
        return heading
    words = []
    for i, word in enumerate(heading.split()):
        if word.lower() in _SMALL_WORDS and i > 0:
            words.append(word.lower())
        elif len(word) <= 4 or not word.isalpha():
            words.append(word)
        else:
            words.append(word.capitalize())
    return " ".join(words)


def novel_words(text: str, context_stems: Set[str]) -> int:
    """How many of `text`'s content words do not appear in the context at all."""
    return len(_stems(text) - context_stems)


def _clean(value: Any, limit: int) -> str:
    text = value.strip() if isinstance(value, str) else ""
    return text if len(text) <= limit else text[: limit - 3].rsplit(" ", 1)[0] + "..."


def parse_notes_json(raw: str, n_chunks: int) -> Optional[Dict[str, Any]]:
    """The note content from the model's JSON, or None if it is unusable."""
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None

    definition = _clean(payload.get("definition"), 400)
    points = payload.get("key_points")
    if not definition or not isinstance(points, list):
        return None
    key_points = [_clean(p, 240) for p in points]
    key_points = [p for p in key_points if p][:MAX_KEY_POINTS]
    if not key_points:
        return None

    chunks = payload.get("chunks")
    cited = sorted({c for c in chunks if isinstance(c, int) and not isinstance(c, bool) and 1 <= c <= n_chunks}) \
        if isinstance(chunks, list) else []
    return {
        "heading": _clean(payload.get("heading"), 80),
        "definition": definition,
        "key_points": key_points,
        "use_case": _clean(payload.get("use_case"), 300) or NO_EXAMPLE,
        "chunks": cited,
    }


class NoteGenerator(Agent):
    name = "Note Generator"
    job = "Write heading-wise study notes (definition, key points, use case) from the material."

    def run(
        self,
        request: str,
        ctx: AgentContext,
        topics: Optional[List[str]] = None,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
        **_,
    ) -> AgentResult:
        topics = [t for t in (topics or [request]) if t and t.strip()]
        chain = _PROMPT | (ctx.llm if ctx.llm is not None else get_json_llm()) | StrOutputParser()

        sections: List[Dict[str, Any]] = []
        skipped: List[str] = []
        failed: List[str] = []
        dropped = 0
        log_ids: List[int] = []

        for index, topic in enumerate(topics):
            if on_progress:
                on_progress(index, len(topics), topic)
            retrieval = retrieve(topic, ctx.vectorstore, k=NOTES_K, source_filenames=ctx.source_filenames)
            if not retrieval.relevant:
                skipped.append(topic)
                log_ids.append(log_query(
                    question=f"Notes on: {topic}", grounded=False, top_score=retrieval.top_score,
                    source_documents=[], scope=ctx.source_filenames, db_dir=ctx.db_dir, student_id=ctx.student_id,
                ))
                continue

            chunks = retrieval.chunks
            variables = {"topic": topic, "context": format_context(chunks), "no_example": NO_EXAMPLE}
            parsed = None
            for _attempt in range(2):  # one retry: small models fail JSON now and then
                parsed = parse_notes_json(chain.invoke(variables), n_chunks=len(chunks))
                if parsed:
                    break
            if not parsed:
                failed.append(topic)
                continue

            context_stems = _stems(" ".join(c.page_content for c in chunks))
            kept = [p for p in parsed["key_points"] if support(p, context_stems) >= KEY_POINT_MIN_SUPPORT]
            dropped += len(parsed["key_points"]) - len(kept)
            if not kept:
                failed.append(topic)
                continue
            use_case = parsed["use_case"]
            if use_case != NO_EXAMPLE and support(use_case, context_stems) < USE_CASE_MIN_SUPPORT:
                use_case = NO_EXAMPLE
                dropped += 1

            # Sources: the chunks the model cited, plus the chunk that best matches each
            # statement. The model's own chunk numbers are often wrong, so they are not trusted alone.
            chunk_stems = [_stems(c.page_content) for c in chunks]
            used_idx = {n - 1 for n in parsed["chunks"]}
            for text in [parsed["definition"], *kept] + ([use_case] if use_case != NO_EXAMPLE else []):
                overlaps = [len(_stems(text) & stems) for stems in chunk_stems]
                if max(overlaps) > 0:
                    used_idx.add(overlaps.index(max(overlaps)))
            used = [chunks[i] for i in sorted(used_idx)] or chunks[:3]
            sources, seen_pages = [], set()
            for c in sorted(used, key=lambda c: (str(c.metadata.get("source")), c.metadata.get("page") or 0)):
                entry = {
                    "file": str(c.metadata.get("source", "unknown")).replace("\\", "/").split("/")[-1],
                    "page": (c.metadata.get("page") or 0) + 1,
                    "slide": topic_title(c.metadata),
                }
                if (entry["file"], entry["page"]) not in seen_pages:
                    seen_pages.add((entry["file"], entry["page"]))
                    sources.append(entry)
            sections.append({
                "topic": topic,
                "heading": prettify_heading(parsed["heading"] or topic),
                "definition": parsed["definition"],
                "definition_support": round(support(parsed["definition"], context_stems), 2),
                "key_points": kept,
                "use_case": use_case,
                # Positions of statements to double-check against the source slides.
                "check": {
                    "key_points": [i for i, p in enumerate(kept) if novel_words(p, context_stems) >= CHECK_NOVEL_WORDS],
                    "use_case": use_case != NO_EXAMPLE and novel_words(use_case, context_stems) >= CHECK_NOVEL_WORDS,
                },
                "sources": sources,
            })

        if on_progress:
            on_progress(len(topics), len(topics), "")
        title = "Study Notes: " + ", ".join(s["heading"] for s in sections[:3]) if sections else "Study Notes"
        if not sections:
            text = NOT_COVERED_MESSAGE if skipped and not failed else "No notes could be generated. Try again."
        else:
            text = f"Notes ready for {len(sections)} topic(s)."
        return AgentResult(
            agent=self.name,
            kind="notes",
            text=text,
            grounded=bool(sections) or bool(failed),  # False only when the notes simply do not cover it
            data={
                "sections": sections,
                "skipped": skipped,
                "failed": failed,
                "dropped": dropped,
                "title": title,
                "files": sorted({s["file"] for sec in sections for s in sec["sources"]}),
                "log_id": log_ids[0] if log_ids else None,
            },
        )
