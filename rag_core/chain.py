"""
RAG orchestration: retrieve -> build grounded prompt -> generate answer.

Exposes `answer_question`, the single function every agent (this Streamlit
app today; Concept Explainer / Quiz Generator / Doubt Resolver later) should
call to go from a raw user query to a grounded answer + the source chunks
used to produce it.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rag_core.config import TOP_K, RELEVANCE_SCORE_THRESHOLD
from rag_core.llm import get_llm
from rag_core.normalize import chunk_sentences

_NOT_COVERED_MESSAGE = (
    "The provided notes do not contain information relevant to this question. "
    "Try rephrasing, or ask something covered by the uploaded course material."
)

NOT_COVERED_MESSAGE = _NOT_COVERED_MESSAGE

# Strict-grounding, citation-enforcing prompt. The explicit negative
# constraint ("state that the notes do not cover X") and the requirement to
# tag claims with [Chunk N] are what most reduce hallucination: they force
# the model to align its answer with retrieved tokens instead of falling
# back on parametric training knowledge.
_SYSTEM_PROMPT = """You are an expert academic teaching assistant. Your goal is to explain \
concepts clearly and accurately to students based strictly on the provided course material.

--- GUIDELINES ---
1. Strict Context Attribution:
   - Answer the question using ONLY the context chunks below.
   - Do not infer, assume, or extrapolate details not directly supported by the text.
   - If the context does not contain enough information to answer completely, state clearly \
what is covered and what is missing, e.g. "The provided notes do not contain details \
regarding X."

2. Concept Separation & Precision:
   - Maintain strict boundaries between distinct technologies, parent categories, and \
sub-topics.
   - Never attribute a property of a broader topic to a specific subject unless the text \
explicitly links them.

3. Pedagogical Structure (keep it concise):
   - Begin with a direct answer or definition in 1-2 sentences.
   - Then give at most 5 short bullet points ("- ...") for the key components, features, or \
use cases. One idea per bullet; no long paragraphs.
   - Mention trade-offs or comparisons only if the material states them.

4. Source Transparency:
   - Cite the relevant chunk for each key point, e.g. [Chunk 2].
   - If terms in the context appear corrupted or concatenated due to PDF extraction \
artifacts, resolve them to standard English terms naturally without altering their meaning.

--- CONTEXT ---
{context}
"""

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("human", "--- STUDENT QUESTION ---\n{question}\n\n--- ACADEMIC EXPLANATION ---"),
    ]
)


_TYPES_QUESTION = re.compile(
    r"\b(?:types|kinds|categories|classes)\s+of\s+(?P<x>[A-Za-z][A-Za-z\- ]{1,30}?)\s*(?:\?|$|,|\band\b|\bin\b)", re.I
)
_DEFINITION_SENTENCE = re.compile(
    r"^(?:An?\s+|The\s+)?(?P<term>[A-Z][A-Za-z0-9&/\-]*(?: [A-Za-z0-9&/\-]+){0,3}?) (?:is|are|refers to|means) (?P<rest>.{20,})$"
)


def type_definitions(
    question: str, vectorstore: FAISS, source_filenames: Optional[Iterable[str]] = None
) -> List[tuple]:
    """
    For "what are the types of X?" return every (name, description) the material defines for an X.

    The list of types is usually spread over several slides ("A Perception Medium refers to ...",
    "A Storage Medium is ..."), so top-k retrieval alone returns only some of them. Returns [] when the
    question is not about types or fewer than three are found.
    """
    match = _TYPES_QUESTION.search(question or "")
    if not match:
        return []
    head = match.group("x").split()[-1].lower()
    heads = {head, head.rstrip("s"), "medium" if head == "media" else head}
    selected = set(source_filenames) if source_filenames is not None else None
    docs = [
        d for d in vectorstore.docstore._dict.values()
        if selected is None or Path(d.metadata.get("source", "")).name in selected
    ]
    docs.sort(key=lambda d: (Path(d.metadata.get("source", "")).name, d.metadata.get("page") or 0))
    found: dict = {}
    for doc in docs:
        for sentence in chunk_sentences(doc, max_len=300):
            m = _DEFINITION_SENTENCE.match(sentence)
            if not m:
                continue
            term, key = m.group("term"), m.group("term").lower()
            if key not in heads and any(key.endswith(" " + h) for h in heads):
                found.setdefault(key, (term, m.group("rest").rstrip(".")))
    return list(found.values()) if len(found) >= 3 else []


def format_type_definitions(types: List[tuple]) -> List[str]:
    return [f"- **{term}**: {rest[:1].upper() + rest[1:]}." for term, rest in types]


_QUESTION_WORD = r"(?:what|how|why|which|who|when|where|explain|define|describe|list|name)"
_LIST_NOUN = (r"(?:types|kinds|categories|parts|components|elements|steps|features|advantages|disadvantages|"
              r"examples|uses|applications|properties|characteristics|layers|stages|benefits)")


def split_question(question: str) -> List[str]:
    """Break "What is X and what are the types of Y?" (or "... and the six types of Y?") into separate questions."""
    text = (question or "").strip()
    parts = re.split(
        rf"(?<=\?)\s+|\s+(?:and|also|&)\s+(?=(?:{_QUESTION_WORD}\b|(?:the\s+)?(?:\w+\s+)?{_LIST_NOUN}\s+of\b))",
        text, flags=re.I,
    )
    parts = [p.strip(" ,.?") for p in parts if len(p.split()) >= 3]
    if len(parts) < 2 or not re.match(rf"{_QUESTION_WORD}\b", parts[0], re.I):
        return [question]
    ask = text.endswith("?")
    out = []
    for i, part in enumerate(parts):
        if i and not re.match(rf"{_QUESTION_WORD}\b", part, re.I):
            part = ("What are " if re.match(rf"(?:the\s+)?(?:\w+\s+)?{_LIST_NOUN}\b", part, re.I) else "What is ") + part
        out.append(part[:1].upper() + part[1:] + ("?" if ask else ""))
    return out


def _extractive_answer(question: str, chunks: List[Document], types: Optional[List[tuple]] = None) -> str:
    """Concise fallback answer from retrieved text when the local LLM is offline."""
    if types:
        return (
            "I found this in your uploaded material. The local LLM is unavailable, so this is an "
            "extractive answer from the most relevant chunks:\n\n"
            f"The material describes {len(types)} types:\n" + "\n".join(format_type_definitions(types))
        )
    terms = _content_terms(question)
    sentences = []
    for i, doc in enumerate(chunks, start=1):
        for sentence in chunk_sentences(doc, max_len=260):
            lower = sentence.lower()
            score = sum(1 for term in terms if term in lower)
            if score or not terms:
                sentences.append((score, i, sentence))
    if not sentences:
        # Nothing reads as a sentence (a table, say): show the start of the best chunk, cut on a word.
        for i, doc in enumerate(chunks[:2], start=1):
            text = " ".join(doc.page_content.split())
            if len(text) > 260:
                text = text[:260].rsplit(" ", 1)[0]
            sentences.append((0, i, text.rstrip(" ,;:—-") + "."))
    sentences.sort(key=lambda item: (-item[0], item[1]))
    bullets = []
    seen = set()
    for _, chunk_index, sentence in sentences:
        key = sentence.lower()[:120]
        if key in seen:
            continue
        seen.add(key)
        bullets.append(f"- {sentence} [Chunk {chunk_index}]")
        if len(bullets) == 4:
            break
    return (
        "I found this in your uploaded material. The local LLM is unavailable, so this is an "
        "extractive answer from the most relevant chunks:\n\n"
        + "\n".join(bullets)
    )


@dataclass
class RagResult:
    """Everything the UI (or another agent) needs to display/consume a query."""

    question: str
    answer: str
    source_documents: List[Document] = field(default_factory=list)
    grounded: bool = True  # False when the relevance gate short-circuited the LLM call
    top_score: Optional[float] = None  # best (lowest) FAISS distance; None if nothing retrieved


def format_context(chunks: List[Document]) -> str:
    """Concatenate retrieved chunks into one context string for the prompt."""
    parts = []
    for i, doc in enumerate(chunks, start=1):
        source = Path(doc.metadata.get("source", "unknown")).name
        page = doc.metadata.get("page", "?")
        section = doc.metadata.get("section")
        tag = f"[Chunk {i} | source: {source} | page: {page}"
        tag += f" | section: {section}]" if section else "]"
        parts.append(f"{tag}\n{doc.page_content}")
    return "\n\n".join(parts)


@dataclass
class Retrieval:
    """Outcome of the retrieval + relevance-gate step, before any LLM call."""

    chunks: List[Document]
    top_score: Optional[float]
    relevant: bool  # False when nothing retrieved is close enough to the query
    via: str = "embedding"  # "keyword" when the keyword rescue accepted the query


# Words that carry no topic information in a student's message.
STOPWORDS = _STOPWORDS = frozenset(
    "what is are was were the a an of in on for to and or how why when where which who whom does do did "
    "explain define describe tell me about give show with by from this that these those it its can could "
    "you please between difference differences vs versus quiz test mcq mcqs practice question questions "
    "simply simple terms step detail detailed use used using work works".split()
)

_RESCUE_MAX_TERMS = 3    # only short, keyword-style queries; long ones embed fine
_RESCUE_MAX_SHARE = 0.2  # a term in more than this share of chunks is not distinctive
_RESCUE_MIN_CAP = 5      # ...but always allow at least this many matching chunks


def _content_terms(question: str) -> List[str]:
    """The distinctive words of a query (no stopwords, no bare numbers, deduplicated)."""
    seen: List[str] = []
    for token in re.findall(r"[a-z0-9][a-z0-9\-]*", question.lower()):
        token = token.strip("-")
        if len(token) >= 3 and not token.isdigit() and token not in _STOPWORDS and token not in seen:
            seen.append(token)
    return seen


def _keyword_rescue(
    question: str,
    vectorstore: FAISS,
    k: int,
    source_filenames: Optional[set],
) -> List[Document]:
    """
    Chunks that literally contain every content word of a short query.

    Embedding similarity is weak for one- or two-word queries ("ACID") against
    long chunks, so the distance gate can reject a topic the notes plainly
    cover. This accepts such a query only when ALL its content words appear
    together in a chunk and those words are distinctive (not in most chunks),
    which keeps genuinely off-topic queries rejected: no chunk contains
    "kubernetes" and "pod". Scans the docstore directly, which is cheap at
    course-notes scale.
    """
    terms = _content_terms(question)
    if not terms or len(terms) > _RESCUE_MAX_TERMS:
        return []

    patterns = [re.compile(rf"\b{re.escape(t)}\b") for t in terms]
    documents = list(vectorstore.docstore._dict.values())
    hits = []
    for doc in documents:
        if source_filenames is not None and Path(doc.metadata.get("source", "")).name not in source_filenames:
            continue
        text = doc.page_content.lower()
        if all(p.search(text) for p in patterns):
            hits.append((sum(len(p.findall(text)) for p in patterns), doc))

    if not hits or len(hits) > max(_RESCUE_MIN_CAP, _RESCUE_MAX_SHARE * len(documents)):
        return []
    hits.sort(key=lambda item: -item[0])
    return [doc for _, doc in hits[:k]]


def retrieve(
    question: str,
    vectorstore: FAISS,
    k: int = TOP_K,
    score_threshold: float = RELEVANCE_SCORE_THRESHOLD,
    source_filenames: Optional[Iterable[str]] = None,
) -> Retrieval:
    """
    Retrieve the top-k chunks for `question` and apply the relevance gate.

    If even the closest chunk is farther than `score_threshold` from the
    query, the question is treated as out of scope for the uploaded notes
    (`relevant=False`, no chunks returned). This is a deterministic backstop
    for the "don't answer from outside the notes" instruction in the prompt --
    small local models don't reliably self-enforce that constraint through
    the prompt alone.

    `source_filenames`, if given, restricts retrieval to chunks whose
    `source` metadata filename is in that set (lets a caller scope a
    question to specific uploaded files instead of the whole index).
    """
    filter_func = None
    if source_filenames is not None:
        selected = set(source_filenames)
        filter_func = lambda metadata: Path(metadata.get("source", "")).name in selected

    # When filtering, fetch across the whole index rather than the default
    # fetch_k=20 candidate pool, so a question scoped to one file among many
    # still finds that file's best-matching chunks instead of missing them
    # because they weren't in the top 20 over the full index.
    fetch_k = vectorstore.index.ntotal if filter_func else k
    scored = vectorstore.similarity_search_with_score(
        question, k=k, filter=filter_func, fetch_k=fetch_k
    )

    top_score = float(min(score for _, score in scored)) if scored else None
    if top_score is None or top_score > score_threshold:
        selected = set(source_filenames) if source_filenames is not None else None
        rescued = _keyword_rescue(question, vectorstore, k, selected)
        if rescued:
            return Retrieval(chunks=rescued, top_score=top_score, relevant=True, via="keyword")
        return Retrieval(chunks=[], top_score=top_score, relevant=False)

    return Retrieval(chunks=[doc for doc, _ in scored], top_score=top_score, relevant=True)


def _answer_one(
    question: str,
    vectorstore: FAISS,
    k: int = TOP_K,
    score_threshold: float = RELEVANCE_SCORE_THRESHOLD,
    source_filenames: Optional[Iterable[str]] = None,
    llm=None,
) -> RagResult:
    """Run the full retrieve -> generate pipeline for one user question."""
    retrieval = retrieve(question, vectorstore, k, score_threshold, source_filenames)

    if not retrieval.relevant:
        return RagResult(
            question=question,
            answer=_NOT_COVERED_MESSAGE,
            grounded=False,
            top_score=retrieval.top_score,
        )

    types = type_definitions(question, vectorstore, source_filenames)
    chunks = list(retrieval.chunks)
    if types:  # give the model the complete list, not just the slides that ranked highest
        chunks.append(Document(
            page_content="Types listed in the material: " + " ".join(f"{t}: {r}." for t, r in types),
            metadata={"source": "types list", "page": 0},
        ))
    context = format_context(chunks)
    chain = _PROMPT | (llm if llm is not None else get_llm()) | StrOutputParser()
    try:
        answer = chain.invoke({"context": context, "question": question})
    except Exception:
        answer = _extractive_answer(question, retrieval.chunks, types)

    return RagResult(
        question=question,
        answer=answer,
        source_documents=retrieval.chunks,
        top_score=retrieval.top_score,
    )


def answer_question(
    question: str,
    vectorstore: FAISS,
    k: int = TOP_K,
    score_threshold: float = RELEVANCE_SCORE_THRESHOLD,
    source_filenames: Optional[Iterable[str]] = None,
    llm=None,
) -> RagResult:
    """Answer a question; a message that asks several things is answered part by part."""
    parts = split_question(question)
    if len(parts) == 1:
        return _answer_one(question, vectorstore, k, score_threshold, source_filenames, llm)

    results = [_answer_one(part, vectorstore, k, score_threshold, source_filenames, llm) for part in parts]
    sections = [f"**{part}**\n{result.answer.strip()}" for part, result in zip(parts, results)]
    scores = [r.top_score for r in results if r.top_score is not None]
    return RagResult(
        question=question,
        answer="\n\n".join(sections),
        source_documents=[d for r in results for d in r.source_documents],
        grounded=any(r.grounded for r in results),
        top_score=min(scores) if scores else None,
    )
