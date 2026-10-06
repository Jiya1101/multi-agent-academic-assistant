"""
Quiz Generator: writes multiple-choice questions strictly from retrieved notes.

Local models often return malformed or sloppy JSON, so nothing is trusted:
every question is validated, every question must point at a real retrieved
chunk, and each saved question carries the slide it came from so a professor
can check it. Options are shuffled in code because models bias the correct
answer toward one position.
"""

import json
import random
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agents.base import Agent, AgentContext, AgentResult
from rag_core.chain import NOT_COVERED_MESSAGE, format_context, retrieve
from rag_core.insights import topic_title
from rag_core.learning_log import SCOPE_PERSONAL, create_quiz
from rag_core.llm import get_json_llm

_SYSTEM = """You write multiple-choice questions for a university course, strictly from the \
course-note context below.

Write exactly {n} questions about: {topic}

Rules:
- Each question must be answerable from the context alone. Do not use outside knowledge.
- Exactly 4 options and exactly one correct. Wrong options must be plausible but clearly wrong \
according to the context.
- "chunk" is the number N of the [Chunk N] the question is based on.
- "explanation" is one sentence saying why the answer is correct, using the context.

Return ONLY JSON in exactly this shape:
{{"questions": [{{"question": "...", "options": ["...", "...", "...", "..."], "answer_index": 0, "chunk": 1, "explanation": "..."}}]}}

--- CONTEXT ---
{context}
"""

_PROMPT = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", "Write the quiz.")])


def parse_quiz_json(raw: str, n_chunks: int) -> List[Dict[str, Any]]:
    """Return only the well-formed questions from the model's raw JSON output."""
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    items = payload.get("questions") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return []

    valid: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        question = item.get("question")
        options = item.get("options")
        answer_index = item.get("answer_index")
        chunk = item.get("chunk")
        explanation = item.get("explanation", "")
        if not (isinstance(question, str) and question.strip()):
            continue
        if not (isinstance(options, list) and len(options) == 4):
            continue
        if not all(isinstance(o, str) and o.strip() for o in options):
            continue
        if len({o.strip().lower() for o in options}) != 4:
            continue
        if not (isinstance(answer_index, int) and not isinstance(answer_index, bool)
                and 0 <= answer_index < 4):
            continue
        if not (isinstance(chunk, int) and not isinstance(chunk, bool) and 1 <= chunk <= n_chunks):
            continue
        valid.append(
            {
                "question": question.strip(),
                "options": [o.strip() for o in options],
                "answer_index": answer_index,
                "chunk": chunk,
                "explanation": explanation.strip() if isinstance(explanation, str) else "",
            }
        )
    return valid


def shuffle_options(question: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """Shuffle the options and keep `answer_index` pointing at the right one."""
    correct = question["options"][question["answer_index"]]
    options = question["options"][:]
    rng.shuffle(options)
    return {**question, "options": options, "answer_index": options.index(correct)}


def grade_answers(questions: List[Dict[str, Any]], answers: List[Optional[int]]) -> Dict[str, Any]:
    """Score a student's answers (None = unanswered, counts as wrong)."""
    per_question = [
        a is not None and a == q["answer_index"] for q, a in zip(questions, answers)
    ]
    return {"correct": sum(per_question), "total": len(questions), "per_question": per_question}


class QuizGenerator(Agent):
    name = "Quiz Generator"
    job = "Write grounded multiple-choice questions on a topic and save them as a quiz."

    def run(
        self,
        request: str,
        ctx: AgentContext,
        n_questions: int = 3,
        scope: str = SCOPE_PERSONAL,
        topic: Optional[str] = None,
        **_,
    ) -> AgentResult:
        retrieval = retrieve(request, ctx.vectorstore, source_filenames=ctx.source_filenames)
        if not retrieval.relevant:
            return AgentResult(
                agent=self.name, kind="quiz", text=NOT_COVERED_MESSAGE, grounded=False,
                data={"error": "not_covered"},
            )

        chunks = retrieval.chunks
        topic = topic or topic_title(chunks[0].metadata)
        chain = _PROMPT | (ctx.llm if ctx.llm is not None else get_json_llm()) | StrOutputParser()
        variables = {"n": n_questions, "topic": topic, "context": format_context(chunks)}

        questions: List[Dict[str, Any]] = []
        for _attempt in range(2):  # one retry: small models fail JSON now and then
            questions = parse_quiz_json(chain.invoke(variables), n_chunks=len(chunks))
            if questions:
                break
        if not questions:
            return AgentResult(
                agent=self.name, kind="quiz", grounded=False,
                text="The model did not produce a usable quiz. Try again.",
                data={"error": "invalid_output"},
            )

        rng = random.Random(f"{topic}-{request}")
        saved = []
        for question in questions[:n_questions]:
            question = shuffle_options(question, rng)
            doc = chunks[question["chunk"] - 1]
            question["source"] = {
                "file": doc.metadata.get("source", "unknown").replace("\\", "/").split("/")[-1],
                "page": (doc.metadata.get("page") or 0) + 1,
                "slide": topic_title(doc.metadata),
            }
            saved.append(question)

        quiz_id = create_quiz(
            topic=topic, prompt=request, questions=saved, scope=scope,
            owner=ctx.student_id, db_dir=ctx.db_dir,
        )
        return AgentResult(
            agent=self.name,
            kind="quiz",
            text=f"{len(saved)}-question quiz on {topic}.",
            sources=chunks,
            data={"quiz_id": quiz_id, "topic": topic, "questions": saved, "scope": scope},
        )
