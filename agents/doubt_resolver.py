"""
Doubt Resolver: answers a student's question from the course notes.

Tools: retrieval with the relevance gate, Llama 3, the question log. It logs
its own outcome (answered or not covered), which is the raw material the
Faculty Insight and Gap Handler agents work from.
"""

from agents.base import Agent, AgentContext, AgentResult
from rag_core.chain import answer_question
from rag_core.query_log import log_query


class DoubtResolver(Agent):
    name = "Doubt Resolver"
    job = "Answer a question from the notes, with citations, or say the notes do not cover it."

    def run(self, request: str, ctx: AgentContext, **_) -> AgentResult:
        result = answer_question(
            request,
            ctx.vectorstore,
            source_filenames=ctx.source_filenames,
            llm=ctx.llm,
        )
        log_id = log_query(
            question=result.question,
            grounded=result.grounded,
            top_score=result.top_score,
            source_documents=result.source_documents,
            scope=ctx.source_filenames,
            db_dir=ctx.db_dir,
            student_id=ctx.student_id,
            agent=self.name,
            session_id=ctx.session_id,
        )
        return AgentResult(
            agent=self.name,
            kind="answer",
            text=result.answer,
            sources=result.source_documents,
            grounded=result.grounded,
            data={"question": result.question, "log_id": log_id, "top_score": result.top_score},
        )
