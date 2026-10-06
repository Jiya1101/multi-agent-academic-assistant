"""
publish_quizzes.py
==================
Run the professor pipeline from the command line: Faculty Insight finds the
topics the class keeps asking about, then the Quiz Generator writes and
publishes a quiz on each.

Each quiz takes a minute or two with Llama 3 on CPU, so run this before a
demo rather than waiting for it in the browser.

Usage:
    python publish_quizzes.py              # up to 3 topics, 3 questions each
    python publish_quizzes.py --topics 2 --questions 4
"""

import argparse

from agents import AgentContext, Orchestrator
from rag_core.config import DB_DIR
from rag_core.embeddings import get_embeddings
from rag_core.vectorstore import load_vectorstore, vectorstore_exists


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish class quizzes from class confusion.")
    parser.add_argument("--topics", type=int, default=3, help="Max confusion topics to quiz.")
    parser.add_argument("--questions", type=int, default=3, help="Questions per quiz.")
    args = parser.parse_args()

    if not vectorstore_exists(DB_DIR):
        print("No index found. Build it first: python ingest.py")
        return

    ctx = AgentContext(
        vectorstore=load_vectorstore(DB_DIR),
        db_dir=DB_DIR,
        embeddings=get_embeddings(),
    )
    result = Orchestrator().publish_class_quizzes(ctx, top_n=args.topics, n_questions=args.questions)
    for line in result.trace:
        print(line)
    published = [r for r in result.results if r.kind == "quiz" and "quiz_id" in r.data]
    print(f"Published {len(published)} new quiz(zes).")


if __name__ == "__main__":
    main()
