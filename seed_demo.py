"""
seed_demo.py
============
Populate the question log with a simulated class so the Professor dashboard
has realistic data to show during a demo.

Each simulated question goes through the real retrieval + relevance gate
(no LLM call, so this runs in seconds), so which questions become "gaps"
is decided by the actual index, not hard-coded. Rows are tagged so they can
be removed without touching real student questions.

Usage:
    python seed_demo.py           # add the simulated class (replaces any earlier demo rows)
    python seed_demo.py --clear   # remove all simulated rows
"""

import argparse
import random

from rag_core.chain import retrieve
from rag_core.config import DB_DIR
from rag_core.query_log import delete_by_scope_marker, log_query
from rag_core.vectorstore import load_vectorstore, vectorstore_exists

DEMO_MARKER = "__demo_seed__"

# Topic -> the different ways students phrase the same doubt. Topics are
# drawn from the indexed notes (AWS database services, Spark), plus a few
# things the notes deliberately do not cover (logistics, off-syllabus).
SIMULATED_QUESTIONS = {
    "ACID vs BASE": [
        "What is the difference between ACID and BASE?",
        "ACID vs BASE consistency models",
        "Explain the ACID properties",
        "what does BASE stand for in databases",
        "Why was the BASE model introduced after ACID?",
        "When should I use BASE instead of ACID?",
        "What are atomicity and isolation in ACID?",
    ],
    "RDS Multi-AZ vs read replicas": [
        "What is the difference between Multi-AZ and read replicas in RDS?",
        "Multi-AZ vs read replica",
        "When do I use a read replica in Amazon RDS?",
        "Does Multi-AZ improve read performance?",
        "How do RDS read replicas work?",
        "Explain Multi-AZ deployment",
    ],
    "OLTP vs OLAP": [
        "What is the difference between OLTP and OLAP?",
        "OLTP vs OLAP systems",
        "Which one is used for data warehouses, OLTP or OLAP?",
        "explain OLAP",
    ],
    "Spark fault tolerance": [
        "How does Spark recover a lost partition?",
        "What is lineage in Spark?",
        "Explain Spark fault tolerance",
        "How does Spark recompute lost data?",
    ],
    "Spark on YARN": [
        "What is YARN client mode?",
        "Difference between YARN client mode and cluster mode",
        "How does Spark run on YARN?",
    ],
    "RDS backups": [
        "How do automated backups work in RDS?",
        "What is the RDS backup window?",
    ],
    "Aurora": [
        "What is Amazon Aurora?",
        "How is Aurora different from RDS?",
    ],
    # Not in the notes: these should surface as gaps for the professor.
    "Assignment logistics": [
        "What is the deadline for assignment 2?",
        "when is assignment 2 due",
        "assignment 2 submission date?",
        "Is there an extension for assignment 2?",
    ],
    "Exam scope": [
        "Is Spark included in the end semester exam?",
        "will spark topics come in the final exam",
    ],
    "Off-syllabus": [
        "What is a Kubernetes pod?",
        "Explain pods in Kubernetes",
        "how do kubernetes pods work",
    ],
}


# One identified student, so the Progress Tracker has history to show. Their
# questions pile up on ACID/BASE (struggling) with a single one elsewhere.
DEMO_STUDENT = "demo_student"
DEMO_STUDENT_QUESTIONS = [
    "What is the difference between ACID and BASE?",
    "Why was the BASE model introduced after ACID?",
    "What does BASE stand for in databases?",
    "When should I use BASE instead of ACID?",
    "What are atomicity and isolation in ACID?",
    "How do RDS read replicas work?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the question log with a simulated class.")
    parser.add_argument("--clear", action="store_true", help="Remove all simulated rows and exit.")
    args = parser.parse_args()

    removed = delete_by_scope_marker(DEMO_MARKER, DB_DIR)
    if args.clear:
        print(f"Removed {removed} simulated question(s).")
        return
    if removed:
        print(f"Replaced {removed} earlier simulated question(s).")

    if not vectorstore_exists(DB_DIR):
        print("No index found. Build it first: python ingest.py")
        return
    vectorstore = load_vectorstore(DB_DIR)

    questions = [(topic, q) for topic, qs in SIMULATED_QUESTIONS.items() for q in qs]
    random.Random(7).shuffle(questions)  # so the log is not grouped by topic

    gaps = []
    for topic, question in questions:
        retrieval = retrieve(question, vectorstore)
        log_query(
            question=question,
            grounded=retrieval.relevant,
            top_score=retrieval.top_score,
            source_documents=retrieval.chunks,
            scope=[DEMO_MARKER],
            db_dir=DB_DIR,
        )
        if not retrieval.relevant:
            gaps.append((topic, question, retrieval.top_score))

    for question in DEMO_STUDENT_QUESTIONS:
        retrieval = retrieve(question, vectorstore)
        log_query(
            question=question,
            grounded=retrieval.relevant,
            top_score=retrieval.top_score,
            source_documents=retrieval.chunks,
            scope=[DEMO_MARKER],
            db_dir=DB_DIR,
            student_id=DEMO_STUDENT,
        )
    print(f"Added {len(DEMO_STUDENT_QUESTIONS)} question(s) for student '{DEMO_STUDENT}'.")

    print(f"Seeded {len(questions)} simulated question(s); {len(gaps)} not covered by the notes:")
    for topic, question, score in gaps:
        print(f"  [{topic}] {question}  (best score {score:.2f})")


if __name__ == "__main__":
    main()
