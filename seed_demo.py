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
    python seed_demo.py --quiz-attempts   # also simulate 24 students taking the published class quizzes
"""

import argparse
import random
from datetime import datetime, timedelta, timezone

from rag_core.chain import retrieve
from rag_core.config import DB_DIR
from rag_core.learning_log import SCOPE_CLASS, delete_attempts_by_prefix, list_quizzes, record_attempt
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
DEMO_SESSION = "demo_session_1"
DEMO_STUDENT_QUESTIONS = [
    "What is the difference between ACID and BASE?",
    "Why was the BASE model introduced after ACID?",
    "What does BASE stand for in databases?",
    "When should I use BASE instead of ACID?",
    "What are atomicity and isolation in ACID?",
    "How do RDS read replicas work?",
]


# Simulated students who take the published class quizzes (--quiz-attempts). Their answers follow patterns
# planted on purpose, by question position, so the Question analysis tab can be checked against the truth.
# The pattern repeats every 6 questions: most are ordinary (stronger students do better), plus one easy
# question, one hard question with a popular wrong answer (a misconception) and one faulty answer key
# (the keyed answer is chosen LESS often by stronger students). Faulty keys are kept rare on purpose:
# the check compares each question with the others, so it cannot work if most questions are faulty.
DEMO_CLASS_PREFIX = "demo_class_"
DEMO_CLASS_SIZE = 24
PLANTED = ("ordinary", "ordinary", "misconception", "easy", "ordinary", "faulty key")


def simulate_class(quizzes, size: int = DEMO_CLASS_SIZE, seed: int = 11):
    """
    Simulate `size` students taking every quiz in `quizzes` (no database involved).
    Returns (plan, attempts): the planted truth as (quiz, number, question, kind, popular_wrong_index)
    tuples, and attempt rows in the shape learning_log.list_attempts returns.
    """
    rng = random.Random(seed)
    position = 0
    plan = []
    for quiz in quizzes:
        for number, q in enumerate(quiz["questions"], start=1):
            popular_wrong = rng.choice([i for i in range(len(q["options"])) if i != q["answer_index"]])
            plan.append((quiz, number, q, PLANTED[position % len(PLANTED)], popular_wrong))
            position += 1

    taken = {s: [] for s in range(size)}
    for student in range(size):
        ability = (student + 0.5) / size                       # 0 = weakest, 1 = strongest
        for quiz, _number, q, kind, popular_wrong in plan:
            p_right = {"easy": 0.96, "ordinary": 0.15 + 0.8 * ability,
                       "misconception": 0.08 + 0.3 * ability, "faulty key": 0.9 - 0.8 * ability}[kind]
            wrong = [i for i in range(len(q["options"])) if i != q["answer_index"]]
            if rng.random() < 0.02:
                pick = None                                    # occasionally skipped
            elif rng.random() < p_right:
                pick = q["answer_index"]
            elif kind == "misconception" and rng.random() < 0.75:
                pick = popular_wrong
            else:
                pick = rng.choice(wrong)
            taken[student].append((quiz["id"], pick))

    attempts = []
    for student, picks in taken.items():
        for quiz in quizzes:
            answers = [p for qid, p in picks if qid == quiz["id"]]
            correct = sum(1 for a, q in zip(answers, quiz["questions"]) if a == q["answer_index"])
            attempts.append({
                "id": len(attempts) + 1, "student_id": f"{DEMO_CLASS_PREFIX}{student + 1:02d}",
                "quiz_id": quiz["id"], "topic": quiz["topic"], "correct": correct,
                "total": len(quiz["questions"]), "answers": answers,
            })
    return plan, attempts


def simulate_quiz_attempts(seed: int = 11) -> None:
    """Have DEMO_CLASS_SIZE simulated students take every published class quiz and save the attempts."""
    quizzes = sorted(list_quizzes(SCOPE_CLASS, DB_DIR), key=lambda q: q["id"])
    if not quizzes:
        print("No class quizzes are published yet, so no quiz attempts were simulated. "
              "Run: python publish_quizzes.py")
        return
    plan, attempts = simulate_class(quizzes, DEMO_CLASS_SIZE, seed)
    for a in attempts:
        record_attempt(a["student_id"], a["quiz_id"], a["topic"], a["correct"], a["total"], a["answers"], DB_DIR)

    print(f"Simulated {DEMO_CLASS_SIZE} students taking {len(quizzes)} class quiz(zes). Planted truth:")
    for quiz, number, q, kind, popular_wrong in plan:
        extra = f" (popular wrong answer: {q['options'][popular_wrong]!r})" if kind == "misconception" else ""
        print(f"  quiz {quiz['id']} question {number}: {kind}{extra}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the question log with a simulated class.")
    parser.add_argument("--clear", action="store_true", help="Remove all simulated rows and exit.")
    parser.add_argument("--quiz-attempts", action="store_true",
                        help="Also simulate students taking the published class quizzes (run publish_quizzes.py first).")
    args = parser.parse_args()

    removed = delete_by_scope_marker(DEMO_MARKER, DB_DIR)
    removed_attempts = delete_attempts_by_prefix(DEMO_CLASS_PREFIX, DB_DIR)
    if args.clear:
        print(f"Removed {removed} simulated question(s) and {removed_attempts} simulated quiz attempt(s).")
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
            agent="Doubt Resolver",
        )
        if not retrieval.relevant:
            gaps.append((topic, question, retrieval.top_score))

    # One sitting, 4 minutes apart, so the repeated ACID/BASE questions show up as follow-ups.
    sitting = datetime.now(timezone.utc) - timedelta(days=1)
    for number, question in enumerate(DEMO_STUDENT_QUESTIONS):
        retrieval = retrieve(question, vectorstore)
        log_query(
            question=question,
            grounded=retrieval.relevant,
            top_score=retrieval.top_score,
            source_documents=retrieval.chunks,
            scope=[DEMO_MARKER],
            db_dir=DB_DIR,
            student_id=DEMO_STUDENT,
            agent="Doubt Resolver",
            session_id=DEMO_SESSION,
            asked_at=(sitting + timedelta(minutes=4 * number)).isoformat(timespec="seconds"),
        )
    print(f"Added {len(DEMO_STUDENT_QUESTIONS)} question(s) for student '{DEMO_STUDENT}'.")
    if args.quiz_attempts:
        simulate_quiz_attempts()

    print(f"Seeded {len(questions)} simulated question(s); {len(gaps)} not covered by the notes:")
    for topic, question, score in gaps:
        print(f"  [{topic}] {question}  (best score {score:.2f})")


if __name__ == "__main__":
    main()
