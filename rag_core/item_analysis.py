"""
Quiz item analysis: what the class's answers say about each quiz QUESTION.

The existing class results table only averages scores per topic. This module
looks one level down, at every question and every answer option, and answers
three things a professor can act on:

  - Which WRONG answer did many students pick? A wrong option chosen far more
    often than chance would give it is the class's likely misconception
    ("most of you think BASE means ..."). A statistical test keeps ordinary
    luck in a small class from being reported as a misconception.
  - Is the question itself any good? Too easy, very hard, or a keyed answer
    that strong students tend to avoid (often a wrong key or bad wording).
    Quizzes are written by a small local model, so this checks its work.
  - Is there enough data to say so? Nothing is reported for a quiz until at
    least MIN_COHORT different students have taken it.

Counting rules (kept simple on purpose, so they can be explained and checked):
  - Only each student's FIRST attempt at a quiz counts, so retakes cannot
    inflate the numbers.
  - A skipped question counts as wrong, the same as in grading.
  - Discrimination compares a question with the student's results on the OTHER
    questions they answered (all quizzes pooled). It is only as good as those
    other questions: if many of them are faulty it is unreliable. With few
    students it is noisy, so it is left out below MIN_FOR_DISCRIMINATION
    students, or when fewer than MIN_GROUP students got the question right
    (or wrong), and the "check the answer key" flag needs a statistically
    clear reversal (p < 0.05), not just a negative number. A hint, never proof.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from scipy.stats import binomtest, pearsonr

from rag_core.config import MIN_COHORT

# Thresholds are judgment calls, not statistical constants; they are named here so they are easy to change.
MISCONCEPTION_SHARE = 0.25     # a wrong option picked by at least this share of the class...
MISCONCEPTION_MIN_STUDENTS = 3 # ...and by at least this many students
TOO_EASY_AT = 0.90
VERY_HARD_AT = 0.30
MIN_FOR_DISCRIMINATION = 10    # students needed before discrimination is computed
MIN_REST_ITEMS = 4             # other answered questions a student needs to give an ability estimate
MIN_GROUP = 5                  # students on the smaller side (right or wrong) before discrimination is computed
KEY_SUSPECT_AT = -0.20         # discrimination at or below this (and significant): strong students avoid the key
SIGNIFICANT_P = 0.05
# Stricter for misconceptions because every wrong option of every question is tested at once: at 0.05, about
# one ordinary question in a nine-question set would be reported by luck alone (seen on simulated data).
MISCONCEPTION_P = 0.01
UNUSED_OPTION_MIN_STUDENTS = 10

FLAG_TOO_EASY = "Too easy"
FLAG_VERY_HARD = "Very hard"
FLAG_CHECK_KEY = "Check the answer key"
FLAG_OUTVOTED = "A wrong answer beat the right one"
FLAG_UNUSED = "An option nobody chose"

FLAG_HELP = {
    FLAG_TOO_EASY: "Nearly everyone got it right, so it tells you little. Fine as a warm-up.",
    FLAG_VERY_HARD: "Most of the class got it wrong. Check the wording first, then whether the topic needs re-teaching.",
    FLAG_CHECK_KEY: "Students who did well on the other questions tended to get this one wrong. "
                    "Re-read the question and the marked answer: the key or the wording may be faulty.",
    FLAG_OUTVOTED: "More students chose one wrong option than the correct one. Either a real misconception "
                   "or a faulty key.",
    FLAG_UNUSED: "At least one wrong option was never chosen, so it is not doing its job as a distractor.",
}


@dataclass
class OptionStat:
    index: int
    text: str
    count: int          # students who picked it
    share: float        # count / students who took the quiz
    is_correct: bool


@dataclass
class QuestionStat:
    quiz_id: int
    topic: str
    number: int                       # 1-based position in the quiz
    question: str
    source: Dict[str, Any]
    students: int
    skipped: int
    correct_rate: float
    options: List[OptionStat]
    misconception: Optional[OptionStat] = None
    discrimination: Optional[float] = None
    discrimination_p: Optional[float] = None   # how likely a value this far from zero is by chance alone
    flags: List[str] = field(default_factory=list)

    @property
    def correct_option(self) -> OptionStat:
        return next(o for o in self.options if o.is_correct)


@dataclass
class QuizAnalysis:
    quiz_id: int
    topic: str
    students: int
    reportable: bool                  # False until MIN_COHORT students have taken it
    needed: int = MIN_COHORT
    average: Optional[float] = None
    questions: List[QuestionStat] = field(default_factory=list)  # empty unless reportable


def _answer_at(answers: List[Optional[int]], index: int, n_options: int) -> Optional[int]:
    """The option picked for question `index`, or None if skipped / missing / out of range."""
    if index >= len(answers):
        return None
    pick = answers[index]
    return pick if isinstance(pick, int) and not isinstance(pick, bool) and 0 <= pick < n_options else None


def _first_attempts(attempts: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Each student's first attempt at each quiz."""
    first: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for attempt in sorted(attempts, key=lambda a: a["id"]):
        first.setdefault((attempt["student_id"], attempt["quiz_id"]), attempt)
    return first


def _ability_pool(
    quizzes: Dict[int, Dict[str, Any]], first: Dict[Tuple[str, int], Dict[str, Any]]
) -> Dict[str, Tuple[int, int]]:
    """Per student: (questions answered correctly, questions in the quizzes they took), all quizzes pooled."""
    pool: Dict[str, List[int]] = {}
    for (student, quiz_id), attempt in first.items():
        quiz = quizzes.get(quiz_id)
        if not quiz:
            continue
        for i, q in enumerate(quiz["questions"]):
            pick = _answer_at(attempt["answers"], i, len(q["options"]))
            entry = pool.setdefault(student, [0, 0])
            entry[0] += int(pick is not None and pick == q["answer_index"])
            entry[1] += 1
    return {s: (c, t) for s, (c, t) in pool.items()}


def _discrimination(
    item_correct: Dict[str, bool], pool: Dict[str, Tuple[int, int]]
) -> Tuple[Optional[float], Optional[float]]:
    """
    (correlation, p-value) between getting this question right and doing well on the student's other
    questions, or (None, None) when there is too little data for it to mean anything.
    """
    xs: List[float] = []
    ys: List[float] = []
    for student, correct in item_correct.items():
        got, total = pool.get(student, (0, 0))
        rest_total = total - 1
        if rest_total < MIN_REST_ITEMS:
            continue
        xs.append(float(correct))
        ys.append((got - int(correct)) / rest_total)
    right = int(sum(xs))
    if (
        len(xs) < MIN_FOR_DISCRIMINATION
        or min(right, len(xs) - right) < MIN_GROUP        # a question nearly everyone got right (or wrong) says little
        or len(set(ys)) < 2
    ):
        return None, None
    r, p = pearsonr(xs, ys)
    return float(r), float(p)


def _beats_guessing(count: int, wrong_picks: int, wrong_options: int) -> bool:
    """
    True when one wrong option was chosen clearly more often than chance would give it if students who
    were wrong simply spread out over the wrong options. In a class of a few dozen, an option reaching
    25 to 30 percent happens by luck alone, so a share on its own is not evidence of a shared mistake.
    The p-value is multiplied by the number of wrong options because the busiest one is picked after looking.
    """
    if wrong_options < 2 or wrong_picks == 0:
        return False
    p = binomtest(count, wrong_picks, 1 / wrong_options, alternative="greater").pvalue
    return min(1.0, p * wrong_options) < MISCONCEPTION_P


def _analyze_question(
    quiz: Dict[str, Any], number: int, students: List[str],
    first: Dict[Tuple[str, int], Dict[str, Any]], pool: Dict[str, Tuple[int, int]],
) -> QuestionStat:
    q = quiz["questions"][number - 1]
    n_options = len(q["options"])
    counts = [0] * n_options
    skipped = 0
    item_correct: Dict[str, bool] = {}
    for student in students:
        pick = _answer_at(first[(student, quiz["id"])]["answers"], number - 1, n_options)
        if pick is None:
            skipped += 1
        else:
            counts[pick] += 1
        item_correct[student] = pick is not None and pick == q["answer_index"]

    n = len(students)
    options = [
        OptionStat(i, str(text), counts[i], counts[i] / n, i == q["answer_index"])
        for i, text in enumerate(q["options"])
    ]
    correct = options[q["answer_index"]]
    wrong = [o for o in options if not o.is_correct]
    top_wrong = max(wrong, key=lambda o: (o.count, -o.index)) if wrong else None
    misconception = None
    if (
        top_wrong
        and top_wrong.share >= MISCONCEPTION_SHARE
        and top_wrong.count >= MISCONCEPTION_MIN_STUDENTS
        and _beats_guessing(top_wrong.count, sum(o.count for o in wrong), len(wrong))
    ):
        misconception = top_wrong

    correct_rate = correct.count / n
    discrimination, p_value = _discrimination(item_correct, pool)
    flags: List[str] = []
    if correct_rate >= TOO_EASY_AT:
        flags.append(FLAG_TOO_EASY)
    if correct_rate <= VERY_HARD_AT:
        flags.append(FLAG_VERY_HARD)
    # Only a clearly reversed pattern counts: in a class this size a mild negative value is ordinary noise.
    if discrimination is not None and discrimination <= KEY_SUSPECT_AT and p_value < SIGNIFICANT_P:
        flags.append(FLAG_CHECK_KEY)
    if misconception and misconception.count > correct.count:
        flags.append(FLAG_OUTVOTED)
    # On an easy question nearly everyone is right, so unused wrong options are expected, not a defect.
    if n >= UNUSED_OPTION_MIN_STUDENTS and correct_rate < TOO_EASY_AT and any(o.count == 0 for o in wrong):
        flags.append(FLAG_UNUSED)

    return QuestionStat(
        quiz_id=quiz["id"], topic=quiz["topic"], number=number, question=q["question"],
        source=q.get("source", {}), students=n, skipped=skipped, correct_rate=correct_rate,
        options=options, misconception=misconception, discrimination=discrimination,
        discrimination_p=p_value, flags=flags,
    )


def analyze_quizzes(
    quizzes: Iterable[Dict[str, Any]],
    attempts: Iterable[Dict[str, Any]],
    min_students: int = MIN_COHORT,
) -> List[QuizAnalysis]:
    """
    Analyse every quiz that has at least one attempt. Quizzes with fewer than
    `min_students` different students come back with `reportable=False` and
    no per-question numbers at all.
    """
    quiz_by_id = {q["id"]: q for q in quizzes}
    first = _first_attempts(attempts)
    pool = _ability_pool(quiz_by_id, first)

    students_by_quiz: Dict[int, List[str]] = {}
    for (student, quiz_id) in first:
        if quiz_id in quiz_by_id:
            students_by_quiz.setdefault(quiz_id, []).append(student)

    results: List[QuizAnalysis] = []
    for quiz_id, quiz in quiz_by_id.items():
        students = sorted(students_by_quiz.get(quiz_id, []))
        if not students:
            continue
        analysis = QuizAnalysis(
            quiz_id=quiz_id, topic=quiz["topic"], students=len(students),
            reportable=len(students) >= min_students, needed=min_students,
        )
        if analysis.reportable:
            analysis.questions = [
                _analyze_question(quiz, number, students, first, pool)
                for number in range(1, len(quiz["questions"]) + 1)
            ]
            fractions = [
                first[(s, quiz_id)]["correct"] / max(first[(s, quiz_id)]["total"], 1) for s in students
            ]
            analysis.average = sum(fractions) / len(fractions)
        results.append(analysis)
    return results


def likely_misconceptions(analyses: Iterable[QuizAnalysis]) -> List[QuestionStat]:
    """Questions where one wrong answer drew a large share of the class, most popular first."""
    found = [q for a in analyses for q in a.questions if q.misconception]
    return sorted(found, key=lambda q: -q.misconception.share)


def questions_to_review(analyses: Iterable[QuizAnalysis]) -> List[QuestionStat]:
    """Questions with at least one quality flag; the most serious kinds (key, outvoted) first."""
    urgent = (FLAG_CHECK_KEY, FLAG_OUTVOTED)
    found = [q for a in analyses for q in a.questions if q.flags]
    return sorted(found, key=lambda q: (not any(f in urgent for f in q.flags), q.correct_rate))
