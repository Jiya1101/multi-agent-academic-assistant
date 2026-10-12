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
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agents.base import Agent, AgentContext, AgentResult
from agents.note_generator import _stems, major_headings, support
from rag_core.config import FACULTY_SOURCE_NAME
from rag_core.chain import NOT_COVERED_MESSAGE, format_context, retrieve
from rag_core.insights import topic_title
from rag_core.learning_log import SCOPE_PERSONAL, create_quiz
from rag_core.llm import get_json_llm
from rag_core.normalize import chunk_sentences

_SYSTEM = """You write multiple-choice questions for a university course, strictly from the \
course-note context below.

Write exactly {n} questions about: {topic}
{focus_note}
{level_note}

Rules:
- Each question must be answerable from the context alone. Do not use outside knowledge.
- Exactly 4 options and exactly one correct. Wrong options must be plausible but clearly wrong \
according to the context.
- Every question must test a DIFFERENT fact or concept. Never repeat or rephrase a question, and never \
reuse the same set of options. Spread the questions across different chunks.
- Mix the question types: "What is X?", "What does X stand for?", "Which statement is true about X?", \
"Which of these is NOT ...?", "Which topic does this describe?". Do not write only fill-in-the-blank questions.
- Each question must be a specific, complete question about one idea (for example "What is X?", \
"Which of these is a function of X?"). Never ask "which statement is supported by the material".
- Wrong options must be plausible statements about the same subject that the context does not support. \
Do not include the quiz topic title or slide headings as an option.
- "chunk" is the number N of the [Chunk N] the question is based on.
- "explanation" is one sentence saying why the answer is correct, using the context.

Return ONLY JSON in exactly this shape:
{{"questions": [{{"question": "...", "options": ["...", "...", "...", "..."], "answer_index": 0, "chunk": 1, "explanation": "..."}}]}}

--- CONTEXT ---
{context}
"""

_PROMPT = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", "Write the quiz.")])

# Difficulty is the writer's INTENT, set by the kind of question and how close the wrong options are. It is not a
# measured property: whether a question really was hard shows only after students answer it (the professor's
# Question analysis reports that).
DIFFICULTIES = ("easy", "medium", "hard")
DEFAULT_DIFFICULTY = "medium"
MIN_QUESTIONS, MAX_QUESTIONS = 1, 15
BATCH = 5                    # questions asked of the local model per call; its reply is capped at about this many

LEVEL_NOTES = {
    "easy": (
        "Difficulty: EASY. Ask direct recall questions about one fact, such as \"What is X?\" or \"What does X "
        "stand for?\". The wrong options must be clearly different from the right one."
    ),
    "medium": (
        "Difficulty: MEDIUM. Mix recall questions with questions that need understanding of how or why. The wrong "
        "options should be plausible but still distinguishable."
    ),
    "hard": (
        "Difficulty: HARD. At least half of the questions must be reasoning questions of these kinds: \"Why ...?\", "
        "\"How does X differ from Y?\", \"Which of these is NOT ...?\" or \"What would happen if ...?\". Do not write "
        "plain \"What is X?\" questions. Every wrong option must be plausible: similar in length and wording to the "
        "right one and based on related ideas in the context."
    ),
}


KEY_MIN_SUPPORT = 0.5        # share of the keyed answer's content words that must appear in the slide it cites
NOT_OTHERS_SUPPORT = 0.6     # in a "which is NOT ..." question, the three wrong options must be backed this much
_NOT_QUESTION = re.compile(r"\bNOT\b|\bEXCEPT\b")


def key_is_backed(question: Dict[str, Any], chunk_text: str) -> bool:
    """
    Is the keyed answer backed by the slide the question cites? A small model sometimes keys an answer the slide never
    says, and the harder the question, the more often. Checked by word overlap, so it is a screen, not a proof.

    A normal question is kept if most content words of the keyed option appear in the cited text. A "which is NOT ..."
    question is the reverse: the keyed option must be the one the text does NOT back, and the other three must be backed
    (a real case: a "NOT a benefit" question whose four options were all listed as benefits on the slide).
    """
    stems = _stems(chunk_text)
    scores = [support(option, stems) for option in question["options"]]
    keyed = question["answer_index"]
    if _NOT_QUESTION.search(question["question"]):
        others = [s for i, s in enumerate(scores) if i != keyed]
        return scores[keyed] < KEY_MIN_SUPPORT and sum(others) / len(others) >= NOT_OTHERS_SUPPORT
    return scores[keyed] >= KEY_MIN_SUPPORT


def normalise_level(level: Optional[str]) -> str:
    """A valid difficulty from whatever was passed in (anything unknown becomes the default)."""
    level = (level or "").strip().lower()
    return level if level in DIFFICULTIES else DEFAULT_DIFFICULTY


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
    seen_questions = set()
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
        if question.strip().lower() in seen_questions:
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
        seen_questions.add(question.strip().lower())
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


_DEFINITION = re.compile(
    r"^(?P<term>[A-Z][A-Za-z0-9&/\-]*(?: [A-Za-z0-9&/\-]+){0,4}?) "
    r"(?P<verb>is|are|refers to|means) (?P<rest>.{25,})$"
)
_NOUN_PHRASE = re.compile(r"^(?:an?|the|one|two|three|any|all|every|each)\b", re.I)  # a full description, not a fragment
_NOT_A_TERM = frozenset("it this that these those there here which what however therefore thus".split())
_BLANK_SKIP = frozenset(
    "multimedia between because through during various several however without within another "
    "different important information application applications across while which where their these those "
    "other about after before being would could should there every still using example examples".split()
)


def _clip_clause(text: str, limit: int = 150) -> str:
    """`text` if short enough, else its leading complete clause ("" if it cannot be cut cleanly)."""
    text = text.strip().rstrip(".")
    if len(text) <= limit:
        return text
    cut = max(text.rfind(token, 0, limit) for token in ("; ", ", which ", ", while ", " — "))
    return text[:cut].rstrip(" ,;—-") if cut >= limit * 0.5 else ""


def _definitions(chunks) -> List[Dict[str, Any]]:
    """'X is/are/refers to/means ...' sentences, one list entry each, with the chunk they came from."""
    found, seen = [], set()
    for chunk_index, doc in enumerate(chunks, start=1):
        for sentence in chunk_sentences(doc, max_len=260):
            match = _DEFINITION.match(sentence)
            if not match:
                continue
            raw_term = match.group("term")
            term = re.sub(r"^(?:An?|The) ", lambda m: m.group(0).lower(), raw_term).strip()
            rest = _clip_clause(match.group("rest"))
            key = re.sub(r"^(?:an?|the) ", "", term.lower())
            if (not term or key.split()[0] in _NOT_A_TERM or key in seen or "(" in term
                    or len(rest) < 25 or not _NOUN_PHRASE.match(rest)):
                continue
            seen.add(key)
            found.append({"chunk": chunk_index, "term": term, "verb": match.group("verb"),
                          "rest": rest, "sentence": sentence})
    return found


def _overlap(a: str, b: str) -> float:
    """Share of content words (4+ letters) that two texts have in common, 0 to 1."""
    wa, wb = set(re.findall(r"[a-z]{4,}", a.lower())), set(re.findall(r"[a-z]{4,}", b.lower()))
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0


def _definition_question(item, others, rng, close: bool = False) -> Optional[Dict[str, Any]]:
    wrong = [o for o in others if o["term"] != item["term"] and o["rest"] != item["rest"]]
    if len(wrong) < 3:
        return None
    if close:
        # Hard: wrong options that share the most wording with the right one are the hardest to rule out.
        wrong.sort(key=lambda o: (-_overlap(o["rest"], item["rest"]), abs(len(o["rest"]) - len(item["rest"])), rng.random()))
    else:
        # Prefer wrong options of similar length so the right one is not given away by its size.
        wrong.sort(key=lambda o: abs(len(o["rest"]) - len(item["rest"])) + rng.random() * 40)
    verb = item["verb"]
    stem = {"is": f"What is {item['term']}?", "are": f"What are {item['term']}?",
            "refers to": f"What does {item['term']} refer to?", "means": f"What does {item['term']} mean?"}[verb]
    options = [item["rest"]] + [o["rest"] for o in wrong[:3]]
    options = [o[:1].upper() + o[1:] + "." for o in options]
    return {"question": stem, "options": options, "answer_index": 0, "chunk": item["chunk"],
            "explanation": item["sentence"]}


def _cloze_question(chunk_index, doc, frequency, rng) -> Optional[Dict[str, Any]]:
    """Blank out a key course term (one used several times in the material) from a sentence."""
    for sentence in chunk_sentences(doc, min_len=60, max_len=200):
        if sentence.startswith(("Example", "For example")):
            continue
        words = re.findall(r"[A-Za-z]{5,}", sentence)
        candidates = [
            w for w in words
            if frequency.get(w.lower(), 0) >= 3 and words.count(w) == 1
            and not sentence.startswith(w) and not re.search(rf"(?:-{w}|{w}-)", sentence)
        ]
        if not candidates:
            continue
        target = max(candidates, key=lambda w: frequency[w.lower()])
        in_sentence = {w.lower() for w in words}
        pool = [w for w in frequency if frequency[w] >= 3 and w not in in_sentence]
        pool.sort(key=lambda w: (abs(len(w) - len(target)) + rng.random() * 4, w))
        if len(pool) < 3:
            continue
        shape = str.capitalize if target[0].isupper() else str.lower
        blanked = re.sub(rf"\b{re.escape(target)}\b", "_____", sentence, count=1)
        return {"question": f"Fill in the blank: {blanked}", "options": [target, *(shape(w) for w in pool[:3])],
                "answer_index": 0, "chunk": chunk_index, "explanation": sentence}
    return None


def fallback_questions_from_chunks(
    chunks, n_questions: int, rng: Optional[random.Random] = None, avoid=(),
    priority: int = 0, n_priority: int = 3, close: bool = False,
) -> List[Dict[str, Any]]:
    """
    Source-grounded questions written without the LLM: "What is X?" from definitions in the
    notes, and fill-in-the-blank from other sentences. One question per chunk, so no two
    questions repeat. If `priority` is given, the first `priority` chunks are the focus topic:
    up to `n_priority` questions come from them and the rest from the other chunks.
    """
    rng = rng or random.Random()
    avoid = set(avoid)
    definitions = _definitions(chunks)
    frequency: Dict[str, int] = {}
    for doc in chunks:
        for word in re.findall(r"[A-Za-z]{5,}", doc.page_content.lower()):
            if word not in _BLANK_SKIP and not word.endswith(("ly", "ing")):
                frequency[word] = frequency.get(word, 0) + 1
    by_chunk: Dict[int, Dict[str, Any]] = {}
    for item in definitions:
        by_chunk.setdefault(item["chunk"], item)

    def ordered(indexes):  # definition questions first, then fill-in-the-blank, each in random order
        with_def = [i for i in indexes if i in by_chunk]
        rest = [i for i in indexes if i not in by_chunk]
        rng.shuffle(with_def)
        rng.shuffle(rest)
        return with_def + rest

    focus = ordered(range(1, priority + 1))
    others = ordered(range(priority + 1, len(chunks) + 1))
    questions: List[Dict[str, Any]] = []
    used: set = set()

    def add_from(order, limit):
        for chunk_index in order:
            if len(questions) >= limit:
                return
            if chunk_index in used:
                continue
            used.add(chunk_index)
            question = None
            if chunk_index in by_chunk:
                question = _definition_question(by_chunk[chunk_index], definitions, rng, close)
            if question is None:
                question = _cloze_question(chunk_index, chunks[chunk_index - 1], frequency, rng)
            if question and question["question"] not in {q["question"] for q in questions} | avoid:
                questions.append(question)

    add_from(focus, n_priority if priority else n_questions)
    add_from(others, n_questions)
    add_from(focus, n_questions)  # still short: more from the focus topic
    return questions


_ABBREVIATION = re.compile(
    r"\b([A-Z][A-Za-z]{1,7})\s*=\s*([A-Z][A-Za-z\-]+(?: (?:of |and |for )?[A-Z][A-Za-z\-]+){1,6})"
)
_SENTENCE_STARTERS = frozenset("This The It These That Your Here Now Remember In A An Its They Think Memory".split())


def _abbreviation_questions(chunks, rng, count: int = 4) -> List[Dict[str, Any]]:
    """'What does CMIP stand for?' from lines such as 'CMIP = Common Management Information Protocol'."""
    found: Dict[str, tuple] = {}
    for index, doc in enumerate(chunks, start=1):
        for m in _ABBREVIATION.finditer(doc.page_content):
            words = m.group(2).split()
            for cut, word in enumerate(words):
                if cut >= 1 and word in _SENTENCE_STARTERS:  # the next sentence began: "Base The MIB ..."
                    words = words[:cut]
                    break
            if len(words) >= 2 and m.group(1)[0].upper() == words[0][0]:
                found.setdefault(m.group(1), (" ".join(words), index))
    if len(found) < 4:
        return []
    items = list(found.items())
    rng.shuffle(items)
    out = []
    for abbr, (expansion, index) in items[:count]:
        wrong = [e for a, (e, _) in found.items() if a != abbr and e != expansion]
        rng.shuffle(wrong)
        out.append({
            "question": f"What does {abbr} stand for?", "options": [expansion, *wrong[:3]], "answer_index": 0,
            "chunk": index, "explanation": f"{abbr} = {expansion}.", "kind": "abbr",
        })
    return out


_GLUED_CAPITALS = re.compile(r"[A-Z]{2,}[A-Z][a-z]")       # "MODELAnnual", "XXPresentation": two headings run together
_GLUED_NUMBER = re.compile(r"[A-Za-z]\d{2,}")                # "title108": a page number run onto a word


def _garbled_heading(heading: str) -> bool:
    """A slide title that PDF extraction ran together with its neighbours or a page number: not fit to use as an answer option."""
    return bool(_GLUED_CAPITALS.search(heading) or _GLUED_NUMBER.search(heading))


def _heading_questions(headings, rng, close: bool = False) -> List[Dict[str, Any]]:
    """
    Questions that use the material's headings: pick the true statement, or name the topic of a statement.
    With `close`, the wrong statements are the ones most like the right one (from the most related other
    slides), which makes them harder to rule out. They are still taken from OTHER headings, so none is true of this one.
    """
    if len(headings) < 4:
        return []

    def words(text):
        return {w for w in re.findall(r"[a-z]{4,}", text.lower())}

    usable = [h for h in headings if len(h["heading"]) <= 60 and not _garbled_heading(h["heading"])]
    if len(usable) < 4:
        return []
    out = []
    for h in usable:
        good = [x for x in h["sentences"] if 50 <= len(x) <= 170 and not x.startswith(("Example", "For example"))]
        if not good:
            continue
        correct = rng.choice(good)
        source = {"file": h["file"], "page": h["pages"][0], "slide": h["heading"]}
        title = h["heading"].rstrip("?").strip()
        pool = [
            (o, x) for o in usable if o is not h for x in o["sentences"]
            if 50 <= len(x) <= 170 and not x.startswith(("Example", "For example")) and not (words(title) & words(x))
        ]
        rng.shuffle(pool)
        if close:
            pool.sort(key=lambda item: -_overlap(item[1], correct))   # stable: ties keep their random order
        wrong, seen_headings = [], set()
        for o, x in pool:
            if o["heading"] not in seen_headings and x != correct:
                wrong.append(x)
                seen_headings.add(o["heading"])
            if len(wrong) == 3:
                break
        if len(wrong) == 3:
            stem = h["heading"] if h["heading"].endswith("?") else f"Which statement is true about {title}?"
            out.append({
                "question": stem, "options": [correct, *wrong], "answer_index": 0, "chunk": 1,
                "explanation": correct, "source": source, "kind": "statement", "_heading": h["heading"],
            })
        if not (words(title) & words(correct)):  # the statement must not give the topic away
            other_titles = [o["heading"].rstrip("?") for o in usable if o is not h and o["heading"].rstrip("?") != title]
            rng.shuffle(other_titles)
            if len(other_titles) >= 3:
                out.append({
                    "question": f"Which topic does this statement belong to? \u201c{correct}\u201d",
                    "options": [title, *other_titles[:3]], "answer_index": 0, "chunk": 1,
                    "explanation": f"This comes from the section on {title}.", "source": source,
                    "kind": "topic", "_heading": h["heading"],
                })
    rng.shuffle(out)
    return out


# Which question types a quiz leans on, by difficulty, when it is written without the model. Recognition questions
# (what is X, what does X stand for, which topic) come first for Easy; true-statement questions, which need the
# student to rule out look-alike statements, come first for Hard, with the most confusable wrong options.
_KINDS_BY_LEVEL = {
    "easy": ["def", "abbr", "topic"],
    "medium": ["def", "abbr", "statement", "topic"],
    "hard": ["statement", "def", "abbr"],
}
_ALL_KINDS = ["def", "abbr", "statement", "topic"]


def mixed_fallback_quiz(
    chunks, headings, n_questions: int, rng: random.Random, avoid=(), focus: Optional[str] = None,
    priority: int = 0, n_focus: int = 3, level: str = DEFAULT_DIFFICULTY,
) -> List[Dict[str, Any]]:
    """
    A quiz written without the LLM that mixes question types: definitions, abbreviations, true-statement
    and which-topic questions, with fill-in-the-blank only to fill gaps. With `focus`, up to `n_focus`
    questions come from that topic first. `level` shifts the mix and, for "hard", the wrong options.
    """
    avoid = set(avoid)
    level = normalise_level(level)
    close = level == "hard"
    base = fallback_questions_from_chunks(
        chunks, n_questions * 3, rng, avoid, priority, n_priority=n_focus * 2, close=close
    )
    pools: Dict[str, List[Dict[str, Any]]] = {
        "def": [q for q in base if not q["question"].startswith("Fill in the blank")],
        "abbr": _abbreviation_questions(chunks, rng),
        "statement": [], "topic": [],
        "cloze": [q for q in base if q["question"].startswith("Fill in the blank")],
    }
    for q in _heading_questions(headings, rng, close=close):
        pools[q["kind"]].append(q)
    for kind, pool in pools.items():
        pool[:] = [q for q in pool if q["question"] not in avoid]

    def on_focus(q) -> bool:
        if not focus:
            return False
        if q.get("_heading"):
            return q["_heading"].lower() == focus.lower()
        return q["chunk"] <= priority

    for pool in pools.values():
        pool.sort(key=lambda q: not on_focus(q))  # stable: focus questions first, the rest keep their random order

    kinds = list(_KINDS_BY_LEVEL[level])
    rng.shuffle(kinds)
    picked: List[Dict[str, Any]] = []

    def take(kinds_to_use, only_focus: bool, limit: int) -> None:
        progress = True
        while progress and len(picked) < limit:
            progress = False
            for kind in kinds_to_use:
                pool = pools[kind]
                if pool and (not only_focus or on_focus(pool[0])) and len(picked) < limit:
                    picked.append(pool.pop(0))
                    progress = True

    if focus:
        take(kinds, True, n_focus)
    take(kinds, False, n_questions)
    take([k for k in _ALL_KINDS if k not in kinds], False, n_questions)  # the other types, if these ran out
    take(["cloze"], False, n_questions)  # only if the other types could not fill the quiz
    rng.shuffle(picked)
    return picked


def _source_of(doc) -> Dict[str, Any]:
    """Where a question came from, so a student or professor can check it against the slide."""
    return {
        "file": doc.metadata.get("source", "unknown").replace("\\", "/").split("/")[-1],
        "page": (doc.metadata.get("page") or 0) + 1,
        "slide": topic_title(doc.metadata),
    }


def _sample_in_order(chunks: list, count: int, rng: random.Random) -> list:
    """`count` randomly chosen chunks (all of them if there are fewer), kept in document order."""
    if len(chunks) <= count:
        return chunks
    return [chunks[i] for i in sorted(rng.sample(range(len(chunks)), count))]


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
        variables = {"n": n_questions, "topic": topic, "context": format_context(chunks), "focus_note": "", "level_note": ""}

        questions: List[Dict[str, Any]] = []
        try:
            for _attempt in range(2):  # one retry: small models fail JSON now and then
                questions = parse_quiz_json(chain.invoke(variables), n_chunks=len(chunks))
                if questions:
                    break
        except Exception:
            questions = fallback_questions_from_chunks(chunks, n_questions)
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

    def run_from_material(
        self,
        ctx: AgentContext,
        n_questions: int = 5,
        scope: str = SCOPE_PERSONAL,
        topic: str = "Selected material",
        avoid=(),
        focus: Optional[str] = None,
        difficulty: str = DEFAULT_DIFFICULTY,
        **_,
    ) -> AgentResult:
        """Write a quiz directly from the selected document chunks, not a named topic.

        `n_questions` (1 to 15) is how many to write and `difficulty` (easy, medium or hard) is the writer's
        intent: the model is told what kind of questions to ask, and the rule-based fallback leans on easier or
        harder question types. A request for more than the local model writes in one reply is made in batches
        of BATCH, each from a fresh sample of the material. Chunks are sampled at random, and any question text
        in `avoid` (e.g. from the previous quiz) is not repeated, so every call gives a different quiz.
        """
        level = normalise_level(difficulty)
        n_questions = max(MIN_QUESTIONS, min(int(n_questions), MAX_QUESTIONS))
        selected = set(ctx.source_filenames or [])
        chunks = []
        for doc in ctx.vectorstore.docstore._dict.values():
            source = Path(doc.metadata.get("source", "")).name
            if source == FACULTY_SOURCE_NAME:
                continue
            if selected and source not in selected:
                continue
            chunks.append(doc)
        chunks.sort(key=lambda d: (Path(d.metadata.get("source", "")).name, d.metadata.get("page") or 0))
        chunks = [d for d in chunks if len(" ".join(d.page_content.split())) >= 150]
        all_chunks = chunks
        rng = random.Random()
        focus_chunks: list = []
        if focus:  # a quiz on one topic: mainly its own chunks, a couple of questions from elsewhere
            hit = retrieve(focus, ctx.vectorstore, k=4, source_filenames=ctx.source_filenames)
            focus_chunks = [d for d in hit.chunks if any(d is c for c in all_chunks)] if hit.relevant else []
            topic = focus
        others = [c for c in all_chunks if not any(c is f for f in focus_chunks)]

        if not (focus_chunks or others):
            return AgentResult(
                agent=self.name, kind="quiz", text=NOT_COVERED_MESSAGE, grounded=False,
                data={"error": "not_covered"},
            )

        chain = _PROMPT | (ctx.llm if ctx.llm is not None else get_json_llm()) | StrOutputParser()
        known = set(avoid)
        questions: List[Dict[str, Any]] = []
        from_model = 0
        discarded = 0                                               # model questions whose keyed answer the slide does not back
        empty_replies = 0
        for _call in range(-(-n_questions // BATCH) + 1):          # one spare call: models fail JSON now and then
            need = n_questions - len(questions)
            if need <= 0 or empty_replies >= 2:
                break
            take = min(BATCH, need)
            batch = focus_chunks + _sample_in_order(others, max(take + 3 - len(focus_chunks), 0), rng)
            variables = {
                "n": take, "topic": topic, "context": format_context(batch),
                "focus_note": "At least 3 of the questions must be about this topic; the others may cover other parts of the context."
                if focus_chunks else "",
                "level_note": LEVEL_NOTES[level],
            }
            try:
                parsed = parse_quiz_json(chain.invoke(variables), n_chunks=len(batch))
            except Exception:
                break                                               # the model is offline: stop asking it
            new = [q for q in parsed if q["question"] not in known]
            fresh = [q for q in new if key_is_backed(q, batch[q["chunk"] - 1].page_content)]
            discarded += len(new) - len(fresh)
            known.update(q["question"] for q in new)                # a rejected question is not retried or counted twice
            if not fresh:
                empty_replies += 1
                continue
            for q in fresh[:take]:
                q["source"] = _source_of(batch[q["chunk"] - 1])
                questions.append(q)
                from_model += 1

        if len(questions) < n_questions:                            # the model is offline or fell short: top up
            pool = focus_chunks + others
            extra = mixed_fallback_quiz(
                pool, major_headings(ctx), n_questions - len(questions), rng, known, focus=focus,
                priority=len(focus_chunks), level=level,
            )
            for q in extra:
                if not q.get("source"):
                    q["source"] = _source_of(pool[q["chunk"] - 1])
                questions.append(q)
        if not questions:
            return AgentResult(
                agent=self.name, kind="quiz", grounded=False,
                text="The model did not produce a usable quiz. Try again.",
                data={"error": "invalid_output"},
            )

        saved = []
        for question in questions[:n_questions]:
            question = shuffle_options({k: v for k, v in question.items() if not k.startswith("_")}, rng)
            question.pop("kind", None)
            question["level"] = level
            saved.append(question)

        quiz_id = create_quiz(
            topic=topic,
            prompt=(f"Quiz on {topic}" if focus_chunks else f"Quiz from {topic}") + f" ({level})",
            questions=saved, scope=scope, owner=ctx.student_id, db_dir=ctx.db_dir,
        )
        return AgentResult(
            agent=self.name,
            kind="quiz",
            text=f"{len(saved)}-question {level} quiz from {topic}.",
            sources=focus_chunks + others,
            data={
                "quiz_id": quiz_id, "topic": topic, "questions": saved, "scope": scope,
                "difficulty": level, "requested": n_questions, "written": len(saved), "from_model": from_model,
                "discarded": discarded,
            },
        )
