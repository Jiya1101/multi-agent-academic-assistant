"""
Conversation memory for the Ask section: follow-up questions.

Until now every message was answered on its own. "What about its disadvantages?" was searched for as it stands,
matched nothing, and got "not covered", because "its" means nothing without the question before it. This module
lets a follow-up lean on the conversation, in a way that stays visible and cheap:

  1. Remember each answered question as a `Turn` (what was asked, what it was understood as, its slide topic).
  2. Decide whether a new message is a follow-up: it opens with a continuation word ("and", "why", "what about",
     "give an example") or points back with "it / this / that / they", AND it does not stand on its own. A message
     stands on its own when searching the notes with it alone already finds a strong match, so "Why is the ring
     algorithm slower?" is left exactly as typed.
  3. Rewrite a follow-up as a standalone question using the previous one ("What are the disadvantages of the
     ring-based algorithm?"). The local model does this when it can; otherwise the topic of the previous answer is
     joined onto the message. Either way the student is shown "Understood as: ...", so nothing is hidden.

The rewritten text is what is answered, routed and logged, so the professor's dashboard groups it by meaning
instead of seeing a meaningless "what about it?". What the student typed is kept next to it.

What this is not: the assistant does not carry a long memory. Only the LAST answered question is used to resolve a
follow-up, and a follow-up to a question the notes could not answer is not rewritten (there is nothing to refer to).
"""

import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agents.base import AgentContext
from rag_core.chain import retrieve
from rag_core.insights import topic_title
from rag_core.llm import get_rewrite_llm

# A message whose own best match is this close (lower = closer) needs no rewriting. Chosen from a small real test
# (8 follow-ups, 6 stand-alone questions on the course index): stand-alone questions scored up to 0.86 and follow-ups
# 1.15 or more, so 1.0 sits in the gap. At 0.85 a stand-alone question (0.86) was rewritten with the previous topic
# wrongly attached, which is the worst failure here. Re-check it on other material.
STANDALONE_SCORE = 1.0
MAX_FOLLOW_UP_WORDS = 18      # a longer message is a question in its own right
MAX_HISTORY = 8               # turns kept in a session

# Words a message starts with when it continues the conversation.
_CONTINUATION = re.compile(
    r"^(?:and|also|but|so|then|what about|how about|what if|why|how come|how|can you|could you|please|give me|"
    r"give an|show me|tell me more|more|another|elaborate|explain (?:that|this|it|more|again|further)|simpler|"
    r"in simple|an example|example|for example|which one|is that|does that|do they|are they|is it|is this|"
    r"is there|what else|anything else)\b",
    re.I,
)
_POINTS_BACK = re.compile(
    r"\b(?:it|its|this|that|these|those|they|them|their|he|she|his|her"
    r"|(?:the )?(?:first|second|third|last|former|latter|other|same) (?:one|thing|point|step|item|part|method|type))\b",
    re.I,
)
_COMMON = frozenset(
    "what when where which while about after again also because before being between could does from have into "
    "just like make more most much only other over really should some such than that their them then there these "
    "they this those through very were what when where which will with would your".split()
)

_SYSTEM = """You rewrite a student's follow-up question so it can be understood on its own.

Use the earlier question and its topic to fill in what words like "it", "this", "that", "they" or a missing subject refer to.
Keep the student's meaning and wording as far as you can. Do NOT answer the question. Do NOT add anything new.
Reply with the rewritten question only, on one line."""

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM),
        ("human", "Earlier question: {previous}\nTopic of the earlier answer: {topic}\nFollow-up: {followup}"),
    ]
)


@dataclass(frozen=True)
class Turn:
    """One answered question in the conversation."""

    question: str                    # what the student typed
    standalone: str                  # what it was understood as (the same text when it needed no rewriting)
    topic: Optional[str]             # the slide topic the answer came from
    grounded: bool                   # False when the notes could not answer it
    log_id: Optional[int] = None     # its row in the question log
    file: Optional[str] = None       # the study file the answer came from


@dataclass(frozen=True)
class FollowUp:
    text: str                        # the question to answer
    how: str = "none"                # "none" (as typed), "model" (rewritten by the model) or "joined" (topic added)
    previous: Optional[Turn] = None  # the turn it follows, when how != "none"


def last_answered_turn(history: Sequence[Any]) -> Optional[Turn]:
    """The most recent turn a follow-up could refer to: one the notes actually answered."""
    for turn in reversed(list(history or ())):
        if getattr(turn, "grounded", False) and (turn.topic or turn.standalone):
            return turn
    return None


def looks_like_follow_up(message: str) -> bool:
    """Whether a message reads as a continuation: it opens with a continuation word or points back with a pronoun."""
    words = re.findall(r"[A-Za-z']+", message or "")
    if not words or len(words) > MAX_FOLLOW_UP_WORDS:
        return False
    return bool(_CONTINUATION.match(message.strip()) or _POINTS_BACK.search(message))


def _content_stems(text: str) -> set:
    return {w[:5] for w in re.findall(r"[a-z]{4,}", (text or "").lower()) if w not in _COMMON}


def _stands_alone(message: str, ctx: AgentContext) -> bool:
    """True if searching the notes with the message alone already finds a strong match."""
    if ctx.vectorstore is None:
        return False
    found = retrieve(message, ctx.vectorstore, k=1, source_filenames=ctx.source_filenames)
    return found.relevant and found.via == "embedding" and found.top_score is not None and found.top_score <= STANDALONE_SCORE


def _join(message: str, previous: Turn) -> str:
    """The always-available rewrite: the message with the previous topic named at the end."""
    anchor = previous.topic or previous.standalone
    return f"{message.strip().rstrip('?.! ')} (about {anchor})?"


def _clean(raw: str) -> str:
    line = next((l.strip() for l in (raw or "").splitlines() if l.strip()), "")
    line = re.sub(r"^(?:rewritten|standalone)?\s*(?:question|follow-up)\s*:\s*", "", line, flags=re.I)
    return line.strip(" \"'")


def _plausible(rewritten: str, previous: Turn, message: str) -> bool:
    """A rewrite is accepted only if it is a short question that still touches the earlier topic."""
    words = rewritten.split()
    if not 3 <= len(words) <= 40 or rewritten.lower().strip("?. ") == message.lower().strip("?. "):
        return False
    return bool(_content_stems(rewritten) & _content_stems(f"{previous.standalone} {previous.topic or ''}"))


def rewrite_follow_up(message: str, previous: Turn, llm: Any = None) -> Tuple[str, str]:
    """(standalone question, how). The model rewrites it if it can; otherwise the topic is joined onto the message."""
    try:
        chain = _PROMPT | (llm if llm is not None else get_rewrite_llm()) | StrOutputParser()
        rewritten = _clean(chain.invoke({
            "previous": previous.standalone, "topic": previous.topic or previous.standalone, "followup": message,
        }))
    except Exception:
        return _join(message, previous), "joined"
    if _plausible(rewritten, previous, message):
        return rewritten if rewritten.endswith("?") else rewritten.rstrip(".") + "?", "model"
    return _join(message, previous), "joined"


def resolve_follow_up(message: str, ctx: AgentContext) -> FollowUp:
    """What to actually answer: the message as typed, or a standalone rewrite of a follow-up."""
    previous = last_answered_turn(ctx.history)
    if previous is None or not looks_like_follow_up(message) or _stands_alone(message, ctx):
        return FollowUp(message)
    text, how = rewrite_follow_up(message, previous, ctx.llm)
    return FollowUp(text, how, previous)


def turn_from(response: Any) -> Optional[Turn]:
    """
    The `Turn` to remember for an Orchestrator response, or None when it was not a question and answer (a quiz,
    notes or progress request has nothing a follow-up could refer to).
    """
    shown = next((r for r in response.results if r.kind in ("answer", "explanation")), None)
    if shown is None:
        return None
    first = shown.sources[0].metadata if shown.sources else None
    return Turn(
        question=response.message,
        standalone=shown.data.get("question") or response.message,
        topic=topic_title(first) if first else None,
        grounded=bool(shown.grounded),
        log_id=shown.data.get("log_id"),
        file=(str(first.get("source", "")).replace("\\", "/").split("/")[-1] or None) if first else None,
    )


def remember(history: Sequence[Any], turn: Optional[Turn]) -> list:
    """`history` with `turn` added, keeping only the most recent MAX_HISTORY turns."""
    kept = list(history or ())
    if turn is not None:
        kept.append(turn)
    return kept[-MAX_HISTORY:]
