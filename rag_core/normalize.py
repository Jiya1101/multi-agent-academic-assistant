"""
Text normalization for raw PDF extractions.

PyPDFLoader gives back text exactly as laid out in the PDF, which for
slide decks / lecture notes usually means: bullet glyphs left over from
the original slide, words hyphenated across a line wrap, mid-sentence
line breaks at every wrapped line (not just paragraph breaks), and —
depending on how the PDF was generated — words glued together with no
space at all (e.g. "isMySQL"). Left uncleaned, these artifacts fragment
embeddings and hurt retrieval precision.

This module also derives a best-effort "section" title per page (the
first heading-like line), so retrieved chunks can be attributed back to
a slide/section title, not just a page number.
"""

import re

import wordninja

# Leftover slide-bullet glyphs at the start of a line (o, •, ▪, ‣, ·, ◦, -).
_BULLET_LINE_RE = re.compile(r"^[ \t]*[o•▪‣·∙◦]\s+", re.MULTILINE)

# A word hyphenated across a PDF line wrap, e.g. "back-\npropagation".
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")

# 3+ consecutive newlines -> a single paragraph break.
_MULTI_BLANK_RE = re.compile(r"\n{3,}")

# A single newline (not part of a paragraph break) -> a space, since PDF
# line-wraps are layout artifacts, not sentence boundaries.
_SINGLE_NEWLINE_RE = re.compile(r"(?<!\n)\n(?!\n)")

# Words glued together with no space where extraction dropped it, e.g.
# "isMySQL" -> "is MySQL". Heuristic: lowercase/digit directly followed by
# an uppercase letter. This can occasionally over-split genuine camelCase
# identifiers, but for prose-heavy course notes it fixes far more than it
# breaks.
_GLUED_WORD_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

# Punctuation glued directly onto the next word with no space at all, e.g.
# "categories:Relational" or "SQL)to access". Digits are excluded from the
# lookahead so decimals like "3.14" are left untouched.
_PUNCT_GLUED_RE = re.compile(r"([.,;:)])(?=[A-Za-z])")

# A bullet "o" glued directly onto whatever precedes it with no space at
# all, e.g. "SERVICESoAlmost every..." or "users.oA database engine...".
# Once _GLUED_WORD_RE has split the "o"/"Almost" boundary, the leftover "o"
# sits stranded right after an uppercase letter or a sentence-ending period,
# immediately before a capitalized word; strip it.
_GLUED_BULLET_TAIL_RE = re.compile(r"(?<=[A-Z.])o(?=\s[A-Z])")

# A single token of all-lowercase letters glued together with no spaces at
# all (no case-boundary signal for _GLUED_WORD_RE to catch), e.g.
# "andrecords" or "theperformance". Below this length a "word" is left
# alone, since short glued runs are more likely a real short word than two
# words stuck together.
_GLUED_LOWERCASE_TOKEN_RE = re.compile(r"\b[a-z]{10,}\b")

_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def _split_glued_lowercase(match: re.Match) -> str:
    """Dictionary-segment one long all-lowercase token, e.g. via wordninja."""
    token = match.group(0)
    parts = wordninja.split(token)
    # Only trust the split if it actually broke the token into multiple
    # plausible words; otherwise leave it untouched (protects genuine long
    # single words, and short/unrecognizable fragments from being mangled).
    if len(parts) > 1 and all(len(p) >= 2 for p in parts):
        return " ".join(parts)
    return token


def normalize_page_text(text: str) -> str:
    """Clean one page's raw extracted text before it is chunked/embedded."""
    text = _BULLET_LINE_RE.sub("", text)
    text = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    text = _SINGLE_NEWLINE_RE.sub(" ", text)
    text = _GLUED_WORD_RE.sub(" ", text)
    # Bullet-tail cleanup must run before the punctuation fix below: it
    # relies on "o" still sitting directly against the preceding "." with
    # no space yet inserted.
    text = _GLUED_BULLET_TAIL_RE.sub("", text)
    text = _PUNCT_GLUED_RE.sub(r"\1 ", text)
    text = _GLUED_LOWERCASE_TOKEN_RE.sub(_split_glued_lowercase, text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def short_section_title(section: str, max_len: int = 48) -> str:
    """
    Shorten a stored section tag for display.

    The tag is the page's first line, which for slides often runs the title
    straight into the body text ("AMAZON RDS BENEFITS Read replicas RDS makes
    it easy..."). If it opens with a run of ALL-CAPS words, that run is the
    slide title; otherwise trim to `max_len` at a word boundary.
    """
    run = []
    for word in section.split():
        if word.isupper() or word.isdigit():
            run.append(word)
        else:
            break
    if len(run) >= 2:
        return " ".join(run)
    if len(section) <= max_len:
        return section
    return section[:max_len].rsplit(" ", 1)[0] + "..."


def extract_section_title(raw_page_text: str) -> str | None:
    """
    Return the first substantial line of a page, used as a "section" tag.

    PyPDFLoader returns one Document per PDF page; for slide-style course
    notes that page's first line is almost always the slide title, which
    makes a good citation/attribution tag even without a real markdown
    structure to split on. Must run on the raw (pre-normalization) text so
    line boundaries are still intact.
    """
    for line in raw_page_text.splitlines():
        line = line.strip()
        if not line or len(line) < 3 or line.isdigit():
            continue
        # The line itself can carry the same glued-word/bullet artifacts as
        # the body text, so run it through the same cleanup before trimming.
        return normalize_page_text(line)[:80]
    return None
