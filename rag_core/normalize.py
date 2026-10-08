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
    section = human_title(section)
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


def _split_compact(word: str) -> str:
    """'MULTIMEDIASYSTEMS' -> 'Multimedia Systems' (falls back to plain title case)."""
    parts = wordninja.split(word.lower())
    if len(parts) > 1 and all(len(p) >= 3 for p in parts):
        return " ".join(p.capitalize() for p in parts)
    return word.capitalize()


def human_title(text: str) -> str:
    """Clean PDF-looking headings for student-facing display."""
    text = re.sub(r"\s+", " ", text or "").strip()
    # "M U L T I M E D I A S Y S T E M S" or "M U L T I M E D I A S Y S T E M S Introduction"
    spaced = re.match(r"^((?:[A-Z]\s+){3,}[A-Z])\b\s*(.*)$", text)
    if spaced:
        compact = _split_compact(spaced.group(1).replace(" ", ""))
        rest = spaced.group(2).strip()
        text = f"{compact} {rest}".strip()
    text = re.sub(r"\bSilberschatz,\s*Galvin and Gagne\s*©?\d{4}\b", "", text, flags=re.I)
    text = re.sub(r"\bOperating System Concepts\s*[-–]\s*10th Edition\b", "", text, flags=re.I)
    text = re.sub(r"\bChapter\s+\d+[:\s-]*", "", text, flags=re.I)
    text = re.sub(r"^\d+(?:\.\d+)*\s*", "", text).strip(" -–:|.•·")
    text = re.sub(r"^\(\w{1,2}\)\s*", "", text)  # "(b) Communication" -> "Communication"
    text = re.sub(r"\s+", " ", text).strip()
    if re.fullmatch(r"[A-Z]{10,}", text):  # one long run of capitals with the spaces lost
        text = _split_compact(text)
    elif text.isupper() and len(text) > 4:
        # Long words become "Systems"; short ones stay as acronyms ("OSI / CMIP", "OLTP VS OLAP Systems").
        text = " ".join(w.capitalize() if len(w) > 5 and w.isalpha() else w for w in text.split())
    return text or "Selected material"


_FRAGMENT_START = re.compile(r"^[\s.\-–•·o]+(?=[A-Z0-9\"(])")
_ABBREVIATIONS = ((re.compile(r"\be\.\s*g\.\s*", re.I), "e.g. "), (re.compile(r"\bi\.\s*e\.\s*", re.I), "i.e. "))
_CLAUSE_BREAKS = ("; ", ", which ", ", while ", " — ")
_DANGLING_END = frozenset("and or of in the to a an for with as such by on at from that which".split())


def _fit(sentence: str, max_len: int) -> str:
    """The sentence, or a complete clause of it if it is too long ("" when it cannot be cut cleanly)."""
    if len(sentence) <= max_len:
        return sentence
    window = sentence[:max_len]
    for token in _CLAUSE_BREAKS:
        cut = window.rfind(token)
        if cut >= max_len * 0.55:
            return window[:cut].rstrip(" ,;—-") + "."
    return ""


def readable_sentences(
    text: str, min_len: int = 35, max_len: int = 200, drop_prefix: str = ""
) -> list[str]:
    """
    Split flattened slide text into short, complete, student-readable sentences.

    PDF text arrives as one run-on blob (headings glued to body text, table
    cells, fragments left by chunk overlap). Only grammatical-looking sentences
    survive: headings, table-like runs, fragments and anything that cannot be
    shortened to a complete clause are dropped. Never ends in "..." and never
    contains "[Chunk N]" markers.
    """
    text = re.sub(r"\[Chunk[^\]]*\]", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    for pattern, replacement in _ABBREVIATIONS:
        text = pattern.sub(replacement, text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    spaced = re.match(r"^((?:[A-Z]\s+){3,}[A-Z])\b\s*(.*)$", text)
    if spaced:
        text = spaced.group(2)
    drop_prefix = " ".join((drop_prefix or "").split())
    if drop_prefix:
        numberless = re.sub(r"^\d{1,2}[.)]\s*", "", text)  # "12. IEEE Standards ..." -> "IEEE Standards ..."
        if numberless.lower().startswith(drop_prefix.lower()):
            text = numberless[len(drop_prefix):]
        elif text.lower().startswith(drop_prefix.lower()):
            text = text[len(drop_prefix):]

    out: list[str] = []
    seen = set()
    for raw in re.split(r"(?<=[.!?])\s+(?=[A-Z\"(•])", text):
        sentence = _FRAGMENT_START.sub("", raw).strip(" •·-–")
        sentence = re.sub(r"(?:\.{2,}|…)\s*$", "", sentence).strip()
        sentence = re.sub(r"\s+\d{1,2}\.$", "", sentence)          # a list number left behind: "... properties: 1."
        sentence = re.sub(r"^(?:Definition|Explanation)\s+(?=[A-Z])", "", sentence)
        sentence = re.sub(r"^[A-Z][A-Za-z&/() -]{2,45}?\s+Definition\s+(?=[A-Z])", "", sentence)  # "Heading Definition A ..."
        repeated = re.match(r"^([A-Z][A-Za-z&/() -]{2,45}?)\s+(?:(?:A|An|The)\s+)?\1\b", sentence)
        if repeated:  # "Heading A Heading refers to ..." -> "A Heading refers to ..."
            sentence = sentence[len(repeated.group(1)) + 1:]
        if _META_TALK.search(sentence) or _CHAT_OPENER.match(sentence):
            continue  # "your sir has highlighted ...", "Perfect. ..." : study-chat filler, not course content
        if re.search(r"[→↓↑↔│┌└]|::=| = ", sentence):
            continue  # a diagram flattened into text
        if "•" in sentence or sentence.count(" — ") >= 2 or re.search(r"\b(?:e\.g|i\.e)\.$", sentence):
            continue  # a bullet list or table row flattened into one line
        sentence = re.sub(r"^[^.]{0,70}\+[^.]{0,70}\)\s+(?=[A-Z])", "", sentence)  # "A + B (gloss) Real sentence"
        if not sentence or not sentence[0].isupper():
            continue
        if sentence[-1] in ",;:" or sentence.rstrip(".!?").split()[-1:] and sentence.rstrip(".!?").split()[-1].lower() in _DANGLING_END:
            continue  # the chunk was cut mid-sentence
        if sentence[-1] not in ".!?":
            sentence += "."
        sentence = _fit(sentence, max_len)
        words = sentence.split()
        if len(sentence) < min_len or len(words) < 6:
            continue
        capitalised = sum(1 for w in words if w[:1].isupper())
        if capitalised / len(words) > 0.55:  # a heading or a table row, not a sentence
            continue
        key = sentence.lower()[:80]
        if key in seen:
            continue
        seen.add(key)
        out.append(sentence)
    return out


_META_TALK = re.compile(
    r"\b(?:your (?:sir|professor|teacher|textbook|book|notes|highlights?)|highlighted|you pasted|pasted notes|"
    r"I['’]ll|I['’]d|I would|let['’]s|CAT ?\d|your (?:note|table)|you (?:should|must|need|don['’]t need)|"
    r"that['’]s (?:the easiest|all you)|exam hint)\b",
    re.I,
)
_DANGLING = frozenset("for to of and or the a an in with that by on is are as at from which your our their this its".split())
_CHAT_OPENER = re.compile(r"^(?:Perfect|Absolutely|Great|Okay|Exactly|Remember|Memory|Notice|Now|Next|Alright|Nice|Sure|So|Think|Just|Important|Simple|And|But)\b[:,]?")


def is_title_like(section: str, numbered: bool = False) -> bool:
    """True when a stored section tag is a real heading, not a sentence, a diagram label or chat filler."""
    section = (section or "").strip()
    if not 3 <= len(section) <= 60 or section.endswith((",", ".", ";", ":")):
        return False
    if re.search(r"[=+>→↓↑↔│┌└├┐┘%]", section):
        return False  # a formula or diagram line, not a heading
    letters = sum(c.isalpha() for c in section)
    if letters < 3 or letters / len(section) < 0.5:
        return False  # "/ \\", arrows, box drawing
    if not (section[0].isupper() or section[0].isdigit() or section[0] in "(“\""):
        return False  # a sentence cut off mid-way
    if _META_TALK.search(section) or _CHAT_OPENER.match(section):
        return False
    words = section.split()
    if words[-1].lower() in _DANGLING or section.count("?") > 1:
        return False  # a sentence cut off ("Three things are required for") or a list of questions
    if len({w.lower() for w in words}) == 1 and len(words) > 1:
        return False  # "MIB MIB MIB"
    if len(words) == 1 and (len(section) < (2 if numbered else 5) or section.lower() in {"this", "that", "here", "note", "so", "example"}):
        return False  # a numbered heading may be a short acronym: "8. TMN"
    return True


# Bullet glyphs PDF slides use (black circle, white circle, bullet, small square, white bullet, black square,
# triangular bullet), written as code points so this file stays plain ASCII.
_BULLET_GLYPHS = "".join(chr(c) for c in (0x25CF, 0x25CB, 0x2022, 0x25AA, 0x25E6, 0x25A0, 0x2023))
_BULLET_MARK = re.compile(r"\s*[" + _BULLET_GLYPHS + r"]\s*")
_APOSTROPHE = chr(0x2019)
_TRAILING_SUBHEADING = re.compile(
    # "... 2. Failure Detectors", "... c) Ricart (Multicast + Clocks)"
    r"\s+(?:\d{1,2}[.)]|[a-z]\))\s+[A-Z][A-Za-z&/,'" + _APOSTROPHE + r"() +-]{2,80}\.?$"
)
# A pictogram (the emoji and symbol blocks) and the label after it, e.g. a "Quick Revision" tag at the end of a slide.
_TRAILING_ICON_LABEL = re.compile(
    r"\s+[" + chr(0x1F300) + "-" + chr(0x1FAFF) + chr(0x2600) + "-" + chr(0x27BF) + r"].*$"
)


def bullet_points(text: str, drop_prefix: str = "") -> list[str]:
    """
    Slide text written as terse bullets ("Pros: simple, easy.", "Token circulates -> only the holder enters.")
    as a list of short points.

    `readable_sentences` rejects these on purpose (fragments, arrows), so a deck that is all bullets gave the
    oral check and the whole-material notes nothing to work with. This reads them as points instead. Whatever
    comes before the first bullet is only headings and is skipped, and a sub-heading or icon label glued to the
    end of a bullet is removed. Text with no bullet glyphs gives an empty list, so prose decks are unaffected.
    """
    text = " ".join(re.sub(r"\[Chunk[^\]]*\]", " ", text or "").split())
    prefix = " ".join((drop_prefix or "").split())
    if prefix and text.lower().startswith(prefix.lower()):
        text = text[len(prefix):]
    points: list[str] = []
    seen = set()
    for segment in _BULLET_MARK.split(text)[1:]:
        segment = _TRAILING_ICON_LABEL.sub("", segment)
        segment = _TRAILING_SUBHEADING.sub("", segment).strip(" .;:-" + chr(0x2013))
        if len(segment.split()) < 3 or len(segment) < 15:
            continue
        if len(segment) > 240:
            segment = segment[:240].rsplit(" ", 1)[0]
        point = segment[0].upper() + segment[1:] + "."
        if point.lower()[:60] not in seen:
            seen.add(point.lower()[:60])
            points.append(point)
    return points


def chunk_sentences(doc, min_len: int = 35, max_len: int = 200) -> list[str]:
    """`readable_sentences` of one chunk, with its slide heading removed from the front."""
    section = doc.metadata.get("section") or ""
    return readable_sentences(
        doc.page_content, min_len=min_len, max_len=max_len,
        drop_prefix=section if is_title_like(section) else "",
    )


_NUMBERED_HEADING = re.compile(r"^(?:\d{1,2}[.)]|[①②③④⑤⑥⑦⑧⑨])\s*(?=\S)")


def page_heading(raw_page_text: str) -> tuple[str | None, bool]:
    """(heading, is_numbered) for a page: the first line that looks like a title, or (None, False)."""
    first_line = None
    for line in raw_page_text.splitlines():
        line = line.strip()
        if not line or len(line) < 3 or line.isdigit():
            continue
        cleaned = normalize_page_text(line)[:80]
        if first_line is None:
            first_line = cleaned
        numbered = bool(_NUMBERED_HEADING.match(cleaned))
        candidate = _NUMBERED_HEADING.sub("", cleaned).strip()
        if is_title_like(candidate, numbered) and (numbered or len(candidate.split()) >= 2):
            return candidate, numbered
    if first_line and is_title_like(first_line):
        return first_line, False  # a one-word title such as "Electronics"
    return None, False


def extract_section_title(raw_page_text: str) -> str | None:
    """
    The page's heading, or None when it has none (such a page continues the previous topic).

    Slide decks put the title first. Notes written as running text start pages mid-sentence and put
    headings further down ("3. OSI / CMIP"), so sentences, diagram labels and chat filler are skipped.
    Must run on the raw (pre-normalization) text so line boundaries are still intact.
    """
    return page_heading(raw_page_text)[0]
