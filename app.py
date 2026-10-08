"""
app.py
======
Streamlit front-end for the multi-agent academic assistant.

Student view
    Ask            one box; the Orchestrator routes it to the Doubt Resolver,
                   Concept Explainer, Quiz Generator or Progress Tracker, and
                   hands off to the Gap Handler when the notes cannot answer.
                   An "Agents involved" panel shows which agents ran and why,
                   and a student can flag a wrong route so the router learns.
    Class quizzes  quizzes the professor published; scores feed the Progress
                   Tracker.
    My progress    strong/weak topics from the student's questions and quizzes.

Professor view
    Class Insight  student questions clustered by meaning.
    Pending Gaps   unanswered questions; an answer is added to the index.
    Class Quizzes  Faculty Insight -> Quiz Generator pipeline: quizzes on the
                   topics the class keeps asking about, plus class results.
    Course Material  upload notes and (re)build the index.

Run with:
    streamlit run app.py
"""

import hashlib
import re
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import streamlit as st

from agents import AgentContext, Orchestrator
from agents.routing import ROUTE_LABELS
from rag_core.config import DATA_DIR, DB_DIR, FACULTY_SOURCE_NAME, MIN_COHORT, TOP_K
from rag_core.confusion import DEFAULT_WEIGHTS, SIGNALS, concept_key, load_and_compute, rank_stability
from rag_core.embeddings import get_embeddings
from rag_core.insights import cluster_questions, slide_label, summarize_clusters
from rag_core.item_analysis import (
    FLAG_HELP,
    KEY_SUSPECT_AT,
    SIGNIFICANT_P,
    analyze_quizzes,
    likely_misconceptions,
    questions_to_review,
)
from rag_core.learning_log import (
    SCOPE_CLASS,
    SCOPE_PERSONAL,
    get_quiz,
    list_attempts,
    list_quizzes,
    record_route_feedback,
)
from rag_core.library import delete_material
from rag_core.loader import load_all_pdfs
from rag_core.normalize import human_title, short_section_title
from rag_core.oral_assessment import list_oral_assessments
from rag_core.pdf_export import CHECK_LEGEND, CHECK_MARK, notes_to_pdf, source_line
from rag_core.query_log import (
    STATUS_GAP,
    STATUS_RESOLVED,
    interaction_summary,
    list_queries,
    resolve_gaps,
    resolved_answers,
)
from rag_core.speech import course_vocabulary_hint, get_recognizer
from rag_core.splitter import split_documents
from rag_core.vectorstore import (
    add_faculty_answer,
    build_vectorstore,
    faculty_documents,
    index_version,
    list_indexed_sources,
    list_indexed_topics,
    load_vectorstore,
    save_vectorstore,
    vectorstore_exists,
)

st.set_page_config(page_title="Academic Assistant", page_icon="📖", layout="wide")

_STYLE = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Lora:wght@500;600;700&family=Nunito:wght@400;600;700&display=swap');
  html, body, .stApp, [data-testid="stMarkdownContainer"], label, input, textarea, button,
  [data-testid="stCaptionContainer"], [data-baseweb="select"] { font-family: 'Nunito', 'Segoe UI', sans-serif; }
  .block-container { max-width: 1080px; padding-top: 2.2rem; padding-bottom: 4rem; }
  h1, h2, h3, h4, h5 { font-family: 'Lora', Georgia, serif; color: #3E5B3C; letter-spacing: 0.1px; }
  h1 { font-weight: 700; margin-bottom: 0.1rem; }
  .stAppDeployButton, [data-testid="stMainMenu"], footer { display: none !important; }
  [data-testid="stHeader"] { background: transparent; }

  /* sidebar */
  [data-testid="stSidebar"] { border-right: 1px solid #D3C8AC; }
  [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { font-family: 'Lora', Georgia, serif; color: #4B3A28; }

  /* navigation row */
  [data-testid="stSegmentedControl"] button { border-radius: 999px; font-weight: 600; }

  /* buttons */
  .stButton > button, .stDownloadButton > button {
    border-radius: 999px; font-weight: 600; border: 1px solid #C9BC9B; background: #FBF8F0; color: #4B3A28;
    box-shadow: 0 1px 0 rgba(75, 58, 40, 0.06);
  }
  .stButton > button:hover, .stDownloadButton > button:hover { border-color: #5E7F5C; color: #3E5B3C; }
  .stButton > button[kind="primary"] { background: #5E7F5C; border-color: #5E7F5C; color: #FBF8F0; }
  .stButton > button[kind="primary"]:hover { background: #4C6B4B; color: #FFFFFF; }

  /* cards, boxes and expanders */
  [data-testid="stVerticalBlockBorderWrapper"], [data-testid="stExpander"], [data-testid="stForm"] {
    background: #FBF8F0; border: 1px solid #DDD2B8 !important; border-radius: 16px;
    box-shadow: 0 2px 8px rgba(75, 58, 40, 0.06);
  }
  [data-testid="stMetric"] { background: #FBF8F0; border: 1px solid #DDD2B8; border-radius: 14px; padding: 0.7rem 1rem; }
  [data-testid="stMetricValue"] { color: #3E5B3C; font-family: 'Lora', Georgia, serif; }
  [data-baseweb="select"] > div, [data-testid="stTextInput"] input { background: #FBF8F0; border-radius: 12px; }
  [data-testid="stAlert"] { border-radius: 14px; }
  .stTabs [data-baseweb="tab"] { font-weight: 600; }
  [class*="st-key-del_"] { display: flex; justify-content: flex-end; }
  [class*="st-key-del_"] button {
    border: none; background: transparent; box-shadow: none; color: #9A8A6B; min-height: 0; padding: 0 0.3rem;
    font-size: 1.05rem; font-weight: 400;
  }
  [class*="st-key-del_"] button:hover { color: #A3462F; background: transparent; }
  .muted { color: #7A6A50; font-size: 0.85rem; }
  h5 { margin: 1.1rem 0 0.4rem 0; }
  /* compact quiz form */
  [data-testid="stForm"] { padding: 0.9rem 1.2rem; }
  [data-testid="stForm"] [role="radiogroup"] { gap: 0.05rem; }
  [data-testid="stForm"] [data-testid="stMarkdownContainer"] p { margin-bottom: 0.25rem; }
  [data-testid="stExpander"] summary { padding: 0.45rem 0.9rem; }
  .tagline { color: #7A6A50; margin: 0 0 1.1rem 0; font-size: 1.02rem; }
</style>
"""
st.markdown(_STYLE, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Cached resources and helpers                                               #
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def _load_embeddings():
    return get_embeddings()


@st.cache_resource(show_spinner=False)
def _orchestrator() -> Orchestrator:
    return Orchestrator()


@st.cache_resource(show_spinner=False, max_entries=1)
def _load_index(version: int):
    """Load the FAISS index; `version` changes whenever the index file does."""
    return load_vectorstore(DB_DIR, _load_embeddings())


def _get_vectorstore():
    """Return the current index, or None if it has not been built yet."""
    if not vectorstore_exists(DB_DIR):
        return None
    return _load_index(index_version(DB_DIR))


def _interaction_session_id() -> str:
    """A random token for this browser session. It links a student's follow-up questions together
    (even when anonymous) and is not tied to a person."""
    return st.session_state.setdefault("interaction_session_id", uuid.uuid4().hex[:12])


def _context(vectorstore, student_id=None, sources=None) -> AgentContext:
    return AgentContext(
        vectorstore=vectorstore,
        db_dir=DB_DIR,
        student_id=student_id,
        session_id=_interaction_session_id(),
        source_filenames=sources,
        embeddings=_load_embeddings(),
    )


def _run_ingestion() -> None:
    """Extract -> chunk -> embed -> build FAISS -> persist, from data/ PDFs."""
    with st.spinner("Extracting text from PDFs ..."):
        documents = load_all_pdfs(DATA_DIR)

    if not documents:
        st.warning("No PDFs found in the data folder. Upload one first.")
        return

    with st.spinner(f"Chunking {len(documents)} page(s) ..."):
        chunks = split_documents(documents)

    if not chunks:
        st.error(
            "No extractable text was found in the uploaded PDF(s). "
            "This usually means a PDF is scanned/image-based rather than "
            "text-based (OCR is not part of this module). Try a different file."
        )
        return

    # Professor answers live in the question log, not in the PDFs, so a rebuild
    # must re-add them or they would silently disappear from the index.
    faculty_docs = faculty_documents(resolved_answers(DB_DIR))
    chunks = chunks + faculty_docs

    with st.spinner(f"Embedding {len(chunks)} chunk(s) and building the FAISS index ..."):
        vectorstore = build_vectorstore(chunks, _load_embeddings())
        save_vectorstore(vectorstore, DB_DIR)
        _load_index.clear()

    st.success(
        f"Index built from {len(documents)} page(s) / {len(chunks) - len(faculty_docs)} "
        f"chunk(s) plus {len(faculty_docs)} professor answer(s)."
    )


def _scope_for_selection(selected, all_sources):
    """Return None for all files, otherwise the exact selected source filenames."""
    return None if set(selected) == set(all_sources) else selected


def _material_label(selected) -> str:
    if not selected:
        return "Selected material"
    if len(selected) == 1:
        return human_title(Path(selected[0]).stem)
    return f"{len(selected)} selected documents"


def _clean_display_text(text: str) -> str:
    text = re.sub(r"\[Chunk\s+\d+\]", "", text or "")
    text = re.sub(r"\bChunk\s+\d+\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _bulletize_answer(text: str) -> list[str]:
    text = _clean_display_text(text)
    text = re.sub(
        r"^I found .*?(?:uploaded material|chunks):\s*",
        "",
        text,
        flags=re.I,
    )
    parts = [
        p.strip(" -•")
        for p in re.split(r"(?:\n+|(?<=[.!?])\s+(?=[A-Z]))", text)
        if len(p.strip(" -•")) > 20
    ]
    return parts[:6] or ([text] if text else [])


def _chunk_label(i: int, doc) -> str:
    source = Path(doc.metadata.get("source", "unknown")).name
    if source == FACULTY_SOURCE_NAME:
        return f"Chunk {i} — Faculty answer"
    label = f"Chunk {i} — {source}"
    page = doc.metadata.get("page")
    if page is not None:
        label += f" (page {int(page) + 1})"
    section = doc.metadata.get("section")
    if section:
        label += f' — "{short_section_title(section)}"'
    return label


# --------------------------------------------------------------------------- #
# Rendering of agent results                                                 #
# --------------------------------------------------------------------------- #
def render_trace(trace) -> None:
    with st.expander("Agents involved", expanded=True):
        for line in trace:
            st.text(line)


_STATUS_BADGE = {
    "Strong": "green", "Developing": "yellow", "Needs practice": "orange",
    "Weak": "red", "Not yet tested": "gray",
}


def _request_topic_test(file: str, topic: str) -> None:
    """Button callback: jump to the Quiz tab and write a quiz mainly about `topic`."""
    st.session_state["main_nav"] = "Quiz"
    st.session_state["active_files"] = [file]
    st.session_state["quiz_request"] = {"file": file, "topic": topic}


def _cards(items, render_card, per_row: int = 2) -> None:
    for start in range(0, len(items), per_row):
        columns = st.columns(per_row)
        for column, item in zip(columns, items[start:start + per_row]):
            with column, st.container(border=True):
                render_card(item)


def render_progress(result) -> None:
    """A dashboard with one tab per uploaded material, so a long list of subjects stays navigable."""
    materials = result.data.get("materials", [])
    if not materials:
        st.info(result.text)
        return
    tabs = st.tabs([_nice_name(m["file"]) for m in materials])
    for tab, material in zip(tabs, materials):
        with tab:
            topics, quizzes = material["topics"], material["quizzes"]
            st.markdown(f"**{material['advice']}**")
            counts = Counter(t.status for t in topics)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Topics", len(topics))
            c2.metric("Strong", counts["Strong"])
            c3.metric("Developing", counts["Developing"])
            c4.metric("Needs work", counts["Weak"] + counts["Needs practice"])

            if quizzes:
                st.markdown("##### Quizzes")
                with st.container(border=True):
                    for q in quizzes:
                        name, score, badge = st.columns([3, 1, 1.4], vertical_alignment="center")
                        name.markdown(f"**{q['label']}**" + (f"  \n<span class='muted'>{q['topic']}</span>" if q["topic"] else ""),
                                      unsafe_allow_html=True)
                        score.markdown(f"{q['correct']}/{q['total']}")
                        badge.markdown(f":{_STATUS_BADGE[q['status']]}-badge[{q['status']}]")

            if topics:
                st.markdown("##### Topics")
                for t in topics:
                    with st.expander(f"**{human_title(t.topic)}**  :{_STATUS_BADGE.get(t.status, 'gray')}-badge[{t.status}]"):
                        st.caption(t.reason)
                        if t.quiz_total:
                            st.markdown(f"**Quiz:** {t.quiz_correct}/{t.quiz_total} correct")
                        if t.asked:
                            st.markdown(f"**Asked ({t.questions}):** " + "; ".join(t.asked[-3:]))
                        if t.oral:
                            st.markdown(f"**Oral check:** {t.oral}")
                            if t.oral_question:
                                st.caption(f"Question: {t.oral_question}")
                        if t.status == "Not yet tested":
                            st.button(
                                "Take test", key=f"taketest_{t.file}_{t.topic}",
                                on_click=_request_topic_test, args=(t.file, t.topic),
                            )


def render_quiz(quiz: dict, student_id, key: str, vectorstore) -> None:
    """A quiz form; on submit the Progress Tracker grades and records it."""
    quiz_key = f"{key}_{quiz['id']}"
    result_key = f"quizresult_{quiz_key}"
    graded = st.session_state.get(result_key)

    st.subheader(f"Quiz: {human_title(quiz['topic'])}")
    with st.form(f"quizform_{quiz_key}"):
        picks = []
        for i, q in enumerate(quiz["questions"]):
            st.markdown(f"**{i + 1}. {q['question']}**")
            picks.append(
                st.radio(
                    f"Question {i + 1}",
                    options=list(range(len(q["options"]))),
                    format_func=lambda j, q=q: q["options"][j],
                    index=None,
                    key=f"pick_{quiz_key}_{i}",
                    label_visibility="collapsed",
                    disabled=graded is not None,
                )
            )
        submitted = st.form_submit_button("Submit answers", disabled=graded is not None)

    if submitted:
        ctx = _context(vectorstore, student_id)
        graded = _orchestrator().progress_tracker.record_attempt(quiz, picks, ctx)
        st.session_state[result_key] = graded
        if not student_id or student_id == st.session_state.get("session_student_id"):
            # Taken without a typed Student ID: remember it, so entering one later still shows this score.
            st.session_state.setdefault("unsaved_attempts", []).append((quiz, picks))

    if graded is None:
        return

    st.success(f"Score: {graded['correct']} / {graded['total']}")
    if graded["saved"]:
        st.caption("Saved to your progress.")
    else:
        st.caption("Not saved: enter a Student ID in the sidebar to track your progress.")
    for i, (q, ok) in enumerate(zip(quiz["questions"], graded["per_question"]), start=1):
        src = q.get("source", {})
        where = f"{src.get('file', '')}, page {src.get('page', '?')} ({src.get('slide', '')})"
        verdict = "Correct" if ok else f"Incorrect. Correct answer: {q['options'][q['answer_index']]}"
        with st.expander(f"Question {i}: {verdict}"):
            st.write(q.get("explanation", ""))
            st.caption(f"Source: {where}")


def render_route_feedback(resp, key: str) -> None:
    """Let a student say the router picked the wrong kind of help; it learns from this."""
    with st.expander("Wrong kind of help?"):
        names = list(ROUTE_LABELS)
        choice = st.selectbox(
            "What did you want?",
            options=names,
            index=names.index(resp.route.name),
            format_func=lambda n: ROUTE_LABELS[n],
            key=f"fbsel_{key}",
        )
        if st.button("Send correction", key=f"fbbtn_{key}"):
            if choice == resp.route.name:
                st.info("That is the kind of help you already got.")
            else:
                record_route_feedback(resp.message, resp.route.name, choice, DB_DIR)
                st.success("Thanks. The router will learn from this on your next message.")


def render_notes(result, key: str) -> None:
    """Show generated notes heading by heading, with a PDF download."""
    data = result.data
    sections = data["sections"]
    if not sections:
        st.warning(result.text)
    if sections:
        st.download_button(
            "Download notes as PDF",
            data=notes_to_pdf(sections, data["title"], data["files"]),
            file_name="study_notes.pdf",
            mime="application/pdf",
            key=f"download_{key}",
        )
    for section in sections:
        st.markdown(f"### {human_title(section['heading'])}")
        st.markdown("**Definition**")
        st.write(_clean_display_text(section["definition"]))
        check = section.get("check", {})
        st.markdown("**Key points**")
        for index, point in enumerate(section["key_points"]):
            st.markdown(f"- {_clean_display_text(point)}{CHECK_MARK if index in check.get('key_points', []) else ''}")
        if section.get("use_case") and section["use_case"] != "The notes do not give an example.":
            st.markdown("**Use case**")
            st.write(_clean_display_text(section["use_case"]) + (CHECK_MARK if check.get("use_case") else ""))
        if check.get("key_points") or check.get("use_case"):
            st.caption(CHECK_LEGEND)
        st.caption(source_line(section["sources"]))
    if data["skipped"]:
        st.info("Not in the uploaded material, so skipped: " + ", ".join(data["skipped"]))
    if data["failed"]:
        st.warning("The model gave no usable notes for: " + ", ".join(data["failed"]) + ". Try again.")
    if data["dropped"]:
        st.caption(
            f"{data['dropped']} statement(s) were removed because the material did not support them."
        )


def _library(vectorstore) -> list[str]:
    """The indexed study files (professor answers excluded)."""
    if vectorstore is None:
        return []
    return [s for s in list_indexed_sources(vectorstore) if s != FACULTY_SOURCE_NAME]


def _nice_name(filename: str) -> str:
    return human_title(Path(filename).stem) if filename.lower().endswith(".pdf") else filename


def _toggle_file(name: str) -> None:
    files = list(st.session_state.get("active_files", []))
    st.session_state["active_files"] = [f for f in files if f != name] if name in files else files + [name]


def _material_gate() -> list[str]:
    """The material the student chose once (sidebar or Material page); every section uses it."""
    files = list(st.session_state.get("active_files", []))
    if not files:
        st.info("Choose what you are studying first: pick your material in the sidebar, or on the Material page.")
        return []
    st.caption("Studying: " + ", ".join(_nice_name(f) for f in files))
    return files


def render_notes_builder(vectorstore, student_id, key: str, files=None) -> None:
    """Generate downloadable notes from the sections of the chosen material (`files`, or a picker for professors)."""
    st.write("Short, heading-wise revision notes from your material, ready to download as a PDF.")
    sources = _library(vectorstore)
    chosen_files = files if files is not None else st.multiselect(
        "From file(s)", options=sources, default=[], key=f"nbfiles_{key}"
    )

    if st.button("Generate notes", type="primary", key=f"nbgo_{key}"):
        if not chosen_files:
            st.warning("Choose your material first (sidebar).")
        else:
            bar = st.progress(0.0, text="Starting ...")

            def on_progress(done: int, total: int, topic: str) -> None:
                bar.progress(min(done / max(total, 1), 1.0), text=f"Reading material: {topic}" if topic else "Done")

            scope = _scope_for_selection(chosen_files, sources)
            st.session_state[f"notes_{key}"] = _orchestrator().note_generator.run_from_material(
                _context(vectorstore, student_id, scope), on_progress=on_progress
            )
            bar.empty()

    if f"notes_{key}" in st.session_state:
        render_notes(st.session_state[f"notes_{key}"], key)


def render_response(resp, student_id, key: str, vectorstore, feedback: bool = True, debug: bool = False) -> None:
    if debug:
        render_trace(resp.trace)
    if debug and feedback and resp.message:
        render_route_feedback(resp, key)
    for result in resp.results:
        if result.kind in ("answer", "explanation"):
            st.subheader("Question")
            st.write(result.data.get("question", ""))
            if debug and result.sources:
                st.subheader("Retrieved Chunks")
                for i, doc in enumerate(result.sources, start=1):
                    with st.expander(_chunk_label(i, doc)):
                        st.write(doc.page_content)
            if result.grounded:
                st.subheader("Explanation" if result.kind == "explanation" else "Answer")
            else:
                st.subheader("Not Covered")
            _render_answer_text(result.text)
        elif result.kind == "gap":
            st.info(result.text)
        elif result.kind == "notes":
            render_notes(result, key)
        elif result.kind == "quiz":
            quiz_id = result.data.get("quiz_id")
            if quiz_id is None:
                st.warning(result.text)
            else:
                render_quiz(get_quiz(quiz_id, DB_DIR), student_id, key, vectorstore)
        elif result.kind == "progress":
            if resp.route.name != "quiz":
                render_progress(result)
        elif result.kind == "clarify":
            st.info(result.text)


_CHUNK_REF = re.compile(r"\s*[\[(]\s*Chunks?\s*\d+(?:\s*(?:,|and|&|-|–)\s*(?:Chunk\s*)?\d+)*\s*[\])]", re.I)
_OFFLINE_NOTICE = re.compile(r"I found (?:this|relevant material)[^\n]*?(?:chunks|explanation):[ \t]*\n?", re.I)


def _student_text(text: str) -> str:
    """The answer as a student should read it: no [Chunk N] markers, no 'LLM unavailable' preamble."""
    text = _OFFLINE_NOTICE.sub("", text or "")
    text = _CHUNK_REF.sub("", text)
    text = re.sub(r"\bChunks?\s+\d+\b", "", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    lines = [re.sub(r"^[*•]\s+", "- ", line) for line in lines]
    out: list[str] = []
    for line in (line for line in lines if line):
        heading = line.startswith("**") and line.endswith("**")  # one part of a multi-part answer
        if out and (heading or out[-1].startswith("**") and out[-1].endswith("**")):
            out.append("")  # a blank line, so markdown does not run it into the bullet before it
        out.append(line)
    return "\n".join(out).strip()


def _bullets_from_text(text: str, limit: int = 5) -> list[str]:
    text = _student_text(text)
    existing = [line.strip(" -•") for line in text.splitlines() if line.strip().startswith(("-", "•"))]
    if existing:
        return existing[:limit]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]
    return sentences[:limit] if sentences else ([text] if text else [])


def _render_answer_text(text: str) -> None:
    """Show an answer with its own structure (lead sentence, bullets, steps) kept; plain text becomes bullets."""
    cleaned = _student_text(text)
    structured = any(re.match(r"(?:-|\d+[.)])\s", line) for line in cleaned.splitlines())
    if structured:
        st.markdown(cleaned)
        return
    bullets = _bullets_from_text(cleaned)
    if len(bullets) > 1:
        for bullet in bullets:
            st.markdown(f"- {bullet}")
    else:
        st.markdown(cleaned)


def render_student_response(resp) -> None:
    for result in resp.results:
        if result.kind in ("answer", "explanation"):
            st.subheader("Question")
            st.write(result.data.get("question", ""))
            st.subheader("Answer" if result.kind == "answer" else "Explanation")
            _render_answer_text(result.text)
        elif result.kind == "gap":
            st.info("The selected material does not contain enough information to answer this.")
        elif result.kind == "clarify":
            st.info(result.text)


_VOICE_PROBLEMS = {
    "too_short": "That recording was too short. Press record, speak, then press stop.",
    "silent": "No speech was picked up. Check that the right microphone is selected and try again.",
    "noise": "That sounded like background noise, not speech. Try again in a quieter spot.",
    "no_speech": "No words could be made out. Try again, a little closer to the microphone.",
}
USE_VOCABULARY_HINT = True


def render_voice_input(vectorstore) -> None:
    """Record a spoken question, transcribe it locally, and put the text in the message box."""
    audio = st.audio_input("Or speak your question", sample_rate=16000, key="ask_audio")
    auto = st.checkbox(
        "Ask right after transcribing",
        key="ask_auto",
        help="Off by default so you can check the text first; a misheard question costs a minute "
        "or two of Llama 3 time.",
    )
    if audio is not None:
        data = audio.getvalue()
        digest = hashlib.sha1(data).hexdigest()
        if st.session_state.get("ask_audio_digest") != digest:  # a new recording, not a rerun
            st.session_state["ask_audio_digest"] = digest
            try:
                with st.spinner("Transcribing (the first time also loads the speech model) ..."):
                    if "speech_hint" not in st.session_state:
                        st.session_state["speech_hint"] = (
                            course_vocabulary_hint([d.page_content for d in vectorstore.docstore._dict.values()])
                            if USE_VOCABULARY_HINT else ""
                        )
                    heard = get_recognizer().transcribe(data, hint=st.session_state["speech_hint"] or None)
            except Exception as exc:  # speech model missing while offline, unreadable audio, ...
                st.session_state["voice_notice"] = ("error", f"Could not transcribe: {exc}")
            else:
                if heard.text:
                    st.session_state["ask_message"] = heard.text
                    st.session_state["auto_ask_pending"] = auto
                    st.session_state["voice_notice"] = (
                        "ok",
                        f"Heard {heard.seconds:.0f} s of speech. "
                        + ("Asking now." if auto else "Edit the text if needed, then press Ask."),
                    )
                else:
                    st.session_state["voice_notice"] = ("warn", _VOICE_PROBLEMS.get(heard.problem, "Nothing was transcribed."))
    notice = st.session_state.get("voice_notice")
    if notice:
        {"ok": st.caption, "warn": st.warning, "error": st.error}[notice[0]](notice[1])


def _metric_row(label: str, value) -> dict:
    return {"Metric": label, "Value": value}


def render_oral_check(vectorstore, student_id) -> None:
    """Generate and score a short spoken-answer understanding check."""
    st.write("Get a question from your material, answer it aloud, then compare your answer with a model answer.")
    sources = _library(vectorstore)
    chosen_files = _material_gate()
    if not chosen_files:
        return

    if st.button("Generate oral question", type="primary", key="oral_generate"):
        if not chosen_files:
            st.warning("Choose your material first (sidebar).")
        else:
            scope = _scope_for_selection(chosen_files, sources)
            with st.spinner("Oral Assessor is preparing a question ..."):
                asked_topics = st.session_state.setdefault("oral_asked_topics", [])
                st.session_state["oral_question_result"] = _orchestrator().oral_assessor.generate_question_from_material(
                    _context(vectorstore, student_id, scope), exclude_topics=asked_topics
                )
                new_topic = st.session_state["oral_question_result"].data.get("topic")
                if new_topic and new_topic not in asked_topics:
                    asked_topics.append(new_topic)
                st.session_state.pop("oral_assessment_result", None)
                st.session_state.pop("oral_transcript", None)
                st.session_state.pop("oral_notice", None)

    question_result = st.session_state.get("oral_question_result")
    if not question_result:
        _render_recent_oral_assessments(student_id)
        return

    if not question_result.grounded:
        st.warning(question_result.text)
        return

    st.subheader("Question")
    st.write(question_result.text)

    audio = st.audio_input("Record your answer", sample_rate=16000, key="oral_audio")
    if audio is not None:
        data = audio.getvalue()
        digest = hashlib.sha1(data).hexdigest()
        if st.session_state.get("oral_audio_digest") != digest:
            st.session_state["oral_audio_digest"] = digest
            try:
                with st.spinner("Transcribing and assessing your answer ..."):
                    if "speech_hint" not in st.session_state:
                        st.session_state["speech_hint"] = (
                            course_vocabulary_hint([d.page_content for d in vectorstore.docstore._dict.values()])
                            if USE_VOCABULARY_HINT else ""
                        )
                    heard = get_recognizer().transcribe(data, hint=st.session_state["speech_hint"] or None)
                    if not heard.text:
                        st.session_state["oral_notice"] = (
                            "warn",
                            _VOICE_PROBLEMS.get(heard.problem, "Nothing was transcribed."),
                        )
                    else:
                        st.session_state["oral_transcript"] = heard.text
                        qdata = question_result.data
                        st.session_state["oral_assessment_result"] = _orchestrator().oral_assessor.assess_response(
                            qdata["topic"],
                            qdata["question"],
                            heard.text,
                            data,
                            _context(vectorstore, student_id, _scope_for_selection(chosen_files, sources)),
                            context=qdata.get("context"),
                        )
                        st.session_state["oral_notice"] = ("ok", f"Heard {heard.seconds:.0f} s of speech.")
            except Exception as exc:
                st.session_state["oral_notice"] = ("error", f"Could not assess the recording: {exc}")

    notice = st.session_state.get("oral_notice")
    if notice:
        {"ok": st.caption, "warn": st.warning, "error": st.error}[notice[0]](notice[1])

    if st.session_state.get("oral_transcript"):
        st.subheader("Transcript")
        st.write(st.session_state["oral_transcript"])

    result = st.session_state.get("oral_assessment_result")
    if result:
        st.subheader("Assessment")
        st.markdown(result.text)
        st.subheader("Best Answer")
        for bullet in _bullets_from_text(question_result.data.get("best_answer") or result.data.get("best_answer", ""), limit=40):
            st.markdown(f"- {bullet}")
        metrics = result.data["metrics"]
        st.dataframe(
            [
                _metric_row("Duration", f"{metrics['duration_seconds']:.1f} s"),
                _metric_row("Speech time", f"{metrics['speech_seconds']:.1f} s"),
                _metric_row("Pause time", f"{metrics['pause_seconds']:.1f} s"),
                _metric_row("Pauses", metrics["pause_count"]),
                _metric_row("Long pauses", metrics["long_pause_count"]),
                _metric_row("Longest pause", f"{metrics['max_pause_seconds']:.1f} s"),
                _metric_row("Leading silence", f"{metrics['leading_silence_seconds']:.1f} s"),
                _metric_row("Words per minute", metrics["words_per_minute"]),
                _metric_row("Filler words", metrics["filler_count"]),
            ],
            hide_index=True,
            width="stretch",
        )

    _render_recent_oral_assessments(student_id)


def _render_recent_oral_assessments(student_id) -> None:
    if not student_id:
        st.caption("Enter a Student ID in the sidebar to save oral understanding checks to your progress.")
        return
    rows = list_oral_assessments(student_id=student_id, db_dir=DB_DIR)[-5:]
    if not rows:
        return
    st.subheader("Recent oral checks")
    st.dataframe(
        [
            {
                "Topic": row.topic,
                "Level": row.understanding_level,
                "Content score": f"{row.content_score:.0%}",
                "Words/min": row.metrics.get("words_per_minute", 0),
                "Long pauses": row.metrics.get("long_pause_count", 0),
            }
            for row in reversed(rows)
        ],
        hide_index=True,
        width="stretch",
    )


def _delete_file(name: str, vectorstore) -> None:
    """Remove a PDF and everything tied to it, then refresh what the app remembers."""
    done = delete_material(name, vectorstore, DB_DIR, DATA_DIR)
    _load_index.clear()
    st.session_state["active_files"] = [f for f in st.session_state.get("active_files", []) if f != name]
    st.session_state.pop("confirm_delete", None)
    st.session_state.pop("unsaved_attempts", None)
    st.session_state["library_notice"] = (
        f"Deleted {_nice_name(name)}, with {done['quizzes']} quiz(es), {done['questions']} question(s) "
        f"and {done['oral_checks']} oral check(s) on it."
    )


def render_student_material(vectorstore=None) -> None:
    """The Material page: add PDFs on the left, pick what to study (or delete) from your library on the right."""
    ready = set(_library(vectorstore))
    notice = st.session_state.pop("library_notice", None)
    if notice:
        st.success(notice)
    left, right = st.columns([1, 1.5], gap="large")

    with left:
        st.markdown("#### Add material")
        upload_key = f"student_material_upload_{st.session_state.get('upload_round', 0)}"
        uploaded_files = st.file_uploader(
            "Drop your PDF notes here", type=["pdf"], accept_multiple_files=True, key=upload_key,
            label_visibility="collapsed",
        )
        if uploaded_files:
            for uploaded_file in uploaded_files:
                with open(DATA_DIR / uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.session_state["upload_round"] = st.session_state.get("upload_round", 0) + 1  # empties the drop box
            st.session_state["library_notice"] = (
                f"Added {len(uploaded_files)} file(s). Press Update library to start using them."
            )
            st.rerun()
        st.caption("PDFs need real text; scanned pages cannot be read.")
        if st.button("Update library", type="primary", key="student_rebuild", use_container_width=True):
            _run_ingestion()

    with right:
        st.markdown("#### Your library")
        pdfs = sorted(p.name for p in DATA_DIR.glob("*.pdf"))
        pending = st.session_state.get("confirm_delete")
        if pending in pdfs:
            with st.container(border=True):
                st.warning(
                    f"Delete **{_nice_name(pending)}**? Its notes, quizzes, questions and all your progress on it "
                    "will be removed for good."
                )
                yes, no = st.columns(2)
                yes.button("Yes, delete it", type="primary", key="delete_yes", use_container_width=True,
                           on_click=_delete_file, args=(pending, vectorstore))
                no.button("Keep it", key="delete_no", use_container_width=True,
                          on_click=lambda: st.session_state.pop("confirm_delete", None))
        if not pdfs:
            st.info("Nothing here yet. Add a PDF to get started.")
        active = set(st.session_state.get("active_files", []))
        for start in range(0, len(pdfs), 2):
            columns = st.columns(2)
            for column, name in zip(columns, pdfs[start:start + 2]):
                with column, st.container(border=True):
                    title, cross = st.columns([6, 1], vertical_alignment="center")
                    title.markdown(f"**{_nice_name(name)}**")
                    cross.button(
                        "✕", key=f"del_{name}", help="Delete this material and its progress",
                        on_click=lambda n=name: st.session_state.update(confirm_delete=n),
                    )
                    if name not in ready:
                        st.caption("Needs an update before you can use it")
                    else:
                        st.button(
                            "✓ Studying" if name in active else "Study this",
                            key=f"pick_{name}", on_click=_toggle_file, args=(name,),
                            type="primary" if name in active else "secondary", use_container_width=True,
                        )


def render_quiz_builder(vectorstore, student_id) -> None:
    st.write("A short quiz from your material. Every new quiz asks different questions.")
    sources = _library(vectorstore)
    chosen_files = _material_gate()
    if not chosen_files:
        return

    def generate(focus=None) -> None:
        st.session_state["quiz_focus"] = focus
        scope = _scope_for_selection(chosen_files, sources)
        seen = st.session_state.setdefault("quiz_seen_questions", [])
        with st.spinner("Quiz Generator is writing questions from the selected material ..."):
            result = _orchestrator().quiz_generator.run_from_material(
                _context(vectorstore, student_id, scope),
                n_questions=5,
                scope=SCOPE_PERSONAL,
                topic=_material_label(chosen_files),
                avoid=seen,  # a new quiz does not repeat the questions of earlier ones
                focus=focus,
            )
        st.session_state["student_material_quiz"] = result
        seen.extend(q["question"] for q in result.data.get("questions", []))
        del seen[:-40]

    request = st.session_state.pop("quiz_request", None)  # a "Take test" button on My progress
    if request and request["file"] in sources:
        chosen_files = [request["file"]]
        st.info(f"Test on: {request['topic']}")
        generate(focus=request["topic"])

    if st.button("Generate quiz", type="primary", key="quiz_generate"):
        if not chosen_files:
            st.warning("Choose your material first (sidebar).")
        else:
            generate()

    result = st.session_state.get("student_material_quiz")
    if not result:
        return
    if result.data.get("quiz_id") is None:
        st.warning(result.text)
        return
    quiz = get_quiz(result.data["quiz_id"], DB_DIR)
    render_quiz(quiz, student_id, "personal_material", vectorstore)
    if st.session_state.get(f"quizresult_personal_material_{quiz['id']}") is not None:
        if st.button("Take a new quiz", key="quiz_retake"):
            if not chosen_files:
                st.warning("Choose your material first (sidebar).")
            else:
                generate(focus=st.session_state.get("quiz_focus"))
                st.rerun()


# --------------------------------------------------------------------------- #
# Student view                                                               #
# --------------------------------------------------------------------------- #
# What each section keeps in the session; cleared when the student leaves the section so it opens empty next time.
_SECTION_STATE = {
    "Ask": ["response", "auto_ask_pending", "voice_notice", "ask_audio_digest", "ask_message"],
    "Oral check": ["oral_question_result", "oral_assessment_result", "oral_transcript", "oral_notice", "oral_audio_digest"],
    "Study notes": [],
    "Quiz": ["student_material_quiz", "quiz_focus"],
    "My progress": [],
}
_SECTION_PREFIXES = {"Study notes": ("notes_",), "Quiz": ("quizresult_", "recap_")}


def _clear_section(section: str) -> None:
    for key in _SECTION_STATE.get(section, []):
        st.session_state.pop(key, None)
    for key in [k for k in st.session_state if k.startswith(_SECTION_PREFIXES.get(section, ("\0",)))]:
        st.session_state.pop(key, None)


def render_student(student_id) -> None:
    st.title("Academic Assistant")
    st.markdown('<p class="tagline">Ask, practise and track what you have learned from your own study material.</p>', unsafe_allow_html=True)
    vectorstore = _get_vectorstore()
    if vectorstore is None:
        st.info("No material has been indexed yet. Upload PDFs below and build the local index.")
        render_student_material()
        return

    sections = ["Material", "Ask", "Oral check", "Study notes", "Quiz", "My progress"]
    # Not st.tabs: code cannot switch tabs, and "Take test" on My progress has to open the Quiz section.
    st.session_state.setdefault("main_nav", "Material")
    chosen = st.segmented_control("Section", sections, key="main_nav", label_visibility="collapsed")
    section = chosen or st.session_state.get("last_section", "Material")
    previous = st.session_state.get("last_section")
    if previous and previous != section:
        _clear_section(previous)  # leaving a section: it opens empty next time
    st.session_state["last_section"] = section

    if section == "Material":
        render_student_material(vectorstore)

    if section == "Ask":
        st.write("Ask a question about your material. You can type it, or speak and check the text before asking.")
        indexed_sources = _library(vectorstore)
        selected = _material_gate()
        render_voice_input(vectorstore)
        message = st.text_input("Your message", key="ask_message", placeholder="e.g. What is backpropagation?")

        auto_ask = st.session_state.pop("auto_ask_pending", False)
        if st.button("Ask", type="primary") or auto_ask:
            if not selected:
                st.warning("Choose your material first (sidebar).")
            elif not message.strip():
                st.warning("Please enter a message first.")
            else:
                # Filtering is only needed for a strict subset of files.
                scope = _scope_for_selection(selected, indexed_sources)
                with st.spinner("Looking through your material ..."):
                    st.session_state["response"] = _orchestrator().handle(
                        message, _context(vectorstore, student_id, scope)
                    )

        if "response" in st.session_state:
            render_student_response(st.session_state["response"])

    if section == "Study notes":
        render_notes_builder(vectorstore, student_id, "student", files=_material_gate())

    if section == "Oral check":
        render_oral_check(vectorstore, student_id)

    if section == "Quiz":
        render_quiz_builder(vectorstore, student_id)

    if section == "My progress":
        if not student_id:
            st.info("Enter a Student ID in the sidebar to track your progress.")
        else:
            pending = [] if student_id == st.session_state.get("session_student_id") else st.session_state.pop(
                "unsaved_attempts", []
            )
            for quiz, picks in pending:
                _orchestrator().progress_tracker.record_attempt(quiz, picks, _context(vectorstore, student_id))
            if pending:
                st.caption(f"Added {len(pending)} quiz result(s) from this session to your progress.")
            render_progress(_orchestrator().progress_tracker.run("", _context(vectorstore, student_id)))


# --------------------------------------------------------------------------- #
# Professor view                                                             #
# --------------------------------------------------------------------------- #
def _render_class_insight() -> None:
    rows = list_queries(db_dir=DB_DIR)
    if not rows:
        st.info("No student questions have been logged yet.")
        return

    unanswered = sum(1 for r in rows if not r["grounded"])
    resolved = sum(1 for r in rows if r["status"] == STATUS_RESOLVED)
    col_total, col_gap, col_resolved = st.columns(3)
    col_total.metric("Questions asked", len(rows))
    col_gap.metric("Not covered by the notes", unanswered)
    col_resolved.metric("Resolved by faculty", resolved)

    activity = interaction_summary(DB_DIR)
    explain = activity["by_agent"].get("Concept Explainer", 0)
    st.caption(
        f"{activity['sessions']} browsing session(s) recorded. {explain} explanation request(s). "
        f"{activity['follow_ups']} question(s) came back to a topic already asked about in the same session. "
        f"{activity['untracked']} question(s) have no session information (asked before sessions were recorded, "
        "or simulated)."
    )

    with st.spinner("Grouping questions by meaning ..."):
        clusters = summarize_clusters(rows, _load_embeddings())

    st.subheader("Topics students are asking about")
    st.caption("Questions are grouped by meaning, so different phrasings of the same doubt count together.")
    for cluster in clusters:
        with st.expander(f"{cluster.size} question(s) — {cluster.label}"):
            if cluster.gap_count:
                st.warning(f"{cluster.gap_count} of these could not be answered from the notes when asked.")
            if cluster.top_slides:
                st.write("Material students landed on: " + "; ".join(cluster.top_slides))
            for index in cluster.indices:
                st.text(f"- {rows[index]['question']}")

    slide_counts = Counter(slide_label(row["retrieved"][0]) for row in rows if row["retrieved"])
    if slide_counts:
        st.subheader("Slides students land on most")
        st.bar_chart(dict(slide_counts.most_common(8)))


def _signal_detail(signal, class_means) -> str:
    if signal.value is not None:
        return signal.evidence
    if signal.name in class_means:
        return f"class average used ({class_means[signal.name]:.0%})"
    return "no data for this signal anywhere in the class"


def _render_concept_confusion() -> None:
    """Which concepts to revisit first: several kinds of evidence combined, shown with that evidence."""
    st.caption(
        "A concept is a slide topic. Many questions about it do not by themselves mean students are confused, so "
        "this combines several kinds of evidence: how many students asked, whether they came back or asked to have "
        "it explained, how they did on the class quiz, oral checks, and whether it kept coming up across days. "
        "It is a ranking aid for deciding what to revisit first, not a measurement of understanding."
    )
    with st.expander("How much each kind of evidence counts"):
        st.caption("Starting values are judgment calls. Change them and watch the ranking and its stability.")
        weights = {
            name: st.slider(label, 0, 100, round(DEFAULT_WEIGHTS[name] * 100), key=f"weight_{name}")
            for name, (label, _w) in SIGNALS.items()
        }
    report = load_and_compute(DB_DIR, {name: v / 100 for name, v in weights.items()})
    if not report.concepts:
        st.info(
            "Not enough data yet. A concept needs at least one kind of evidence from enough different students "
            f"(at least {MIN_COHORT}) or enough questions."
        )
    else:
        stability = rank_stability(report)
        st.dataframe(
            [
                {
                    "Rank": rank,
                    "Concept": c.concept,
                    "Confusion score": round(c.score),
                    "Confidence": c.confidence,
                    "Stays in top 3 when weights change": f"{stability[c.key].top_share:.0%}",
                    "What stands out": c.stands_out or "Nothing above the class average",
                }
                for rank, c in enumerate(report.concepts, start=1)
            ],
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Confidence: Low means one kind of evidence backs the score, Medium two, High three (questions, quizzes, "
            "oral checks). A signal with too little data is filled in with the class average, so missing data never "
            "lowers or raises a concept's score on its own."
        )

        quizzes = list_quizzes(SCOPE_CLASS, DB_DIR)
        found = likely_misconceptions(analyze_quizzes(quizzes, list_attempts(db_dir=DB_DIR)))
        for rank, c in enumerate(report.concepts, start=1):
            with st.expander(f"{rank}. {c.concept} (score {round(c.score)}, {c.confidence.lower()} confidence)"):
                st.dataframe(
                    [
                        {
                            "Evidence": s.label,
                            "Weight": f"{s.weight:.0%}",
                            "Reading": f"{s.value:.0%}" if s.value is not None else "Not enough data",
                            "Detail": _signal_detail(s, report.means),
                        }
                        for s in c.signals
                    ],
                    hide_index=True,
                    width="stretch",
                )
                for q in (m for m in found if concept_key(m.topic) == c.key):
                    wrong = q.misconception
                    st.markdown(
                        f"Quiz finding: {wrong.share:.0%} of the class chose \"{wrong.text}\" instead of "
                        f"\"{q.correct_option.text}\" on \"{q.question}\"."
                    )
    if report.unscored:
        st.caption(
            "Asked about but too little evidence to score: "
            + "; ".join(f"{name} ({n})" for name, n in report.unscored[:10])
        )
    st.caption(
        f"Based on {report.active_students} identified student(s). Questions the notes could not answer have no slide, "
        "so they are in Pending Gaps instead. Slides on the same idea are not merged yet."
    )


def _render_pending_gaps() -> None:
    gaps = list_queries(status=STATUS_GAP, db_dir=DB_DIR)
    if not gaps:
        st.success("No unresolved questions. Everything students asked was covered.")
        return

    st.caption(
        "These questions were not covered by the notes. Write one answer per topic; "
        "it is added to the course material and used for all future questions."
    )
    groups = cluster_questions([g["question"] for g in gaps], _load_embeddings())

    for group in groups:
        members = [gaps[i] for i in group]
        representative = members[0]["question"]
        key = members[0]["id"]

        with st.container(border=True):
            st.markdown(f"**{len(members)} student question(s)** — e.g. \"{representative}\"")
            for member in members:
                st.text(f"- {member['question']}")

            answer = st.text_area("Your answer", key=f"answer_{key}")
            if st.button("Add to knowledge base", key=f"resolve_{key}"):
                if not answer.strip():
                    st.warning("Write an answer first.")
                elif not vectorstore_exists(DB_DIR):
                    st.error("Build the index first (Course Material tab).")
                else:
                    # Fresh copy from disk so the shared cached index is never mutated.
                    fresh_index = load_vectorstore(DB_DIR, _load_embeddings())
                    add_faculty_answer(fresh_index, representative, answer.strip(), DB_DIR)
                    resolve_gaps([m["id"] for m in members], representative, answer.strip(), DB_DIR)
                    st.toast(f"Added to the knowledge base ({len(members)} question(s) resolved).")
                    st.rerun()


def _render_class_quizzes() -> None:
    vectorstore = _get_vectorstore()
    if vectorstore is None:
        st.info("Build the index first (Course Material tab).")
        return

    st.caption(
        "Faculty Insight finds topics the notes cover but the class keeps asking about, then the "
        "Quiz Generator writes a quiz on each and publishes it to students. Each quiz takes "
        "2 to 3 minutes on CPU."
    )
    if st.button("Generate quizzes from class confusion", type="primary"):
        with st.spinner("Faculty Insight -> Quiz Generator ..."):
            st.session_state["pipeline"] = _orchestrator().publish_class_quizzes(_context(vectorstore))
    if "pipeline" in st.session_state:
        render_trace(st.session_state["pipeline"].trace)

    quizzes = list_quizzes(SCOPE_CLASS, DB_DIR)
    st.subheader(f"Published quizzes ({len(quizzes)})")
    st.caption("Each question names the slide it was written from, so you can check it before students rely on it.")
    for quiz in quizzes:
        with st.expander(f"{quiz['topic']} — {len(quiz['questions'])} question(s)"):
            for i, q in enumerate(quiz["questions"], start=1):
                st.markdown(f"**{i}. {q['question']}**")
                for j, option in enumerate(q["options"]):
                    st.text(f"  {'[correct] ' if j == q['answer_index'] else '          '}{option}")
                src = q.get("source", {})
                st.caption(f"Source: {src.get('file', '')}, page {src.get('page', '?')} ({src.get('slide', '')})")

    attempts = list_attempts(db_dir=DB_DIR)
    st.subheader("Class results by topic")
    if not attempts:
        st.info("No quiz attempts yet.")
        return
    totals = defaultdict(lambda: [0, 0, 0])  # correct, total, attempts
    for attempt in attempts:
        totals[attempt["topic"]][0] += attempt["correct"]
        totals[attempt["topic"]][1] += attempt["total"]
        totals[attempt["topic"]][2] += 1
    st.dataframe(
        [
            {
                "Topic": topic,
                "Attempts": n,
                "Average score": f"{100 * c / t:.0f}%",
                "Flag": "Consider re-teaching" if c / t < 0.6 else "",
            }
            for topic, (c, t, n) in sorted(totals.items(), key=lambda kv: kv[1][0] / kv[1][1])
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption("Results are aggregated; individual students are not shown.")
    _render_item_analysis(quizzes, attempts)


def _discrimination_text(q) -> str:
    value = q.discrimination
    if value is None:
        return "Not enough data"
    if q.discrimination_p >= SIGNIFICANT_P:
        return f"Unclear ({value:+.2f})"       # too few students to tell this apart from chance
    if value >= 0.1:
        return f"Yes ({value:+.2f})"
    if value > KEY_SUSPECT_AT:
        return f"No difference ({value:+.2f})"
    return f"Reversed ({value:+.2f})"


def _render_item_analysis(quizzes, attempts) -> None:
    """What the class's answers say about each quiz question: likely misconceptions and weak questions."""
    st.subheader("Question analysis")
    st.caption(
        f"Looks at every question and every answer option. Numbers appear only once at least {MIN_COHORT} different "
        "students have taken a quiz, and only each student's first attempt counts."
    )
    analyses = analyze_quizzes(quizzes, attempts)
    waiting = [a for a in analyses if not a.reportable]
    ready = [a for a in analyses if a.reportable]
    for a in waiting:
        st.caption(f"{human_title(a.topic)}: {a.students} of the {a.needed} students needed so far.")
    if not ready:
        st.info("No quiz has enough students yet for a question-level view.")
        return

    st.markdown("**Likely misconceptions**")
    misconceptions = likely_misconceptions(ready)
    if not misconceptions:
        st.info(
            "No wrong answer clearly stood out. That does not rule one out: with a small class a real shared "
            "mistake can be missed."
        )
    for q in misconceptions:
        wrong = q.misconception
        with st.container(border=True):
            st.markdown(
                f"**{human_title(q.topic)}, question {q.number}.** {wrong.share:.0%} of the class "
                f"({wrong.count} of {q.students}) chose \"{wrong.text}\" instead of \"{q.correct_option.text}\"."
            )
            st.write(q.question)
            src = q.source
            st.caption(
                f"This is a pattern worth a look, not proof of a misconception. Source: {src.get('file', '')}, "
                f"page {src.get('page', '?')} ({src.get('slide', '')})"
            )

    st.markdown("**Questions to check**")
    flagged = questions_to_review(ready)
    if not flagged:
        st.info("No question was flagged. The checks are cautious, so this is not a guarantee that every question is sound.")
    for q in flagged:
        with st.container(border=True):
            st.markdown(f"**{human_title(q.topic)}, question {q.number}** ({q.correct_rate:.0%} correct)")
            st.write(q.question)
            for flag in q.flags:
                st.caption(f"{flag}: {FLAG_HELP[flag]}")

    st.markdown("**All questions**")
    for a in ready:
        with st.expander(f"{human_title(a.topic)} ({a.students} students, average {a.average:.0%})"):
            st.dataframe(
                [
                    {
                        "Question": f"{q.number}. {q.question}",
                        "Correct": f"{q.correct_rate:.0%}",
                        "Skipped": q.skipped,
                        "Strong students did better?": _discrimination_text(q),
                        "Flags": ", ".join(q.flags),
                    }
                    for q in a.questions
                ],
                hide_index=True,
                width="stretch",
            )
            for q in a.questions:
                st.markdown(f"**{q.number}. {q.question}**")
                st.dataframe(
                    [
                        {
                            "Answer option": o.text + ("  (correct)" if o.is_correct else ""),
                            "Chosen by": o.count,
                            "Share of class": f"{o.share:.0%}",
                        }
                        for o in q.options
                    ],
                    hide_index=True,
                    width="stretch",
                )
    st.caption(
        '"Strong students did better?" compares each question with the results of the same students on the other '
        "questions. It is a hint from a small class, not a measurement."
    )


def _render_course_material() -> None:
    uploaded_files = st.file_uploader("Upload PDF course notes", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with open(DATA_DIR / uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"Saved {len(uploaded_files)} file(s) to '{DATA_DIR.name}/'.")

    existing_pdfs = sorted(p.name for p in DATA_DIR.glob("*.pdf"))
    st.caption(f"{len(existing_pdfs)} PDF(s) currently in data/:")
    for name in existing_pdfs:
        st.text(f"- {name}")

    if st.button("Build / Rebuild Index"):
        _run_ingestion()


def render_professor() -> None:
    st.title("Faculty Dashboard")
    tab_concepts, tab_insight, tab_gaps, tab_quizzes, tab_notes, tab_material = st.tabs(
        ["Concepts to Revisit", "Class Insight", "Pending Gaps", "Class Quizzes", "Study Notes", "Course Material"]
    )
    with tab_concepts:
        _render_concept_confusion()
    with tab_insight:
        _render_class_insight()
    with tab_gaps:
        _render_pending_gaps()
    with tab_quizzes:
        _render_class_quizzes()
    with tab_notes:
        vectorstore = _get_vectorstore()
        if vectorstore is None:
            st.info("Build the index first (Course Material tab).")
        else:
            render_notes_builder(vectorstore, None, "prof")
    with tab_material:
        _render_course_material()


# --------------------------------------------------------------------------- #
# Sidebar + routing                                                          #
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Academic Assistant")
    role = st.radio("I am a", ["Student", "Professor"], horizontal=True)
    student_id = None
    if role == "Student":
        typed_student_id = st.text_input(
            "Student ID (optional)",
            placeholder="e.g. 23BIT0305",
            help="Add your ID to keep your progress between visits. Leave it blank to track this session only.",
        ).strip()
        if "session_student_id" not in st.session_state:
            st.session_state["session_student_id"] = "session_" + hashlib.sha1(str(id(st.session_state)).encode()).hexdigest()[:10]
        student_id = typed_student_id or st.session_state["session_student_id"]

        library = _library(_get_vectorstore())
        if library:
            st.session_state["active_files"] = [f for f in st.session_state.get("active_files", []) if f in library]
            st.multiselect(
                "Studying", library, key="active_files", format_func=_nice_name,
                placeholder="Choose your material",
                help="Choose once. Ask, Oral check, Study notes and Quiz all use this material.",
            )

if role == "Student":
    render_student(student_id)
else:
    render_professor()
