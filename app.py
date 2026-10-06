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
from collections import Counter, defaultdict
from pathlib import Path

import streamlit as st

from agents import AgentContext, Orchestrator
from agents.routing import ROUTE_LABELS
from rag_core.config import DATA_DIR, DB_DIR, FACULTY_SOURCE_NAME, TOP_K
from rag_core.embeddings import get_embeddings
from rag_core.insights import cluster_questions, slide_label, summarize_clusters
from rag_core.learning_log import (
    SCOPE_CLASS,
    get_quiz,
    list_attempts,
    list_quizzes,
    record_route_feedback,
)
from rag_core.loader import load_all_pdfs
from rag_core.normalize import short_section_title
from rag_core.pdf_export import CHECK_LEGEND, CHECK_MARK, notes_to_pdf, source_line
from rag_core.query_log import (
    STATUS_GAP,
    STATUS_RESOLVED,
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

st.set_page_config(page_title="Academic Assistant", layout="wide")


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


def _context(vectorstore, student_id=None, sources=None) -> AgentContext:
    return AgentContext(
        vectorstore=vectorstore,
        db_dir=DB_DIR,
        student_id=student_id,
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

    st.success(
        f"Index built from {len(documents)} page(s) / {len(chunks) - len(faculty_docs)} "
        f"chunk(s) plus {len(faculty_docs)} professor answer(s)."
    )


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


def render_progress(result) -> None:
    topics = result.data["topics"]
    if not topics:
        st.info(result.text)
        return
    st.write(result.text)
    st.dataframe(
        [
            {
                "Topic": t.topic,
                "Status": t.status,
                "Questions asked": t.questions,
                "Quiz score": f"{t.quiz_correct}/{t.quiz_total}" if t.quiz_total else "-",
                "Why": t.reason,
            }
            for t in topics
        ],
        hide_index=True,
        width="stretch",
    )


def render_quiz(quiz: dict, student_id, key: str, vectorstore) -> None:
    """A quiz form; on submit the Progress Tracker grades and records it."""
    quiz_key = f"{key}_{quiz['id']}"
    result_key = f"quizresult_{quiz_key}"
    graded = st.session_state.get(result_key)

    st.subheader(f"Quiz: {quiz['topic']}")
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

    if graded["correct"] / max(graded["total"], 1) < 0.8:
        recap_key = f"recap_{quiz_key}"
        if st.button("Explain this topic", key=f"explainbtn_{quiz_key}"):
            with st.spinner("Concept Explainer is preparing a recap ..."):
                st.session_state[recap_key] = _orchestrator().handle(
                    f"explain {quiz['topic']} step by step", _context(vectorstore, student_id)
                )
        if recap_key in st.session_state:
            render_response(
                st.session_state[recap_key], student_id, f"recap_{quiz_key}", vectorstore, feedback=False
            )


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
    for section in sections:
        st.markdown(f"### {section['heading']}")
        st.markdown("**Definition**")
        st.write(section["definition"])
        check = section.get("check", {})
        st.markdown("**Key points**")
        for index, point in enumerate(section["key_points"]):
            st.markdown(f"- {point}{CHECK_MARK if index in check.get('key_points', []) else ''}")
        st.markdown("**Use case**")
        st.write(section["use_case"] + (CHECK_MARK if check.get("use_case") else ""))
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
    if sections:
        st.download_button(
            "Download notes as PDF",
            data=notes_to_pdf(sections, data["title"], data["files"]),
            file_name="study_notes.pdf",
            mime="application/pdf",
            key=f"download_{key}",
        )


def render_notes_builder(vectorstore, student_id, key: str) -> None:
    """Pick topics from the material (or type them) and generate downloadable notes."""
    st.write(
        "Short, heading-wise notes in simple language: a definition, key points and a use case "
        "for each topic, written only from your uploaded material."
    )
    sources = [s for s in list_indexed_sources(vectorstore) if s != FACULTY_SOURCE_NAME]
    chosen_files = st.multiselect("From file(s)", options=sources, default=sources, key=f"nbfiles_{key}")
    available = list_indexed_topics(vectorstore, chosen_files)
    picked = st.multiselect(
        "Topics from the material (up to 5)", options=available, max_selections=5, key=f"nbtopics_{key}"
    )
    typed = st.text_input(
        "Or type topics, separated by commas", placeholder="e.g. ACID, BASE", key=f"nbtyped_{key}"
    )
    topics = (picked + [t.strip() for t in typed.split(",") if t.strip()])[:5]
    st.caption(f"{len(topics)} topic(s) selected. Allow about 2 to 3 minutes per topic on CPU.")

    if st.button("Generate notes", type="primary", key=f"nbgo_{key}"):
        if not chosen_files:
            st.warning("Select at least one file.")
        elif not topics:
            st.warning("Pick or type at least one topic.")
        else:
            bar = st.progress(0.0, text="Starting ...")

            def on_progress(done: int, total: int, topic: str) -> None:
                bar.progress(done / total, text=f"Writing notes {done + 1} of {total}: {topic}" if topic else "Done")

            scope = None if set(chosen_files) == set(sources) else chosen_files
            st.session_state[f"notes_{key}"] = _orchestrator().note_generator.run(
                "notes", _context(vectorstore, student_id, scope), topics=topics, on_progress=on_progress
            )
            bar.empty()

    if f"notes_{key}" in st.session_state:
        render_notes(st.session_state[f"notes_{key}"], key)


def render_response(resp, student_id, key: str, vectorstore, feedback: bool = True) -> None:
    render_trace(resp.trace)
    if feedback and resp.message:
        render_route_feedback(resp, key)
    for result in resp.results:
        if result.kind in ("answer", "explanation"):
            st.subheader("Question")
            st.write(result.data.get("question", ""))
            if result.sources:
                st.subheader("Retrieved Chunks")
                for i, doc in enumerate(result.sources, start=1):
                    with st.expander(_chunk_label(i, doc)):
                        st.write(doc.page_content)
            if result.grounded:
                st.subheader("Explanation" if result.kind == "explanation" else "Answer")
            else:
                st.subheader("Not Covered")
            st.markdown(result.text)
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


# --------------------------------------------------------------------------- #
# Student view                                                               #
# --------------------------------------------------------------------------- #
def render_student(student_id) -> None:
    st.title("Academic Assistant")
    vectorstore = _get_vectorstore()
    if vectorstore is None:
        st.info("No course material has been indexed yet. Ask your professor to upload the notes.")
        return

    tab_ask, tab_notes, tab_quizzes, tab_progress = st.tabs(
        ["Ask", "Study notes", "Class quizzes", "My progress"]
    )

    with tab_ask:
        st.write(
            "Ask a question, or try: 'explain BASE simply', 'quiz me on RDS read replicas', "
            "'what should I study next'."
        )
        indexed_sources = list_indexed_sources(vectorstore)
        selected = st.multiselect(
            "Search within",
            options=indexed_sources,
            default=indexed_sources,
            help="Only chunks from the selected file(s) are used.",
        )
        render_voice_input(vectorstore)
        message = st.text_input("Your message", key="ask_message", placeholder="e.g. What is backpropagation?")

        auto_ask = st.session_state.pop("auto_ask_pending", False)
        if st.button("Ask", type="primary") or auto_ask:
            if not selected:
                st.warning("Select at least one file to search within.")
            elif not message.strip():
                st.warning("Please enter a message first.")
            else:
                # Filtering is only needed for a strict subset of files.
                scope = None if set(selected) == set(indexed_sources) else selected
                with st.spinner("Agents are working (Llama 3 on CPU can take a minute) ..."):
                    st.session_state["response"] = _orchestrator().handle(
                        message, _context(vectorstore, student_id, scope)
                    )

        if "response" in st.session_state:
            render_response(st.session_state["response"], student_id, "ask", vectorstore)

    with tab_notes:
        render_notes_builder(vectorstore, student_id, "student")

    with tab_quizzes:
        quizzes = list_quizzes(SCOPE_CLASS, DB_DIR)
        if not quizzes:
            st.info("Your professor has not published any quizzes yet.")
        else:
            choice = st.selectbox(
                "Quiz",
                options=[q["id"] for q in quizzes],
                format_func=lambda qid: next(
                    f"{q['topic']} ({len(q['questions'])} questions)" for q in quizzes if q["id"] == qid
                ),
            )
            render_quiz(get_quiz(choice, DB_DIR), student_id, "class", vectorstore)

    with tab_progress:
        if not student_id:
            st.info("Enter a Student ID in the sidebar to track your progress.")
        else:
            render_progress(_orchestrator().progress_tracker.run("", _context(vectorstore, student_id)))
            if st.button("What should I study next?"):
                with st.spinner("Progress Tracker and Concept Explainer are working ..."):
                    st.session_state["study_response"] = _orchestrator().handle(
                        "what should I study next", _context(vectorstore, student_id)
                    )
            if "study_response" in st.session_state:
                render_response(
                    st.session_state["study_response"], student_id, "study", vectorstore, feedback=False
                )


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
    tab_insight, tab_gaps, tab_quizzes, tab_notes, tab_material = st.tabs(
        ["Class Insight", "Pending Gaps", "Class Quizzes", "Study Notes", "Course Material"]
    )
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
    role = st.radio("View as", ["Student", "Professor"])
    st.caption("Demo toggle only: there is no login yet.")
    student_id = None
    if role == "Student":
        student_id = st.text_input(
            "Student ID (optional)",
            help="Lets the Progress Tracker remember your questions and quizzes. "
            "Leave blank to stay anonymous. Professors see class-wide patterns, not who asked.",
        ).strip() or None
    st.divider()
    st.caption(f"Top-k retrieved chunks: {TOP_K}")
    st.caption("LLM: Llama 3 (via Ollama, local)")
    st.caption("Embeddings: all-MiniLM-L6-v2 (HuggingFace)")

if role == "Student":
    render_student(student_id)
else:
    render_professor()
