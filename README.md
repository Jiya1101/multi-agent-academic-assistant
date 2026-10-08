# Academic Assistant

A multi-agent study assistant that runs **entirely on your own computer**. Students ask questions, get
quizzed, and generate study notes from their own course PDFs. Professors see where the whole class is
confused and turn unanswered questions into new course content.

Built as a B.Tech project: *Multi-Agent AI Framework for Personalized Academic Assistance*.

- Local and offline: Llama 3 (through [Ollama](https://ollama.com)), MiniLM embeddings, FAISS and Whisper.
  No API keys, no cloud, no per-use cost. Course material and questions never leave the machine.
- Answers come only from the uploaded notes, with `[Chunk N]` citations. If the notes do not cover a
  question, it says so instead of guessing.

## What it does

**For students**
- **Material**: add PDFs, choose what you are studying once (every section uses it), and delete a file together with the progress tied to it.
- **Ask** questions about the material (by typing or **speaking**); get a short, structured answer. A message with two questions gets both answered.
- **Oral check**: a spoken question on one of the material's main headings, a different one each time. You get a content score, the key points you missed, the full model answer and pause/fluency metrics.
- **Study notes**: heading-wise notes (definition, key points, use case when the slides give one), as a **downloadable PDF**.
- **Quiz**: a new set of multiple-choice questions every time (definitions, abbreviations, true statements, which topic), plus a focused test on one topic.
- **My progress**: one tab per material with quiz scores, topics, oral checks and a "focus next" suggestion.

**For professors**
- **Concepts to Revisit**: which slide topics the class seems to struggle with, ranked by a **confusion score** that
  combines several kinds of evidence (who asked, who came back or asked for an explanation, class quiz results, oral
  checks, whether it kept coming up across days) and shows that evidence, how confident the score is, and how much the
  ranking changes if the weights change. Many questions alone do not rank a topic high. Under the ranking, the
  **Content Advisor** suggests what to do about the top topics (rule-based, citing its evidence) and can draft a
  clarification note or a remedial quiz with the local model. Drafts stay hidden from students until the professor
  reads, edits and approves them.
- **Class Insight**: student questions grouped by meaning, with the slides students keep landing on.
- **Pending Gaps**: questions the notes could not answer. One answer is added to the course material for everyone.
- **Class Quizzes**: quizzes on the topics the class keeps asking about, with aggregated (anonymous) results,
  plus a **Question analysis** that shows the wrong answers many students chose (likely misconceptions) and
  flags weak quiz questions. Numbers appear only once at least 5 students have taken a quiz.
- **Course Material** (upload PDFs, rebuild the index).

## The agents

Nine agents with different jobs, coordinated by an orchestrator that routes each message and hands work
between them. The app shows which agents ran in an "Agents involved" panel.

| Agent | Job |
|---|---|
| Doubt Resolver | Answer a question from the notes, with citations |
| Concept Explainer | Teach a concept step by step (beginner or detailed) |
| Quiz Generator | Write multiple-choice questions strictly from the notes |
| Note Generator | Write heading-wise study notes and export them as PDF |
| Oral Assessor | Ask a spoken question on a heading, score the answer's content and fluency, and list the points missed |
| Progress Tracker | Per-student strong and weak topics from questions and quiz scores |
| Faculty Insight | Cluster the class's questions; find topics the class keeps asking about |
| Content Advisor | Suggest what to change for the topics the class finds hardest; draft a clarification note (professor approves) |
| Gap Handler | Queue a question the notes cannot answer for the professor |

Routing uses a small trained classifier (with hand-written rules as a fallback) that learns from corrections
made in the app. How it all fits together is in [docs/DESIGN.md](docs/DESIGN.md).

## Quick start

### Requirements

- **Python 3.10 or newer** (developed and tested on 3.13 on Windows 11)
- **[Ollama](https://ollama.com/download)** installed and running
- About **10 GB of free disk** (Llama 3 is 4.7 GB, the optional voice model about 1 GB, plus Python packages)
- At least 8 GB of RAM (Llama 3 alone uses about 5 GB while it runs). A GPU is optional: on a CPU, an
  answer takes about a minute and a quiz or a note topic 2 to 3 minutes.

### Install and run

```bash
# 1. Get the code
git clone https://github.com/Jiya1101/multi-agent-academic-assistant.git
cd multi-agent-academic-assistant

# 2. Create a virtual environment and install the packages
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

# 3. Download the language model (one time, 4.7 GB)
ollama pull llama3

# 4. Optional: download the speech model for voice input (one time, about 970 MB)
python download_speech_model.py

# 5. Add your course PDFs to the data/ folder, then build the search index
python ingest.py

# 6. Start the app, then open http://localhost:8501
streamlit run app.py
```

Notes:
- PDFs must contain real text. **Scanned or image-only PDFs are not supported** (there is no OCR).
- The first run also downloads a small embedding model (`all-MiniLM-L6-v2`, about 90 MB) automatically.
- You can also upload PDFs from the app (Professor view, Course Material tab) and click **Build / Rebuild Index**
  instead of running `ingest.py`.
- If a newer release of a library breaks something, install the exact versions the project was tested with:
  `pip install -r requirements-tested.txt`.

### Try it with demo data

A single laptop is not a class, so the professor dashboard starts empty. To simulate one:

```bash
python seed_demo.py              # adds a simulated class of students asking questions
python publish_quizzes.py        # writes class quizzes on the topics they asked about (takes minutes)
python seed_demo.py --quiz-attempts   # 24 simulated students take those quizzes (feeds Question analysis)
python seed_demo.py --class-activity  # the same students ask questions over several days (feeds Concepts to Revisit)
python seed_demo.py --clear      # removes the simulated questions and quiz attempts again
```

Then switch the sidebar to **Professor**. [DEMO.md](DEMO.md) is a 10-minute presentation script with the
questions a panel is likely to ask.

### Run the tests

```bash
python -m unittest discover -s tests
```

The tests use a small synthetic index and a fake chat model, so they take seconds and need neither Ollama nor
any PDFs.

## Using the app

Pick **Student** or **Professor** in the sidebar (a demo switch: there is no login yet). Students can enter an
optional Student ID so their progress is remembered; leaving it blank keeps them anonymous.

Things to try in the **Ask** box:

| Say or type | What happens |
|---|---|
| `What is the difference between ACID and BASE?` | Doubt Resolver answers with citations |
| `explain BASE simply` | Concept Explainer teaches it step by step |
| `quiz me on RDS read replicas` | Quiz Generator writes a quiz you can take |
| `make notes on ACID and BASE` | Note Generator builds notes with a PDF download |
| `how am I doing` / `what should I study next` | Progress Tracker (needs a Student ID) |
| a question the notes do not cover | "Not covered", and it is queued for the professor |

If the router picks the wrong kind of help, open **Wrong kind of help?** under the answer and correct it; the
router learns from it.

For voice, press the microphone button in the Ask tab, speak, and stop. The text appears in the message box for
you to check before pressing **Ask**. The browser will ask for microphone permission.

## Configuration

Settings live in [`rag_core/config.py`](rag_core/config.py).

| Setting | Default | Meaning |
|---|---|---|
| `OLLAMA_MODEL` | `llama3` | Chat model; any model you have pulled with Ollama |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Where Ollama is listening |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Embedding model for search |
| `SPEECH_MODEL_NAME` | `openai/whisper-small.en` | Whisper model for voice input |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | How notes are split for search |
| `TOP_K` | `4` | Passages retrieved per question |
| `RELEVANCE_SCORE_THRESHOLD` | `1.15` | How close a passage must be for the notes to count as covering a question |
| `CLUSTER_DISTANCE_THRESHOLD` | `0.55` | How similar questions must be to group together |

The relevance and clustering thresholds were tuned on a few dozen questions from two documents; re-check them
if you index very different material.

## Troubleshooting

| Problem | Fix |
|---|---|
| Connection refused / "could not connect" when asking | Ollama is not running. Start the Ollama app, or run `ollama serve`. |
| `model "llama3" not found` | Run `ollama pull llama3`. |
| "No course material has been indexed yet" | Add PDFs to `data/` and run `python ingest.py` (or use the Course Material tab). |
| "No extractable text was found" | The PDF is scanned or image-only. Use a text-based PDF. |
| Answers are slow | Normal on a CPU. A GPU makes Ollama much faster. |
| Voice: "Could not transcribe" | Run `python download_speech_model.py` once, then try again. |
| Voice: the recorder does nothing | Allow microphone access for `localhost` in the browser; try a normal browser window. |
| Port 8501 is in use | `streamlit run app.py --server.port 8502` |
| A new library version breaks the app | `pip install -r requirements-tested.txt` |

## Project layout

```
app.py                   Streamlit app: Student view and Professor dashboard
ingest.py                Build the search index from the PDFs in data/
seed_demo.py             Simulate a class of students for demos
publish_quizzes.py       Pre-generate class quizzes from the command line
evaluate_router.py       Measure the router (rules vs learned vs hybrid)
evaluate_item_analysis.py  Check the quiz question analysis against simulated classes with known problems
evaluate_confusion.py    Check the confusion score against simulated classes with known confusing topics
download_speech_model.py One-time download of the Whisper model
agents/                  The multi-agent layer (agents, orchestrator, router)
rag_core/                Shared retrieval layer (loading, search, LLM, PDF export, speech)
tests/                   Unit tests
docs/DESIGN.md           How it works, measurements, limitations
DEMO.md                  Presentation script
data/                    Put your course PDFs here (kept out of git)
db/                      Search index and question log (generated, kept out of git)
```

## Results in brief

Measured on this project's own data; full details and caveats are in [docs/DESIGN.md](docs/DESIGN.md).

| What | Result |
|---|---|
| Router, 96 held-out messages | learned 96.9%, hybrid used by the app 95.8%, hand-written rules 52.1% |
| Voice transcription, 8 recordings | about 3% word error; course acronyms spelled correctly 16 of 16 times |
| Study notes (real Llama 3, 2 topics) | right structure every time; unsupported details are dropped or flagged for checking |
| Quiz question analysis, 200 simulated classes of 24 | planted shared mistakes found 92% of the time, wrongly reported on ordinary questions 0.04 times per class; faulty answer keys found only 35% of the time (97% with 100 students) |
| Concept confusion score, 200 simulated classes of 24 (2 of 8 topics planted as confusing, 1 as merely popular) | picks the confusing topics 82 to 100% of the time depending on how strongly confusion shows in behaviour; ranking by number of questions manages 50 to 69% (chance is about 25%) because it picks the popular topic |
| Tests | unit tests for every module (command above) |

These come with real caveats: the router examples and the test recordings were made by the project authors
(the recordings with Windows text-to-speech voices), so they show behaviour on new wording and clean audio, not
on real students and real microphones. The quiz analysis and the confusion score were checked on simulated students generated by models
the authors wrote, so they show how the methods behave when the effects are the size we assumed, not on a real class.
The confusion weights are judgment calls, and the score has not been compared with real exam results.
The quiz and note generators can still write weak or unsupported
statements; the app flags the risky ones, and a person should read the output before relying on it.

## Limitations

- No OCR, so scanned PDFs do not work.
- No real login: the Student/Professor switch and the Student ID are not authentication.
- Routing and the relevance thresholds are heuristics tuned on small data.
- Quizzes and notes are written by a small local model and need a read-through.
- English only for voice input.
- Not yet tested with real students or professors.

## Built with

[Streamlit](https://streamlit.io), [LangChain](https://www.langchain.com) components, [Ollama](https://ollama.com)
with Llama 3, [FAISS](https://github.com/facebookresearch/faiss), [sentence-transformers](https://sbert.net)
(`all-MiniLM-L6-v2`), [OpenAI Whisper](https://github.com/openai/whisper) via
[Transformers](https://huggingface.co/docs/transformers), scikit-learn, [ReportLab](https://www.reportlab.com), pypdf.

## License

No license has been chosen yet, so by default all rights are reserved. Add a `LICENSE` file (for example MIT)
to let others use and modify the code.
