# Demo walkthrough (about 10 minutes)

## Setup (before the panel arrives)

```bash
ollama pull llama3          # once
python ingest.py            # build the index from data/
python seed_demo.py         # simulate a class of students asking questions
python publish_quizzes.py   # pre-generate class quizzes (a few minutes on CPU)
streamlit run app.py
```

`seed_demo.py` adds simulated questions to the log using the real retrieval
and relevance gate, plus one identified student, `demo_student`, whose
questions pile up on ACID/BASE. Say so openly: the dashboard needs a class's
worth of data to be worth showing, and one laptop is not a class. Remove it
afterwards with `python seed_demo.py --clear`.

Llama 3 runs on CPU here: a live answer takes about a minute and a quiz about
2 to 3 minutes. Ask, then talk through the "Agents involved" panel while it generates.

## The pitch (30 seconds)

"Most study assistants are one chatbot that only helps the student. We built
seven agents with different jobs, and an orchestrator that routes between them
and hands work from one to another. The same system serves the professor: it
shows where the class is confused, turns that into quizzes, and turns
questions the notes cannot answer into new course content."

## 1. Student: a grounded answer (1.5 min)

Sidebar: **View as: Student**. Student ID: **demo_student**.

Ask: **What is the difference between Multi-AZ and read replicas in RDS?**

Point out:
- **Agents involved** shows `Router -> doubt`. The Doubt Resolver answered
  with `[Chunk N]` citations, and the chunks are listed with file, page and
  slide title.
- Everything is local: Llama 3 through Ollama, MiniLM embeddings, FAISS.

## 2. Student: the notes cannot answer, so another agent takes over (1 min)

Ask: **When is assignment 2 due?**

Point out:
- **Agents involved** now shows `Doubt Resolver -> Gap Handler`. No LLM call
  was made: a retrieval-distance check refused to guess.
- The Gap Handler reports that other students already asked something
  similar, so the professor sees one priority topic, not four separate rows.

## 3. Student: the Progress Tracker feeds the Quiz Generator (2.5 min)

Ask: **quiz me**

Point out:
- No topic was given. **Agents involved** shows
  `Progress Tracker -> Quiz Generator: weakest topic is ...`. The tracker
  found that `demo_student` keeps asking about ACID/BASE and picked it.
- Take the quiz and submit. Each answer shows an explanation and the slide
  the question was written from. The score is saved to this student.
- Under 80 percent, an **Explain this topic** button appears (Concept
  Explainer). Open the **My progress** tab to show strong and weak topics.

To show the router learned rather than matching keywords, type a wording no
rule knows, such as **can you test my knowledge of RDS read replicas** or
**which area should I improve first**. The first line of **Agents involved**
reads `Router -> quiz (learned, NN% sure)`. If it ever picks wrongly, open
**Wrong kind of help?** and correct it; the router retrains on that.

Then ask **what should I study next**: `Progress Tracker -> Concept Explainer`
runs automatically on the weakest topic.

## 3a. Student: ask by voice (1 min)

In the **Ask** tab press the microphone button, say **What is the difference
between ACID and BASE**, and press stop. The text appears in the message box
within a few seconds (it was transcribed on this laptop, nothing uploaded).
Press **Ask**. Mention that course acronyms are spelled correctly because the
recogniser is primed with terms taken from the uploaded notes. Use the browser
at `http://localhost:8501`, allow the microphone when asked, and try it once
before the demo: the first use loads the speech model (a few seconds).

## 3b. Student: study notes as a PDF (2 min, pre-generate if short on time)

Open the **Study notes** tab. Pick a topic such as **ACID DATABASE CONSISTENCY
MODEL** (or type one), click **Generate notes** and watch the progress bar
(about 2 minutes per topic on CPU). You can also type **make notes on ACID and
BASE** in the Ask box; the trace then shows `Router -> notes (learned, ...)`.

Point out the layout (heading, definition, key points, use case, source
pages), then click **Download notes as PDF** and open the file. Say plainly:
statements marked with an asterisk contain wording that is not in the
slides, and an unsupported example is replaced by "The notes do not give an
example". Notes on a topic the material does not cover are refused and
reported to the professor.

## 4. Professor: Faculty Insight to Quiz Generator (2.5 min)

Sidebar: **View as: Professor**.

**Class Insight**: questions are grouped by meaning, not by wording. "ACID vs
BASE", "Explain the ACID properties" and "Why was BASE introduced after ACID?"
are one topic of confusion. The chart shows which slides students land on.

**Class Quizzes**: click **Generate quizzes from class confusion**.
- The trace shows `Faculty Insight -> Quiz Generator` for each topic the
  class keeps asking about (it skips topics already published, so it is
  instant if you pre-generated).
- Each published question names its source slide so the professor can check
  it before students rely on it.
- **Class results by topic** aggregates quiz scores without naming students.
  Topics under 60 percent are flagged for re-teaching.

## 5. Professor: closing the loop (1.5 min)

Tab **Pending Gaps**: the unanswered questions are grouped by topic. Answer
the assignment group, for example: "Assignment 2 is due Friday 11:59 PM on
the course portal." Click **Add to knowledge base**.

Switch to Student and ask **What is the submission date for assignment 2?**
(a phrasing nobody used before). It is now answered and the chunk is labelled
"Faculty answer".

## Questions the panel is likely to ask

**Is this just a chatbot?** No. The chatbot is one agent. The system is six
agents with separate jobs and tools, and the hand-offs between them
(visible in the trace) are the point.

**Is it really multi-agent?** The agents are separate units with their own
tools, coordinated by an orchestrator that passes one agent's output to
another. The honest limit: the agents do not negotiate or plan on their own.
It is an orchestrated pipeline of specialised agents, not autonomous agents.

**Is the routing learned or hard-coded?** Learned. A classifier trained on
labelled example messages decides from the meaning and the words of a
message, with hand-written rules as a fallback when it is unsure. On 80
held-out messages it scored 96.9 percent against 52 percent for the rules
(`python evaluate_router.py --test`). Be upfront that we wrote the examples
ourselves, so this measures new wording, not real students, and the test set
leans towards requests the rules were never written for. Show the "Wrong kind
of help?" control: corrections are stored and the router retrains on them.

**How do you stop hallucination?** A retrieval-distance gate refuses
out-of-scope questions before the model is called (unit-tested with a call
counter); the prompts require `[Chunk N]` citations; quiz questions must name
a real retrieved chunk and are validated; retrieved chunks are shown with
every answer.

**Why local models?** Privacy of course material and student questions, zero
running cost, works offline.

**What are the limits?** Say them before they are asked:
- The Student/Professor switch and the Student ID are not authentication.
- Scanned PDFs (images with no text layer) are not supported; there is no OCR.
- The relevance cutoff (1.15), the keyword rescue and the clustering
  threshold (0.55) were tuned on a few dozen questions from two documents.
- Quiz questions are written by a local 8B model. They are validated and
  traceable to a slide, but in testing some had weak distractors and one had
  two defensible answers, so a professor should review them.
- Llama 3 on CPU takes about a minute per answer and 2 to 3 minutes per quiz. A GPU would make
  it near instant.
- Not tested with real students or professors yet.
