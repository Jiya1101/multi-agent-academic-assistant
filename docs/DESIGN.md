# Design notes and measurements

How the Academic Assistant works inside, what was measured, and where it falls short.
For setup and everyday use, see the [README](../README.md); for a presentation script, see [DEMO.md](../DEMO.md).

## The agents

| Agent | Job | Tools |
|---|---|---|
| Doubt Resolver | Answer a question from the notes, with `[Chunk N]` citations | retrieval + relevance gate, Llama 3, question log |
| Concept Explainer | Teach a concept step by step (beginner or detailed) | retrieval + relevance gate, Llama 3 |
| Quiz Generator | Write multiple-choice questions strictly from the notes | retrieval, Llama 3 (JSON mode), quiz tables |
| Note Generator | Write heading-wise study notes (definition, key points, use case) as a downloadable PDF | retrieval, Llama 3 (JSON mode), PDF export |
| Progress Tracker | Per-student strong/weak topics from questions and quiz scores | question log, quiz attempts (no LLM) |
| Faculty Insight | Cluster the whole class's questions; find topics the class keeps asking about | embeddings, clustering, question log |
| Content Advisor | Suggest what to change for the topics the class finds hardest; draft a clarification note | concept scores, quiz analysis, retrieval, Llama 3 (draft only), advice log |
| Gap Handler | Queue a question the notes cannot answer for the professor; spot repeats | question log, embeddings |

An **Orchestrator** routes each request (`agents/orchestrator.py`) and shows
which agents ran, and how the message was routed, in an "Agents involved"
panel.

### The router learns

Deciding which agent should handle a message is done by a **learned router**
(`agents/routing.py`), not just keyword rules:

- A classifier (logistic regression) is trained on labelled example messages
  (`agents/router_data.py`). Its inputs are the message's sentence embedding
  (its meaning) plus its words (TF-IDF), so "can you test my knowledge of
  RDS" is recognised as a quiz request although no rule mentions it.
- When the classifier is unsure (confidence under 40 percent), hand-written
  rules take over. No LLM call is involved, so routing is instant.
- **It keeps learning.** Under each answer, "Wrong kind of help?" lets a
  student say what they wanted. The correction is stored and the classifier
  is retrained on it at the next message.

There are six routes: answer a question, explain, quiz, progress, study next,
and make notes. Measured with `python evaluate_router.py --test` on 96
held-out messages (never used for training or tuning):

| Router | Accuracy |
|---|---|
| Hand-written rules | 52.1% |
| Learned classifier | 96.9% |
| Hybrid used by the app | 95.8% |

Read these numbers with care. Both the training and test messages were
written by the project authors, not collected from students, so they show
how well the router handles new wording, not how it will do on real traffic.
The test set is also weighted towards requests the rules were never written
for; real students send mostly plain questions, which the rules already
handle (about 95 percent on those). The rule for the notes route was written
by the same authors who wrote the test phrasings, which flatters the rules on
that route (86 percent there). Cross-validation on the training set gives
about 90 percent for the learned router, so expect something between the two
on unseen wording. The hybrid scored slightly below the learned classifier
alone here, a one-message difference that is within noise; the rule fallback
is kept as a safety net, not because the data shows it helps.

### Study notes as a PDF

The **Study notes** tab (students only) generates notes for up to five topics at a time. Pick topics from
the slide titles in the uploaded files, or type your own, or just ask in chat:
"make notes on ACID and BASE". Each topic becomes:

- a **heading**, a short **definition**, **key points** as bullets, and a
  **use case**, all in simple language;
- a **source line** naming the file and pages it came from;
- a **Download notes as PDF** button.

Notes come only from the uploaded material. A topic the material does not
cover is skipped and reported as a gap for the professor, not written from the
model's own knowledge.

**Slide decks written as bullets.** The whole-material notes and the oral check
first read a heading's text as full sentences. A deck made of terse bullets
("Pros: simple, easy.", "Token circulates -> only the holder enters.") has almost
none, so both used to find nothing (the oral check even said "not covered" about
material it simply could not read as prose). When no sentences are found, the
bullets are now read as points (`bullet_points` in `rag_core/normalize.py`):
text before the first bullet is skipped as headings, sub-headings and icon
labels glued to the end of a bullet are removed, and text with no bullet glyphs
is left to the sentence reader, so prose decks behave as before. For the oral
check the model answer is capped at 6 points, the ones sharing most words with
the question. That ranking is plain word overlap, so a loosely related point can
slip in; treat the model answer as a guide. Found and fixed on a real
terse-bullet file (`MUTUAL EXCLUSION.pdf`): the oral check now writes a question
in 10 to 30 seconds and the notes give 2 sections instead of 0.

Checks on what the model writes (it is a small local model and does drift):
output must be valid JSON in the expected shape (one retry); key points that
share too little wording with the source are dropped and counted; an example
with no support in the source is replaced by "The notes do not give an
example"; and any remaining statement containing two or more words that are
absent from the source is marked with an asterisk and a "check this" legend,
in the app and in the PDF. Sources are attributed from the text itself, since
the model's own page citations were wrong in testing.

What testing with real Llama 3 showed (two topics from `Module 4.pdf`): it took
about 95 to 110 seconds per topic; the structure was right every time; and the
first prompt let the model add details that are not in the slides ("such as
financial transactions", "business intelligence"). Adding a rule against
outside examples to the prompt removed those in the next run. We also tried a
second Llama 3 pass to verify each statement against the source; on ten
statements it caught two of three real additions but wrongly flagged two
supported ones (about 50 percent precision), and added about a minute per
topic, so it was not adopted. These are small samples; notes still need a
read-through before they are relied on.

### Voice input

In the **Ask** tab, press the microphone button, speak, and press stop. The
question is transcribed on your computer by OpenAI's Whisper
(`whisper-small.en`, set in `rag_core/config.py`) and appears in the message
box. Read it, fix any mistake, then press **Ask**; or tick **Ask right after
transcribing** to skip that step. Nothing is uploaded: the recording stays on
the machine. First use needs the one-time model download (`python
download_speech_model.py`, about 970 MB), and the browser asks for microphone
permission.

Around the model there are some safeguards, because Whisper invents words
(such as "Thank you.") when it is given silence or noise: recordings that are
too short, too quiet, or noise-like (a spectral-flatness check) never reach
the model, and a few phantom phrases are discarded. The model is also given a
list of your course's acronyms, found from how the notes themselves write
them (ACID, OLTP, RDS, YARN), so it writes "ACID" instead of "acid".

Measured on 8 spoken test recordings (`rag_core/speech.py`, scored against
what was said): about 3 percent word error, 2 to 3.5 seconds per short
question and about 9 seconds for a 35-second recording on this CPU. Course
acronyms came out in the correct form 16 of 16 times with the vocabulary hint,
against 8 of 16 without it; the only wrong word was "red" for "read". Silence,
a short click, quiet hiss and loud white noise were all rejected.

Limits to state plainly: the test recordings were made with the two built-in
Windows text-to-speech voices, which are much cleaner than a human voice on a
laptop microphone, so real accuracy will be lower, especially with background
noise or a strong accent. The thresholds for silence and noise were set on the
same synthetic audio. The model is English only. The browser recording itself
(microphone permission, device choice) could not be tested here; everything
after the recording was.

### What the question log records

Each question stores more than its text, so a professor-side analysis can use behaviour and not just topics:

- **agent**: which agent handled it (Doubt Resolver, Concept Explainer, Note Generator), so explanation requests
  can be told apart from plain questions;
- **session id**: a random token per browser session. It is not tied to a person, so an anonymous student's
  questions can still be linked together;
- **follow_up_of**: set when the same session already asked about the same slide topic within 30 minutes
  (`FOLLOW_UP_WINDOW_MINUTES`). This measures "came back to the same topic", a proxy for a clarifying follow-up,
  not proof of one. Unanswerable and sessionless questions are never marked as follow-ups.

Databases from before this change are upgraded on first open; older rows simply have no agent or session.

### Quiz question analysis (professor)

`rag_core/item_analysis.py` looks at each quiz question and each answer option once a quiz has been taken by at
least 5 different students (`MIN_COHORT`; below that no per-question numbers are computed at all). Only each
student's first attempt counts, and a skipped question counts as wrong.

| Output | How it is decided |
|---|---|
| Likely misconception | a wrong option chosen by at least 25% of the class (and 3 students), and more often than chance would give it if wrong answers spread evenly over the wrong options (binomial test, adjusted for the number of wrong options, p < 0.01) |
| Too easy / very hard | 90% or more / 30% or less correct |
| Check the answer key | students who did well on the other questions tend to get this one wrong: correlation of -0.2 or lower with p < 0.05, needing 10+ students and at least 5 on each side |
| A wrong answer beat the right one | a reported misconception drew more students than the correct answer |
| An option nobody chose | 10+ students, a wrong option with zero picks, on a question that is not too easy |

How it was tested: unit tests with planted patterns (including mutation checks that disable a rule and confirm
the right test fails), and `python evaluate_item_analysis.py`, which simulates 200 classes per size and compares
the report with the planted truth:

| Class size | Planted shared mistake found | Wrongly reported, per class | Faulty key found | Wrong key alarms, per class |
|---|---|---|---|---|
| 8 | 25% | 0.000 | 0% | 0.000 |
| 12 | 49% | 0.005 | 17% | 0.010 |
| 24 | 92% | 0.035 | 35% | 0.010 |
| 48 | 100% | 0.030 | 65% | 0.005 |
| 100 | 100% | 0.015 | 97% | 0.015 |

Read this with care:

- The students are generated by a simple model we wrote, with effects of a size we chose, so this shows how the
  method behaves under those assumptions. It says nothing yet about real classes.
- The thresholds were adjusted after looking at one simulated class of 24. A first version flagged ordinary
  questions as misconceptions by luck (several wrong options reached 29%), which is why the chance test and the
  stricter p < 0.01 were added. The 24 row is therefore the least independent one.
- The answer-key check compares a question with the student's other questions, so it is unreliable if many of
  those are also faulty, and it is weak in small classes (35% at 24 students). "Nothing flagged" never means
  "the questions are fine", and "no wrong answer stood out" never means "there is no misconception".
- It finds patterns in answers. Whether a popular wrong answer is a real misconception, a badly worded option, or a
  faulty key is the professor's call, and the dashboard says so.

### Concept confusion (professor)

`rag_core/confusion.py` ranks slide topics by how likely the class is struggling with them. The reasoning: 40
students asking about deadlock does not mean they are confused, they may just be curious. So no single count is
used. Seven kinds of evidence about the same concept are combined, and the dashboard (**Concepts to Revisit**)
shows each one next to the score.

| Signal | Reads as | Needs |
|---|---|---|
| Share of the class who asked | breadth (full marks at a quarter of active students) | 5+ identified students |
| Students who asked more than once | repeat | 5+ identified students |
| Questions that were follow-ups | came back to the topic in the same session (full marks at half) | 5+ questions with a session |
| Requests to explain it | asked the Concept Explainer instead of the Doubt Resolver (full marks at half) | 5+ questions with an agent |
| Students who missed it on the class quiz | 1 minus the pooled first-attempt score | 5+ students on class quizzes |
| Oral checks not yet strong | share of first oral checks below "Strong" | 5+ students |
| Asked about on several days | persistence (full marks at 3 different days) | 5+ questions |

Default weights are 15, 10, 15, 10, 30, 10 and 10 percent; the quiz is heaviest because it is the only direct
performance measure. The professor can change every weight on the page.

Design decisions that matter:

- **Missing evidence is never read as "no confusion".** A signal without enough data is replaced by the class
  average for that signal, so a concept is neither rewarded nor punished for having less data. Confidence says how
  many kinds of evidence (questions, quizzes, oral checks) actually back the score: Low, Medium or High.
- **People-level signals need 5 different students** (the same rule as the quiz analysis).
- **Anonymous questions count as questions, never as students.** Most of the old demo questions have no identity.
- **Questions the notes could not answer are not in the score.** They match no slide, so they cannot belong to a
  concept; they stay in Pending Gaps.
- **Ranking stability:** the ranking is recomputed under 300 randomly perturbed weight sets, and the page shows how
  often each concept stays in the top 3. A concept that is first only under one weighting is a weak finding.
- **A concept is one slide topic.** "ACID" and "BASE" slides are separate concepts; merging slides about one idea
  is not done yet.

How it was tested: 25 unit tests on hand-built classes (a merely popular topic must not outrank a confusing one;
missing data must not be read as zero; retakes and personal quizzes must not count; small groups must stay hidden),
with deliberate breaks of four rules to confirm the tests notice. `python evaluate_confusion.py` simulates 200
classes per setting with 8 concepts, 2 planted as truly confusing and 1 planted as merely popular. Precision@2 is
the share of the top 2 that are the confusing ones:

| Setting (class of 24) | Number of questions | Different students asking | Quiz only | Composite, no quiz | Composite |
|---|---|---|---|---|---|
| Weak effect of confusion on behaviour | 0.49 | 0.48 | 0.75 | 0.71 | 0.82 |
| Moderate effect | 0.52 | 0.49 | 0.98 | 0.95 | 0.99 |
| Strong effect | 0.69 | 0.51 | 1.00 | 1.00 | 1.00 |
| Moderate effect, only 20% take each quiz | 0.54 | 0.48 | 0.54 | 0.95 | 0.97 |

Read this with care:

- The students come from a model we wrote. In it, confusion causes follow-ups, explanation requests, repeat days,
  quiz misses and weak oral checks, and the "popular" topic gets more one-off questions than the confusing ones.
  So the table shows that, IF students behave like that, combining signals beats counting questions. It does not
  show that real students behave like that, and the counting baselines look bad partly because we built a popular
  decoy. If confusion produced most of the questions, counting would do better (0.69 in the strong case).
- The honest summary of the result is narrower than "the composite wins": when most students take the quizzes,
  the quiz result alone is nearly as good (0.98 vs 0.99). What the other signals add is robustness: they carry the
  ranking when few students take quizzes (0.97 vs 0.54), when the effects are weak (0.82 vs 0.75), and when there
  is no quiz at all (0.71 to 0.95 without it, against about 0.5 for counting).
- Small classes are hard: with 12 students and weak effects the composite gets only 0.68.
- The weights, the saturation points (a quarter of the class, half the questions, 3 days) and the 5-student
  minimum are judgment calls. In the demo data the breadth signal saturates for almost every concept, so it adds
  little there.
- Nothing here has been compared with real exam results or a professor's own judgment. That comparison is what
  would turn a ranking aid into an evidence-backed measure.

### Content Advisor (professor)

Turns "the class is confused about X" into things a professor can do, in two separate steps.

**1. Diagnosis: fixed rules, no language model** (`rag_core/advice.py`). For the top-scoring concepts, each rule
fires only on stated evidence and names it:

| Finding | Fires when |
|---|---|
| Check quiz question N | a quiz question that strong students tend to miss (the "check the answer key" flag): fix the question before re-teaching |
| Many students share one mistake | a reported quiz misconception |
| Students may be missing an earlier idea | another high-scoring concept sits earlier in the same file (a guess from slide order, and it says so) |
| Students keep returning | follow-up or explanation signal of 50% or more, with the slide they land on |
| Recognise but cannot explain | quiz miss rate 40% or less while 60% or more of oral checks are not yet strong |
| Wording matches the slide poorly | average retrieval distance at or above 80% of the relevance cut-off, with the students' own phrasings |
| Only one kind of evidence | confidence is Low: collect more before changing anything |

It also lists what students actually ask (grouped by meaning) and which slides they land on.

**2. Drafts: the local model, only when the professor asks.** A clarification note is written from the course
material found by searching the concept AND the confusing quiz question (the question can be about a neighbouring
idea), with the instruction naming the correct answer and the wrong one. Sentences with too little support in the
material are dropped, sentences with two or more words absent from the material are flagged, citation markers and
chatty openings are removed, and the professor's edits are re-checked each time the box is updated. A remedial quiz comes from the
Quiz Generator through the Orchestrator (Faculty Insight -> Content Advisor -> Quiz Generator).

**Nothing reaches students unapproved.** Drafts live in the advice log. A clarification enters the course index only
when approved (and can be removed again); a remedial quiz is saved with a hidden "draft" scope until published.
Approved clarifications are re-added on an index rebuild. An approved clarification is stored under the concept's
own title, not a new heading: otherwise students' later questions would land under "Clarification" instead of the
concept, and its confusion score would fall because of the relabelling and not because students understood more.
Approval times are recorded so the effect of an action on the score can be measured later.

How it was tested: 37 unit tests (each rule on a hand-built class, the advice log, the draft/publish gate, the
index, the draft checks with a fake model, the Orchestrator hand-offs), deliberate breaks of six promises (five
caught; the sixth was a redundant sort), and a scripted click-through of the real dashboard on a copy of the
database. The language-model part was run three times on your real data with Llama 3, and the runs were not equally
good, which is the reason for the approval step:

- Run 1 (searched only the concept title): 169 seconds. It opened with "Here is a clarification ...", and explained
  ACID while the class's mistake was about BASE, even repeating the wrong answer as if related. The support check
  flagged 3 sentences.
- Run 2 (after adding the quiz question to the search and removing chatter): 79 seconds, two generic sentences that
  ignored the mistake entirely.
- Run 3 (after putting the correct and wrong answers into the instruction itself): 38 seconds, on target. It explains
  that BASE gives consistency eventually, contrasts it with ACID, and the check dropped 2 shaky sentences and flagged 1.

Three runs on one concept is a very small sample, and each fix was made after seeing the previous output. The honest
conclusion is that a small local model can produce a usable first draft when the instruction is specific, that it
sometimes does not, and that the professor must read every draft. The lexical support check catches invented
content words but not a fluent sentence that is wrong or beside the point (run 2 passed it).

Limits: the rules are heuristics with thresholds we chose; "earlier in the file" is not a real prerequisite
relation; a concept is one slide topic, so a quiz about BASE can end up under the ACID concept; nothing yet measures
whether an approved action helped (that needs real students before and after); and there is no way to edit a
published remedial quiz.

### Hand-offs between agents

- **Doubt Resolver / Concept Explainer / Note Generator -> Gap Handler**: the
  notes do not cover the question or topic. The professor sees it grouped
  with similar ones.
- **Progress Tracker -> Quiz Generator**: "quiz me" with no topic quizzes the
  student's weakest topic.
- **Progress Tracker -> Concept Explainer**: "what should I study next"
  explains the weakest topic.
- **Faculty Insight -> Quiz Generator** (professor): topics the notes cover
  but many students keep asking about become published class quizzes. Quiz
  results then show the professor which topics need re-teaching.
- **Faculty Insight -> Content Advisor** (professor): the scored concepts become suggestions per concept.
- **Content Advisor -> Quiz Generator** (professor): a remedial quiz on the class's shared mistake, saved as a hidden
  draft until the professor publishes it.
- **Professor answer -> index**: a professor's answer to a gap is added to the
  index, so every future student gets it, cited as a "Faculty answer".
  Rebuilding the index re-adds these answers automatically.

### Privacy

A Student ID is optional. Without one, questions are logged with no
identifier (only a random session token, so follow-ups can be linked). With
one, the Progress Tracker can remember that student's questions and quiz
scores. Professors only see class-wide patterns: clustered questions,
aggregated quiz results, per-question answer counts (only once 5 or more
students have taken the quiz), and unanswered questions as text. Student IDs
are never shown on the professor dashboard. There is no login yet; the Student/Professor switch is a demo toggle.

## How it works

1. **Extract** — `PyPDFLoader` pulls text out of each PDF, page by page.
2. **Normalize** — `rag_core/normalize.py` cleans up common PDF-extraction
   artifacts before chunking: stray bullet glyphs (`o`, `•`, ...), words
   hyphenated across a line wrap, punctuation glued to the next word
   (`categories:Relational`), case-boundary glued words (`isMySQL` ->
   `is MySQL`), and long all-lowercase glued runs (`andrecords` ->
   `and records`, via `wordninja`). It also tags each page with a
   best-effort `section` title (its first substantial line) so chunks can
   be cited by slide/section, not just page number. This is heuristic
   cleanup, not perfect — see **Known limitations** below.
3. **Chunk** — `RecursiveCharacterTextSplitter` splits the cleaned pages
   into ~1000-character overlapping chunks along natural boundaries.
4. **Embed** — each chunk is embedded with
   `sentence-transformers/all-MiniLM-L6-v2` via `langchain-huggingface`.
5. **Store** — embeddings are indexed in a local **FAISS** vector store and
   persisted to `db/`.
6. **Retrieve** — a user query is embedded and matched against the FAISS
   index to fetch the top-k most similar chunks *with similarity scores*.
7. **Relevance gate** — if even the closest chunk is below the similarity
   threshold (`RELEVANCE_SCORE_THRESHOLD` in `config.py`), the question is
   answered "not covered by the notes" **without calling the LLM at all**.
   This is a deterministic backstop: small local models don't reliably
   self-enforce "don't answer from outside the context" through prompt
   instructions alone.
8. **Generate** — otherwise, the retrieved chunks are inserted into a
   strict-grounding, citation-enforcing prompt (chunk-tagged, negative-
   constraint-aware, structured pedagogical format) sent to **Llama 3** via
   **Ollama**, which produces the final answer.

## Known limitations

- **OCR is out of scope.** Scanned/image-based PDFs (no text layer) yield
  zero chunks and are reported as such, not indexed.
- **Text normalization is heuristic, not perfect.** On messy source PDFs
  (e.g. slide decks exported with no real spacing) you may still see
  occasional artifacts: a genuine long word incorrectly split ("recover
  ability"), a short glued word left untouched ("byusing"), or a word
  broken by a stray mid-word line break the extractor introduced with no
  hyphen to detect ("well-architect ed"). These are documented trade-offs
  in `rag_core/normalize.py`, not silent failures.
- **Answer quality depends heavily on the LLM.** Small models (tried during
  development: `qwen2.5:0.5b`) ground their retrieval correctly but don't
  reliably follow every prompt instruction (like inline `[Chunk N]`
  citations, or the negative-constraint "say so if it's not covered"
  rule — which is why the relevance gate above exists as a deterministic
  backstop instead of relying on the prompt alone). `llama3` follows
  instructions noticeably better; swap `OLLAMA_MODEL` in `config.py` to try
  a different local model.

- **The relevance gate is a heuristic.** A retrieval distance cutoff
  (`RELEVANCE_SCORE_THRESHOLD`) decides whether the notes cover a question,
  tuned on 37 questions from two documents. Very short queries embed poorly
  ("ACID" alone scores just over the cutoff), so a keyword rescue accepts a
  short query when every content word literally appears together in a
  chunk and the words are distinctive. Re-check both on different notes.
- **Quiz questions come from a small local model.** Output is validated
  (4 distinct options, one correct answer, a real source chunk) and every
  question names the slide it came from, but a question can still be
  poorly worded, have a weak distractor, or occasionally have two defensible
  correct answers (seen in testing with real Llama 3). Professors can review
  each question on the Class Quizzes tab. Generation takes about 2 to 3
  minutes per quiz on CPU; `publish_quizzes.py` does it ahead of time.
- **Topics are slide titles.** A "topic" is the title of the slide a
  question or quiz maps to, taken from the page's first line. When a slide's
  body starts with an acronym, it can be glued onto the title.
- **The router learns from small, self-written data.** About 220 training
  messages written by the authors, plus any student corrections. It can
  still misroute unusual wording (the ambiguous line between "explain X" as a
  plain question and "explain X simply" is a convention we chose). Wrongly
  routed messages can be corrected in the app, and the agents themselves do
  not plan or negotiate: it is an orchestrated pipeline, not autonomous
  agents.
- **No authentication.** The Student ID is self-declared, so anyone can type
  another student's ID.

## Running the tests

```bash
python -m unittest discover -s tests -v
```

The suite uses a small synthetic index and a fake chat model, so it runs in
seconds without Ollama or any PDFs.

## Using `rag_core` directly

All retrieval logic lives in `rag_core/` behind a small function API:

```python
from rag_core import load_vectorstore, answer_question

vectorstore = load_vectorstore()
result = answer_question("Explain gradient descent", vectorstore)

result.grounded            # False if the relevance gate short-circuited the LLM call
result.answer               # answer string (or the fixed "not covered" message)
result.source_documents     # the chunks used to produce it (empty if not grounded)
```

New agents should import from `rag_core` rather than re-implementing
loading, chunking, embedding, or FAISS access, and subclass `agents.base.Agent`
so the Orchestrator can run them.
