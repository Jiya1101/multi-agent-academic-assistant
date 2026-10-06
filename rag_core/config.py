"""
Central configuration for the RAG module.

Keeping every path/model-name/hyperparameter in one file means the future
agents (Quiz Generator, Concept Explainer, etc.) can all import the same
settings instead of hard-coding their own copies.
"""

from pathlib import Path

# --- Paths -----------------------------------------------------------------
# Project root = the folder that contains this "rag_core" package.
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"   # raw uploaded PDFs live here
DB_DIR = BASE_DIR / "db"       # persisted FAISS index lives here

DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)

# --- Chunking ----------------------------------------------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# --- Embeddings --------------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Retrieval -----------------------------------------------------------------
TOP_K = 4  # number of chunks retrieved per query

# Max FAISS L2 distance (lower = more similar) for a retrieved chunk to be
# considered "relevant" to the query. Chunks are normalized embeddings, so
# distance ranges roughly 0 (identical) to 2 (unrelated). Measured on 37
# realistic questions (see seed_demo.py) against the AWS-databases + Spark
# notes: questions the notes cover score 0.00-1.10 (most under 0.95, with
# terse or broadly-worded ones like "Explain the ACID properties" reaching
# ~1.10), while questions they do not cover score 1.25-1.66 (topic-adjacent
# ones like exam scope ~1.25, unrelated ones ~1.5). 1.15 sits in that gap.
# When the best match is still above this, the question is answered "not
# covered by the notes" without calling the LLM at all, since small local
# models don't reliably self-enforce that constraint via prompt instructions
# alone. It is a heuristic: re-check it after indexing very different notes.
RELEVANCE_SCORE_THRESHOLD = 1.15

# --- LLM (Ollama) --------------------------------------------------------------
OLLAMA_MODEL = "llama3"
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_TEMPERATURE = 0.2

# --- FAISS index name -----------------------------------------------------------
FAISS_INDEX_NAME = "course_notes_index"

# --- Speech to text ------------------------------------------------------------
# Whisper, run locally. English-only "small" model: the most accurate of the
# small variants on technical words, at the cost of a few seconds per question on CPU.
SPEECH_MODEL_NAME = "openai/whisper-small.en"

# --- Faculty insight -----------------------------------------------------------
# SQLite file (inside db/) that records every student question and its outcome.
QUERY_LOG_FILENAME = "query_log.sqlite"

# Two questions land in the same cluster when the average cosine distance
# between their embeddings is below this. Lower = stricter (only near-
# paraphrases group together); higher = broader topic groups. Checked on the
# 37 questions in seed_demo.py: 0.45 fragments one concept (ACID/BASE) into
# three clusters; 0.55 groups each concept once; from 0.65 a logistics
# question ("is Spark in the exam?") starts merging into a concept cluster.
CLUSTER_DISTANCE_THRESHOLD = 0.55

# Source name given to professor-written answers that are added to the index.
FACULTY_SOURCE_NAME = "Faculty Answers"
