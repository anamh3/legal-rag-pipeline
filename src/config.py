"""
config.py — Central configuration for the Legal RAG pipeline.

All settings loaded from .env with sensible defaults.
Add RERANKER_MODEL and TOP_K_RERANK for the enhanced retrieval pipeline.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.parent
RAW_DATA_DIR    = BASE_DIR / os.getenv("RAW_DATA_DIR",    "data/raw")
PROCESSED_DIR   = BASE_DIR / os.getenv("PROCESSED_DIR",   "data/processed")
VECTORSTORE_DIR = BASE_DIR / os.getenv("VECTORSTORE_DIR", "data/vectorstore")

for _p in (RAW_DATA_DIR, PROCESSED_DIR, VECTORSTORE_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_PROVIDER       = os.getenv("LLM_PROVIDER", "groq")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
OPENAI_LLM_MODEL   = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
GROQ_LLM_MODEL     = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")

def get_llm_model() -> str:
    return OPENAI_LLM_MODEL if LLM_PROVIDER == "openai" else GROQ_LLM_MODEL

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM   = int(os.getenv("EMBEDDING_DIM", "384"))

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_DENSE  = int(os.getenv("TOP_K_DENSE",  "10"))
TOP_K_SPARSE = int(os.getenv("TOP_K_SPARSE", "10"))
TOP_K_FINAL  = int(os.getenv("TOP_K_FINAL",  "5"))

# ── Re-ranker (NEW) ──────────────────────────────────────────────────────────
RERANKER_MODEL  = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2")
TOP_K_RERANK    = int(os.getenv("TOP_K_RERANK", "20"))
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# ── Hallucination Detection ───────────────────────────────────────────────────
NLI_MODEL                  = os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-large")
NLI_BATCH_SIZE             = int(os.getenv("NLI_BATCH_SIZE", "16"))
NLI_THRESHOLD              = float(os.getenv("NLI_THRESHOLD", "0.5"))
FAITHFULNESS_WARN_THRESHOLD = float(os.getenv("FAITHFULNESS_WARN_THRESHOLD", "0.7"))
RERANKER_MODEL  = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2")
TOP_K_RERANK    = int(os.getenv("TOP_K_RERANK", "20"))
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"