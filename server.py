"""
server.py — FastAPI backend for the Legal RAG pipeline.

Features:
  - SSE streaming for real-time LLM answer delivery
  - Re-ranker integration with toggle
  - Query history with persistence
  - Structured JSON responses for sources and claims
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import shutil

from fastapi import FastAPI, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="Legal RAG API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load pipeline + reranker ──────────────────────────────────────────────────

pipeline = None
reranker = None

@app.on_event("startup")
def startup():
    global pipeline, reranker

    from src.pipeline import RAGPipeline
    print("Loading RAG pipeline ...")
    pipeline = RAGPipeline.load()
    print("Pipeline ready.")

    try:
        from src.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()
        print("Re-ranker available.")
    except Exception as e:
        print(f"Re-ranker not available: {e}")
        reranker = None


# ── Query history ─────────────────────────────────────────────────────────────

HISTORY_FILE = Path("data/query_history.json")
query_history: List[dict] = []

def _load_history():
    global query_history
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE) as f:
                query_history = json.load(f)
        except Exception:
            query_history = []

def _save_history():
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(query_history[-100:], f, indent=2)  # keep last 100

_load_history()


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    doc_type_filter: str = "All"
    top_k: int = 5
    use_reranker: bool = True
    temperature: float = 0.0


class ChunkResponse(BaseModel):
    chunk_id: str
    text: str
    source_file: str
    section: str
    doc_type: str
    page_num: Optional[int]
    token_count: int
    rrf_score: float
    reranker_score: Optional[float] = None


class ClaimResponse(BaseModel):
    claim: str
    is_grounded: bool
    label: str
    entailment_prob: float


class QueryResponse(BaseModel):
    query_id: str
    question: str
    answer: str
    model: str
    retrieval_method: str
    faithfulness_score: float
    is_flagged: bool
    n_grounded: int
    n_ungrounded: int
    n_contradicted: int
    n_claims: int
    chunks: List[ChunkResponse]
    claims: List[ClaimResponse]
    latency_ms: float
    timestamp: str


# ── Streaming endpoint ────────────────────────────────────────────────────────

@app.post("/api/query/stream")
async def query_stream(req: QueryRequest):
    """
    SSE streaming endpoint.

    Sends events in this order:
      1. {"type": "status", "message": "Retrieving..."}
      2. {"type": "chunks", "data": [...]}
      3. {"type": "answer_start"}
      4. {"type": "answer_token", "token": "..."}  (multiple)
      5. {"type": "claims", "data": [...]}
      6. {"type": "faithfulness", "data": {...}}
      7. {"type": "done", "data": {...}}
    """
    def event_stream():
        start_time = time.time()
        query_id = str(uuid.uuid4())[:8]

        filter_val = (
            None if req.doc_type_filter == "All"
            else req.doc_type_filter.lower().replace(" ", "_")
        )

        # ── Step 1: Retrieval ─────────────────────────────────────────────
        yield _sse({"type": "status", "message": "Retrieving relevant chunks..."})

        try:
            if req.use_reranker and reranker is not None:
                retrieval_results = pipeline.retriever.retrieve(
                    req.question, top_k_final=20, doc_type_filter=filter_val,
                )
                reranked = reranker.rerank(req.question, retrieval_results, top_k=req.top_k)
                chunks = [r.chunk for r in reranked]
                retrieval_method = "Hybrid + Re-ranker"

                chunk_responses = [
                    {
                        "chunk_id": r.chunk.chunk_id,
                        "text": r.chunk.text,
                        "source_file": r.chunk.source_file,
                        "section": r.chunk.section or "",
                        "doc_type": r.chunk.doc_type,
                        "page_num": r.chunk.page_num,
                        "token_count": r.chunk.token_count,
                        "rrf_score": round(r.rrf_score, 4),
                        "reranker_score": round(r.reranker_score, 3),
                    }
                    for r in reranked
                ]
            else:
                result = pipeline.retriever.retrieve(
                    req.question, top_k_final=req.top_k, doc_type_filter=filter_val,
                )
                chunks = [r.chunk for r in result]
                retrieval_method = "Hybrid (RRF)"

                chunk_responses = [
                    {
                        "chunk_id": r.chunk.chunk_id,
                        "text": r.chunk.text,
                        "source_file": r.chunk.source_file,
                        "section": r.chunk.section or "",
                        "doc_type": r.chunk.doc_type,
                        "page_num": r.chunk.page_num,
                        "token_count": r.chunk.token_count,
                        "rrf_score": round(r.rrf_score, 4),
                        "reranker_score": None,
                    }
                    for r in result
                ]
        except Exception as e:
            yield _sse({"type": "error", "message": str(e)})
            return

        yield _sse({"type": "chunks", "data": chunk_responses})

        # ── Step 2: Generation (streaming) ────────────────────────────────
        yield _sse({"type": "status", "message": "Generating answer..."})
        yield _sse({"type": "answer_start"})

        try:
            from src.generation import generate_answer
            gen_result = generate_answer(
                req.question, chunks, temperature=req.temperature,
            )

            # Simulate token streaming (Groq/OpenAI returns full response)
            answer = gen_result.answer
            words = answer.split(" ")
            buffer = ""
            for i, word in enumerate(words):
                buffer += word + " "
                # Send in small batches for smooth streaming
                if i % 3 == 2 or i == len(words) - 1:
                    yield _sse({"type": "answer_token", "token": buffer})
                    buffer = ""
                    time.sleep(0.02)  # Small delay for streaming effect

        except Exception as e:
            yield _sse({"type": "error", "message": f"Generation failed: {e}"})
            return

        # ── Step 3: Faithfulness check ────────────────────────────────────
        yield _sse({"type": "status", "message": "Verifying claims..."})

        try:
            from src.hallucination import compute_faithfulness
            faith = compute_faithfulness(req.question, answer, chunks)

            claim_responses = [
                {
                    "claim": c.claim,
                    "is_grounded": c.is_grounded,
                    "label": (
                        c.best_label.value
                        if hasattr(c.best_label, "value")
                        else str(c.best_label)
                    ),
                    "entailment_prob": round(c.best_entailment_prob, 3),
                }
                for c in faith.claims
            ]

            yield _sse({"type": "claims", "data": claim_responses})
            yield _sse({
                "type": "faithfulness",
                "data": {
                    "score": round(faith.faithfulness_score, 3),
                    "is_flagged": faith.is_flagged,
                    "n_grounded": faith.n_grounded,
                    "n_ungrounded": faith.n_ungrounded,
                    "n_contradicted": faith.n_contradicted,
                    "n_claims": len(faith.claims),
                },
            })
        except Exception as e:
            yield _sse({"type": "error", "message": f"Verification failed: {e}"})
            return

        # ── Step 4: Done ──────────────────────────────────────────────────
        latency = (time.time() - start_time) * 1000
        timestamp = datetime.now().isoformat()

        done_data = {
            "query_id": query_id,
            "model": gen_result.model,
            "retrieval_method": retrieval_method,
            "latency_ms": round(latency, 0),
            "timestamp": timestamp,
        }
        yield _sse({"type": "done", "data": done_data})

        # Save to history
        history_entry = {
            "query_id": query_id,
            "question": req.question,
            "answer_preview": answer[:150],
            "faithfulness_score": round(faith.faithfulness_score, 3),
            "is_flagged": faith.is_flagged,
            "model": gen_result.model,
            "retrieval_method": retrieval_method,
            "latency_ms": round(latency, 0),
            "timestamp": timestamp,
        }
        query_history.append(history_entry)
        _save_history()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Non-streaming endpoint (fallback) ────────────────────────────────────────

@app.post("/api/query", response_model=QueryResponse)
def query_sync(req: QueryRequest):
    """Synchronous query endpoint (non-streaming fallback)."""
    start_time = time.time()
    query_id = str(uuid.uuid4())[:8]

    filter_val = (
        None if req.doc_type_filter == "All"
        else req.doc_type_filter.lower().replace(" ", "_")
    )

    if req.use_reranker and reranker is not None:
        retrieval_results = pipeline.retriever.retrieve(
            req.question, top_k_final=20, doc_type_filter=filter_val,
        )
        reranked = reranker.rerank(req.question, retrieval_results, top_k=req.top_k)
        chunks_obj = [r.chunk for r in reranked]
        retrieval_method = "Hybrid + Re-ranker"

        chunk_responses = [
            ChunkResponse(
                chunk_id=r.chunk.chunk_id, text=r.chunk.text,
                source_file=r.chunk.source_file, section=r.chunk.section or "",
                doc_type=r.chunk.doc_type, page_num=r.chunk.page_num,
                token_count=r.chunk.token_count, rrf_score=round(r.rrf_score, 4),
                reranker_score=round(r.reranker_score, 3),
            ) for r in reranked
        ]
    else:
        result = pipeline.retriever.retrieve(
            req.question, top_k_final=req.top_k, doc_type_filter=filter_val,
        )
        chunks_obj = [r.chunk for r in result]
        retrieval_method = "Hybrid (RRF)"

        chunk_responses = [
            ChunkResponse(
                chunk_id=r.chunk.chunk_id, text=r.chunk.text,
                source_file=r.chunk.source_file, section=r.chunk.section or "",
                doc_type=r.chunk.doc_type, page_num=r.chunk.page_num,
                token_count=r.chunk.token_count, rrf_score=round(r.rrf_score, 4),
            ) for r in result
        ]

    from src.generation import generate_answer
    gen_result = generate_answer(req.question, chunks_obj, temperature=req.temperature)

    from src.hallucination import compute_faithfulness
    faith = compute_faithfulness(req.question, gen_result.answer, chunks_obj)

    latency = (time.time() - start_time) * 1000

    claim_responses = [
        ClaimResponse(
            claim=c.claim, is_grounded=c.is_grounded,
            label=c.best_label.value if hasattr(c.best_label, "value") else str(c.best_label),
            entailment_prob=round(c.best_entailment_prob, 3),
        ) for c in faith.claims
    ]

    return QueryResponse(
        query_id=query_id, question=req.question, answer=gen_result.answer,
        model=gen_result.model, retrieval_method=retrieval_method,
        faithfulness_score=round(faith.faithfulness_score, 3),
        is_flagged=faith.is_flagged, n_grounded=faith.n_grounded,
        n_ungrounded=faith.n_ungrounded, n_contradicted=faith.n_contradicted,
        n_claims=len(faith.claims), chunks=chunk_responses,
        claims=claim_responses, latency_ms=round(latency, 0),
        timestamp=datetime.now().isoformat(),
    )


# ── History endpoint ──────────────────────────────────────────────────────────

@app.get("/api/history")
def get_history(limit: int = Query(default=20, le=100)):
    """Return recent query history."""
    return {"history": list(reversed(query_history[-limit:]))}


@app.delete("/api/history")
def clear_history():
    global query_history
    query_history = []
    _save_history()
    return {"status": "cleared"}


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    from src.config import RAW_DATA_DIR
    raw_files = list(RAW_DATA_DIR.glob("*.*")) if RAW_DATA_DIR.exists() else []
    return {
        "status": "ok",
        "pipeline_loaded": pipeline is not None,
        "reranker_available": reranker is not None,
        "history_count": len(query_history),
        "documents_count": len(raw_files),
    }


# ── Document upload & management ─────────────────────────────────────────────

ALLOWED_EXTENSIONS = {".pdf", ".html", ".htm", ".txt"}

@app.get("/api/documents")
def list_documents():
    """List all uploaded documents with metadata."""
    from src.config import RAW_DATA_DIR, VECTORSTORE_DIR

    docs = []
    if RAW_DATA_DIR.exists():
        for f in sorted(RAW_DATA_DIR.iterdir()):
            if f.suffix.lower() in ALLOWED_EXTENSIONS and f.is_file():
                docs.append({
                    "filename": f.name,
                    "size_kb": round(f.stat().st_size / 1024, 1),
                    "type": f.suffix.lower().replace(".", ""),
                    "uploaded_at": datetime.fromtimestamp(
                        f.stat().st_mtime
                    ).isoformat(),
                })

    # Check if vector store exists
    vs_exists = (VECTORSTORE_DIR / "index.faiss").exists() or \
                (VECTORSTORE_DIR / "faiss.index").exists()

    return {
        "documents": docs,
        "total": len(docs),
        "index_built": vs_exists,
    }


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF, HTML, TXT) to data/raw/.
    Does NOT auto-ingest — call /api/documents/ingest after uploading.
    """
    from src.config import RAW_DATA_DIR

    # Validate extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return {
            "status": "error",
            "message": f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        }

    # Save file
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DATA_DIR / file.filename

    # Avoid overwriting — rename if exists
    if dest.exists():
        stem = dest.stem
        i = 1
        while dest.exists():
            dest = RAW_DATA_DIR / f"{stem}_{i}{ext}"
            i += 1

    with open(dest, "wb") as f:
        content = await file.read()
        f.write(content)

    return {
        "status": "ok",
        "filename": dest.name,
        "size_kb": round(len(content) / 1024, 1),
        "message": f"Uploaded {dest.name}. Run 'Ingest' to add it to the search index.",
    }


@app.post("/api/documents/upload-multiple")
async def upload_multiple(files: List[UploadFile] = File(...)):
    """Upload multiple documents at once."""
    results = []
    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            results.append({"filename": file.filename, "status": "error", "message": f"Unsupported: {ext}"})
            continue

        from src.config import RAW_DATA_DIR
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        dest = RAW_DATA_DIR / file.filename

        if dest.exists():
            stem = dest.stem
            i = 1
            while dest.exists():
                dest = RAW_DATA_DIR / f"{stem}_{i}{ext}"
                i += 1

        with open(dest, "wb") as f:
            content = await file.read()
            f.write(content)

        results.append({
            "filename": dest.name,
            "status": "ok",
            "size_kb": round(len(content) / 1024, 1),
        })

    return {"results": results, "uploaded": sum(1 for r in results if r["status"] == "ok")}


@app.post("/api/documents/ingest")
def ingest_documents(reset: bool = False):
    """
    Run the ingestion pipeline on all documents in data/raw/.
    Rebuilds the FAISS + BM25 indexes.

    After this completes, the pipeline is reloaded automatically.
    """
    global pipeline

    from src.config import RAW_DATA_DIR, PROCESSED_DIR, VECTORSTORE_DIR
    from src.ingestion import ingest_directory

    if not RAW_DATA_DIR.exists() or not list(RAW_DATA_DIR.glob("*.*")):
        return {"status": "error", "message": "No documents found in data/raw/"}

    try:
        # Clear old data if reset
        if reset:
            if PROCESSED_DIR.exists():
                shutil.rmtree(PROCESSED_DIR)
            if VECTORSTORE_DIR.exists():
                shutil.rmtree(VECTORSTORE_DIR)
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

        # Step 1: Parse and chunk documents
        chunks = ingest_directory(RAW_DATA_DIR, PROCESSED_DIR)

        # Step 2: Build embeddings + FAISS index
        from src.embeddings import VectorStore
        vector_store = VectorStore()
        vector_store.build_from_corpus(chunks)
        vector_store.save(VECTORSTORE_DIR)

        # Step 3: Build BM25 index
        from src.retrieval import BM25Index
        bm25 = BM25Index(chunks)
        bm25.save(VECTORSTORE_DIR)

        # Step 4: Reload pipeline
        from src.pipeline import RAGPipeline
        pipeline = RAGPipeline.load()

        # Step 5: Generate question suggestions from chunks
        suggestions = _generate_suggestions(chunks)
        _save_suggestions(suggestions)

        return {
            "status": "ok",
            "chunks": len(chunks),
            "suggestions": suggestions,
            "message": f"Ingested {len(chunks)} chunks. Pipeline reloaded.",
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.delete("/api/documents/{filename}")
def delete_document(filename: str):
    """Delete a document from data/raw/. Requires re-ingestion after."""
    from src.config import RAW_DATA_DIR

    filepath = RAW_DATA_DIR / filename
    if not filepath.exists():
        return {"status": "error", "message": f"File not found: {filename}"}

    filepath.unlink()
    return {
        "status": "ok",
        "message": f"Deleted {filename}. Run 'Ingest' to rebuild the index.",
    }


# ── Question suggestions ──────────────────────────────────────────────────────

SUGGESTIONS_FILE = Path("data/suggestions.json")

def _generate_suggestions(chunks: list, max_suggestions: int = 8) -> List[str]:
    """
    Generate question suggestions from ingested chunks.
    Uses section headers and key legal terms to create relevant questions.
    No LLM call needed — fast, deterministic, and free.
    """
    import re

    # Map of legal topics to question templates
    TOPIC_QUESTIONS = {
        "indemnif":        "What are the indemnification obligations?",
        "terminat":        "What are the termination conditions and notice requirements?",
        "confidential":    "What confidentiality obligations apply?",
        "non-compete":     "What is the non-compete period and scope?",
        "non-solicitat":   "What are the non-solicitation restrictions?",
        "governing law":   "What is the governing law and jurisdiction?",
        "jurisdiction":    "What is the governing law and jurisdiction?",
        "liabilit":        "What is the limitation of liability?",
        "warrant":         "What warranties are provided?",
        "intellectual property": "Who owns the intellectual property?",
        "payment":         "What are the payment terms and fees?",
        "license":         "What are the license grant and restrictions?",
        "severance":       "What severance benefits are provided?",
        "compensation":    "What is the compensation structure?",
        "force majeure":   "What are the force majeure provisions?",
        "arbitrat":        "How are disputes resolved?",
        "assignment":      "Can the agreement be assigned or transferred?",
        "confidential.*surviv": "How long do confidentiality obligations survive?",
        "damages":         "What damages or remedies are available for breach?",
    }

    found_questions = []
    found_topics = set()

    # Scan all chunks for matching topics
    for chunk in chunks:
        text_lower = chunk.text.lower()
        for topic_pattern, question in TOPIC_QUESTIONS.items():
            if topic_pattern not in found_topics and re.search(topic_pattern, text_lower):
                found_questions.append(question)
                found_topics.add(topic_pattern)

    # Deduplicate (some patterns map to the same question)
    seen = set()
    unique = []
    for q in found_questions:
        if q not in seen:
            unique.append(q)
            seen.add(q)

    # Add document-specific questions based on source files
    source_files = list({c.source_file for c in chunks})
    for src in source_files[:3]:
        name = src.replace("_", " ").replace(".txt", "").replace(".pdf", "").title()
        unique.append(f"Summarize the key terms of the {name}")

    return unique[:max_suggestions]


def _save_suggestions(suggestions: List[str]):
    SUGGESTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SUGGESTIONS_FILE, "w") as f:
        json.dump(suggestions, f, indent=2)


def _load_suggestions() -> List[str]:
    if SUGGESTIONS_FILE.exists():
        with open(SUGGESTIONS_FILE) as f:
            return json.load(f)
    return []


@app.get("/api/suggestions")
def get_suggestions():
    """Return auto-generated question suggestions for current documents."""
    suggestions = _load_suggestions()
    has_index = pipeline is not None
    return {
        "suggestions": suggestions,
        "has_index": has_index,
    }


# ── SSE helper ────────────────────────────────────────────────────────────────

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)