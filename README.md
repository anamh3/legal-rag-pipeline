# Legal RAG Pipeline with Hallucination Detection

A RAG (Retrieval-Augmented Generation) system built for legal documents. Upload contracts, NDAs, or employment agreements, ask questions, and get answers that are checked for hallucinations using an NLI model.

The main idea: instead of blindly trusting LLM output, every claim in the answer is verified against the retrieved source text. If the model makes something up, the system catches it.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![React](https://img.shields.io/badge/React-18-61dafb)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)
![Tests](https://img.shields.io/badge/Tests-37%20passed-brightgreen)

---

## What it does

1. **Upload** legal PDFs/TXT files through the web UI
2. **Ask** questions like "What is the non-compete period?" or "Who owns the IP?"
3. **Get answers** with chunk citations ([Chunk 1], [Chunk 2])
4. **See a faithfulness score** — what percentage of claims are actually supported by the source documents
5. **Review individual claims** — each one is labeled as grounded, ungrounded, or contradicted

## How retrieval works

Documents go through a two-stage retrieval pipeline:

```
Query
 │
 ├── Dense retrieval (FAISS + BGE embeddings)
 ├── Sparse retrieval (BM25 keyword matching)
 │
 ├── Reciprocal Rank Fusion (merges both result sets)
 │
 └── Cross-encoder re-ranking (scores each query-chunk pair)
      └── Top 5 chunks → sent to LLM
```

The hybrid approach (dense + sparse + RRF) consistently outperforms either method alone. The cross-encoder re-ranker adds another layer by actually reading the query and chunk together, which catches things embedding similarity misses.

## How hallucination detection works

After the LLM generates an answer:

1. The answer is split into atomic claims (individual factual statements)
2. Each claim is scored against the retrieved chunks using a DeBERTa NLI model
3. Claims are classified as **entailed** (grounded), **neutral** (ungrounded), or **contradicted**
4. A faithfulness score = grounded claims / total claims

This isn't perfect — the NLI model can be overly strict with paraphrased language. In testing, straightforward factual questions (like "What are the license restrictions?") score 100%, while broader questions sometimes score lower even when the answer is correct. That's a known limitation.


```

## Tech stack

- **LLM**: Llama 3.3 70B via Groq (free tier)
- **Embeddings**: BAAI/bge-small-en-v1.5 (local, no API needed)
- **Vector store**: FAISS (IndexFlatIP with L2 normalization)
- **Sparse retrieval**: BM25 via rank_bm25
- **Re-ranker**: cross-encoder/ms-marco-MiniLM-L-12-v2
- **Hallucination detection**: cross-encoder/nli-deberta-v3-large
- **Backend**: FastAPI with SSE streaming
- **Frontend**: React 18 (single HTML file, Babel in-browser)
- **Containerization**: Docker

## Project structure

```
legal_rag/
├── server.py              # FastAPI backend (API + streaming + upload)
├── app.py                 # Gradio app (alternative UI)
├── ingest.py              # CLI document ingestion
├── evaluate.py            # CLI evaluation runner
├── Dockerfile
├── docker-compose.yml
│
├── frontend/
│   └── index.html         # React frontend (single file)
│
├── src/
│   ├── config.py           # All settings from .env
│   ├── ingestion.py        # PDF/HTML/TXT parsing, clause-level chunking
│   ├── embeddings.py       # BGE embeddings + FAISS index
│   ├── retrieval.py        # Hybrid retrieval (dense + BM25 + RRF)
│   ├── reranker.py         # Cross-encoder re-ranking + query expansion
│   ├── generation.py       # LLM prompting with citation format
│   ├── hallucination.py    # NLI-based claim verification
│   ├── pipeline.py         # Orchestrates retrieval → generation → verification
│   └── evaluation.py       # Ablation study + RAGAS + hallucination analysis
│
├── tests/
│   └── test_legal_rag.py   # 37 tests (no API calls needed)
│
└── data/
    ├── raw/                # Upload documents here
    ├── processed/          # Chunked documents (auto-generated)
    └── vectorstore/        # FAISS + BM25 indexes (auto-generated)
```

## Setup

### Local (without Docker)

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/legal-rag-pipeline.git
cd legal-rag-pipeline

# Virtual environment
python -m venv venv
venv\Scripts\Activate    # Windows
# source venv/bin/activate  # Mac/Linux

# Install
pip install -r requirements.txt
pip install fastapi uvicorn

# Set API key
copy .env.example .env
# Edit .env → add your GROQ_API_KEY (free at https://console.groq.com)

# Start API server
python server.py

# In another terminal, serve frontend
cd frontend
python -m http.server 3000

# Open http://localhost:3000
```

### Docker

```bash
docker compose up --build
# Open http://localhost:3000
```

### Run tests

```bash
pytest tests/test_legal_rag.py -v
```

## Sample results

Tested with a software license agreement, mutual NDA, and employment agreement:

| Query | Faithfulness | Claims |
|-------|-------------|--------|
| What are the license restrictions? | 100% | 3/3 grounded |
| What severance benefits is the employee entitled to? | 100% | 2/2 grounded |
| Can you explain the license agreement briefly? | 83% | 5/6 grounded |
| What is the non-compete period? | 50% | 1/2 grounded |
| How long do confidentiality obligations survive? | 25% | 1/4 grounded |

The lower scores on broader questions are usually the NLI model being strict about paraphrasing rather than the answer being wrong. This is documented as a known limitation.

## Evaluation

The project includes a 4-strategy retrieval ablation framework:

```bash
python evaluate.py --generate-testset --no-llm --ablation --k 5
```

This compares dense-only, sparse-only, hybrid (RRF), and hybrid+re-ranker across MRR@k, Recall@k, NDCG@k, and Precision@k. The evaluation also breaks down hallucination rates by document type and legal category.

## Known limitations

- **NLI strictness**: The DeBERTa model sometimes marks paraphrased but correct claims as "ungrounded." This inflates hallucination rates on broader questions.
- **Chunk boundaries**: Legal clauses that span multiple chunks can lose context. The clause-level splitter helps but isn't perfect.
- **Single-user**: No authentication. The query history and document store are shared.
- **Frontend**: Built as a single HTML file with Babel for simplicity. A proper React build would be better for production.
- **Model size**: First startup downloads ~1.5GB of models (BGE + re-ranker + DeBERTa). Subsequent starts use the cache.

## What I'd improve next

- Fine-tune the NLI model on legal text to reduce false positives
- Add chunk overlap visualization to show which parts of the source support each claim
- Implement proper user sessions and authentication
- Build a proper React app with a build step instead of in-browser Babel
- Add PDF rendering in the sources panel so users can see the original document

## 
