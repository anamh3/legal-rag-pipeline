# ════════════════════════════════════════════════════════════
# Legal RAG Pipeline — Dockerfile
# Multi-stage build: dependencies → runtime
# ════════════════════════════════════════════════════════════

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# ── Stage 1: Install dependencies ─────────────────────────
FROM base AS dependencies

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install fastapi uvicorn[standard] python-multipart

# Pre-download models so they're baked into the image
# (avoids downloading on every container start)
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2'); \
print('Embedding + Re-ranker models cached.')"

RUN python -c "\
from sentence_transformers import CrossEncoder; \
CrossEncoder('cross-encoder/nli-deberta-v3-large'); \
print('NLI model cached.')"

# ── Stage 2: Runtime ──────────────────────────────────────
FROM base AS runtime

WORKDIR /app

# Copy installed packages from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy cached HuggingFace models
COPY --from=dependencies /root/.cache/huggingface /root/.cache/huggingface

# Copy application code
COPY src/ ./src/
COPY server.py .
COPY ingest.py .
COPY evaluate.py .
COPY frontend/ ./frontend/

# Copy sample data (if present)
COPY data/ ./data/

# Create directories
RUN mkdir -p data/raw data/processed data/vectorstore

# Expose ports: 8000 = API, 3000 = frontend
EXPOSE 8000 3000

# Health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Start both API server and frontend file server
CMD ["sh", "-c", "\
    python -m http.server 3000 --directory frontend &\
    uvicorn server:app --host 0.0.0.0 --port 8000\
"]