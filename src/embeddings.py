from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import EMBEDDING_MODEL, EMBEDDING_DIM, VECTORSTORE_DIR
from src.ingestion import Chunk

_model: SentenceTransformer | None = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL} ...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        print("Embedding model loaded.")
    return _model

def embed_texts(
    texts: List[str],
    batch_size: int = 64,
    **kwargs,
) -> np.ndarray:
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)

def embed_query(query: str, **kwargs) -> np.ndarray:
    return embed_texts([query])

class VectorStore:
    INDEX_FILE  = "faiss.index"
    CHUNKS_FILE = "chunks.pkl"
    META_FILE   = "meta.json"

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim    = dim
        self.index  = faiss.IndexFlatIP(dim)
        self.chunks: List[Chunk] = []

    def add(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        assert len(chunks) == len(embeddings)
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def build_from_corpus(self, chunks: List[Chunk], batch_size: int = 64) -> None:
        print(f"Building FAISS index for {len(chunks)} chunks ...")
        texts = [c.text for c in chunks]
        embeddings = embed_texts(texts, batch_size=batch_size)
        self.add(chunks, embeddings)
        print(f"FAISS index built: {self.index.ntotal} vectors, dim={self.dim}")

    def search(self, query: str, top_k: int = 10) -> List[tuple[Chunk, float]]:
        q_emb = embed_query(query)
        scores, indices = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, directory: Path = VECTORSTORE_DIR) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / self.INDEX_FILE))
        with open(directory / self.CHUNKS_FILE, "wb") as f:
            pickle.dump(self.chunks, f)
        meta = {
            "n_chunks": len(self.chunks),
            "dim": self.dim,
            "embedding_model": EMBEDDING_MODEL,
        }
        with open(directory / self.META_FILE, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Vector store saved: {len(self.chunks)} chunks")

    @classmethod
    def load(cls, directory: Path = VECTORSTORE_DIR) -> "VectorStore":
        index_path  = directory / cls.INDEX_FILE
        chunks_path = directory / cls.CHUNKS_FILE
        if not index_path.exists():
            raise FileNotFoundError(f"No vector store at {directory}. Run ingestion first.")
        store = cls()
        store.index  = faiss.read_index(str(index_path))
        store.dim    = store.index.d
        with open(chunks_path, "rb") as f:
            store.chunks = pickle.load(f)
        print(f"Vector store loaded: {store.index.ntotal} vectors")
        return store

    def __len__(self) -> int:
        return self.index.ntotal