"""
retrieval.py — Hybrid retrieval combining dense (FAISS) + sparse (BM25) via RRF.

Reciprocal Rank Fusion consistently outperforms either approach alone on legal text.
The RRF formula: score(d) = sum(1 / (k + rank_i(d))) across retrieval methods.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import (
    TOP_K_DENSE, TOP_K_SPARSE, TOP_K_FINAL, VECTORSTORE_DIR
)
from src.embeddings import VectorStore
from src.ingestion import Chunk


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    chunk:        Chunk
    dense_score:  Optional[float]  # FAISS cosine score (None if not in dense top-k)
    sparse_score: Optional[float]  # BM25 score (None if not in sparse top-k)
    rrf_score:    float            # Final fused score
    dense_rank:   Optional[int]
    sparse_rank:  Optional[int]


# ── BM25 index ────────────────────────────────────────────────────────────────

class BM25Index:
    """Thin wrapper around rank_bm25.BM25Okapi with persist support."""

    BM25_FILE = "bm25.pkl"

    def __init__(self, chunks: Optional[List[Chunk]] = None):
        self.chunks: List[Chunk] = []
        self._bm25: Optional[BM25Okapi] = None

        if chunks:
            self.build(chunks)

    def build(self, chunks: List[Chunk]) -> None:
        """Tokenize and build BM25 index."""
        self.chunks = chunks
        tokenized  = [self._tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + lowercase tokenizer."""
        import re
        text = text.lower()
        tokens = re.findall(r"\b[a-z][a-z0-9]{1,}\b", text)
        return tokens or [""]

    def search(self, query: str, top_k: int = TOP_K_SPARSE) -> List[tuple[Chunk, float]]:
        """Return top-k (chunk, score) pairs."""
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built. Call build() first.")

        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:
                results.append((self.chunks[idx], score))
        return results

    def save(self, directory: Path = VECTORSTORE_DIR) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / self.BM25_FILE, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, directory: Path = VECTORSTORE_DIR) -> "BM25Index":
        path = directory / cls.BM25_FILE
        if not path.exists():
            raise FileNotFoundError(f"BM25 index not found at {path}")
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: List[List[tuple[Chunk, float]]],
    k: int = 60,
) -> List[tuple[Chunk, float]]:
    """
    Merge multiple ranked lists using RRF.
    k=60 is the standard default from the original paper (Cormack et al., 2009).

    score(doc) = sum over lists of: 1 / (k + rank_in_list)
    """
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, Chunk] = {}

    for ranked_list in ranked_lists:
        for rank, (chunk, _raw_score) in enumerate(ranked_list, start=1):
            cid = chunk.chunk_id
            scores[cid]    = scores.get(cid, 0.0) + 1.0 / (k + rank)
            chunk_map[cid] = chunk

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [(chunk_map[cid], scores[cid]) for cid in sorted_ids]


# ── Hybrid Retriever ──────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Combines dense FAISS retrieval + sparse BM25 retrieval via RRF.

    Usage:
        retriever = HybridRetriever.load()
        results = retriever.retrieve("What are the indemnification obligations?")
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
    ):
        self.vector_store = vector_store
        self.bm25_index   = bm25_index

    # ── Build ─────────────────────────────────────────────────────────────────

    @classmethod
    def build(cls, chunks: List[Chunk]) -> "HybridRetriever":
        """Build both indices from a list of chunks."""
        print("Building BM25 index ...")
        bm25 = BM25Index(chunks)
        print(f"BM25 index built: {len(chunks)} documents")
        return cls(VectorStore(), bm25)

    # ── Retrieve ──────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k_dense: int  = TOP_K_DENSE,
        top_k_sparse: int = TOP_K_SPARSE,
        top_k_final: int  = TOP_K_FINAL,
        doc_type_filter: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Full hybrid retrieval pipeline.

        Args:
            query:           User question
            top_k_dense:     Candidates from FAISS
            top_k_sparse:    Candidates from BM25
            top_k_final:     Final chunks after RRF fusion
            doc_type_filter: Optionally restrict to "contract" | "case_law" | "statute"

        Returns:
            List of RetrievalResult sorted by descending RRF score
        """
        # Step 1: Dense retrieval
        dense_results  = self.vector_store.search(query, top_k=top_k_dense)

        # Step 2: Sparse retrieval
        sparse_results = self.bm25_index.search(query, top_k=top_k_sparse)

        # Step 3: Optional doc type filter
        if doc_type_filter:
            dense_results  = [(c, s) for c, s in dense_results  if c.doc_type == doc_type_filter]
            sparse_results = [(c, s) for c, s in sparse_results if c.doc_type == doc_type_filter]

        # Step 4: RRF fusion
        fused = reciprocal_rank_fusion([dense_results, sparse_results])
        fused = fused[:top_k_final]

        # Step 5: Build result objects with rank info
        dense_rank_map  = {c.chunk_id: (r+1, s) for r, (c, s) in enumerate(dense_results)}
        sparse_rank_map = {c.chunk_id: (r+1, s) for r, (c, s) in enumerate(sparse_results)}

        results = []
        for chunk, rrf_score in fused:
            cid = chunk.chunk_id
            d_rank, d_score = dense_rank_map.get(cid,  (None, None))
            s_rank, s_score = sparse_rank_map.get(cid, (None, None))
            results.append(RetrievalResult(
                chunk        = chunk,
                dense_score  = d_score,
                sparse_score = s_score,
                rrf_score    = rrf_score,
                dense_rank   = d_rank,
                sparse_rank  = s_rank,
            ))
        return results

    def retrieve_for_ablation(
        self,
        query: str,
        top_k: int = TOP_K_FINAL,
    ) -> Dict[str, List[Chunk]]:
        """
        Return results from each retrieval strategy separately.
        Used for ablation experiments.
        """
        dense_results  = self.vector_store.search(query, top_k=top_k)
        sparse_results = self.bm25_index.search(query, top_k=top_k)
        fused          = reciprocal_rank_fusion([dense_results, sparse_results])[:top_k]

        return {
            "dense":  [c for c, _ in dense_results[:top_k]],
            "sparse": [c for c, _ in sparse_results[:top_k]],
            "hybrid": [c for c, _ in fused],
        }

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self, directory: Path = VECTORSTORE_DIR) -> None:
        self.vector_store.save(directory)
        self.bm25_index.save(directory)

    @classmethod
    def load(cls, directory: Path = VECTORSTORE_DIR) -> "HybridRetriever":
        vector_store = VectorStore.load(directory)
        bm25_index   = BM25Index.load(directory)
        return cls(vector_store, bm25_index)
