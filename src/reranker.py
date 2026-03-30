"""
reranker.py — Cross-encoder re-ranking for retrieval quality.

After hybrid retrieval (dense + BM25 + RRF), the re-ranker scores each
(query, chunk) pair with a cross-encoder model and re-sorts. This typically
adds +5-15% MRR over RRF alone on legal text.

Architecture:
    Query → Hybrid Retriever (top_k_rerank candidates)
          → Cross-Encoder re-scoring (each candidate individually)
          → Top-K final results

The cross-encoder sees both query and passage together, making it much
more accurate than bi-encoder similarity — but too slow for first-stage
retrieval, hence the two-stage pipeline.

Models tested:
    - cross-encoder/ms-marco-MiniLM-L-12-v2  (fast, good baseline)
    - BAAI/bge-reranker-v2-m3                 (multilingual, stronger)
    - cross-encoder/ms-marco-electra-base     (balanced speed/quality)
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

load_dotenv()

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2")
TOP_K_RERANK   = int(os.getenv("TOP_K_RERANK", "20"))   # candidates to re-rank
TOP_K_FINAL    = int(os.getenv("TOP_K_FINAL", "5"))      # final results after re-rank


@dataclass
class RerankedResult:
    """Single re-ranked result with scores from all retrieval stages."""
    chunk: object              # ingestion.Chunk
    reranker_score: float      # cross-encoder relevance score
    rrf_score: float           # original RRF score (pre-rerank)
    dense_rank: Optional[int]  # rank in dense-only retrieval
    sparse_rank: Optional[int] # rank in BM25-only retrieval
    final_rank: int            # rank after re-ranking


class CrossEncoderReranker:
    """
    Two-stage retrieval: Hybrid RRF → Cross-Encoder Re-ranking.

    Usage:
        reranker = CrossEncoderReranker()
        results  = reranker.rerank(query, hybrid_results, top_k=5)
    """

    def __init__(self, model_name: str = RERANKER_MODEL, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._load_time: Optional[float] = None

    @property
    def model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            print(f"Loading re-ranker model: {self.model_name} ...")
            start = time.time()
            self._model = CrossEncoder(
                self.model_name,
                max_length=512,
                device=self.device,
            )
            self._load_time = time.time() - start
            print(f"Re-ranker loaded in {self._load_time:.1f}s")
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list,  # List[RetrievalResult]
        top_k: int = TOP_K_FINAL,
    ) -> List[RerankedResult]:
        """
        Re-rank candidates using cross-encoder scoring.

        Args:
            query:      User question
            candidates: Results from HybridRetriever.retrieve()
            top_k:      Number of final results to return

        Returns:
            List of RerankedResult sorted by cross-encoder score (descending)
        """
        if not candidates:
            return []

        # Build (query, passage) pairs for cross-encoder
        pairs = [(query, c.chunk.text) for c in candidates]

        # Score all pairs in one batch
        scores = self.model.predict(
            pairs,
            batch_size=32,
            show_progress_bar=False,
        )

        # Sort by cross-encoder score (descending)
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Build final results
        results = []
        for rank, (candidate, ce_score) in enumerate(scored[:top_k], start=1):
            results.append(RerankedResult(
                chunk          = candidate.chunk,
                reranker_score = float(ce_score),
                rrf_score      = candidate.rrf_score,
                dense_rank     = candidate.dense_rank,
                sparse_rank    = candidate.sparse_rank,
                final_rank     = rank,
            ))

        return results

    def rerank_with_metadata(
        self,
        query: str,
        candidates: list,
        top_k: int = TOP_K_FINAL,
    ) -> dict:
        """
        Re-rank and return both results + diagnostic metadata.
        Useful for evaluation and ablation studies.

        Returns:
            {
                "results": List[RerankedResult],
                "metadata": {
                    "n_candidates": int,
                    "top_k": int,
                    "reranker_model": str,
                    "rank_changes": List[dict],  # how ranks shifted
                    "avg_score": float,
                    "score_range": (float, float),
                }
            }
        """
        results = self.rerank(query, candidates, top_k)

        # Track rank changes for analysis
        rank_changes = []
        for r in results:
            # Find original RRF rank
            original_rank = next(
                (i + 1 for i, c in enumerate(candidates)
                 if c.chunk.chunk_id == r.chunk.chunk_id),
                None
            )
            rank_changes.append({
                "chunk_id":      r.chunk.chunk_id,
                "rrf_rank":      original_rank,
                "reranked_rank": r.final_rank,
                "rank_delta":    (original_rank - r.final_rank) if original_rank else None,
                "reranker_score": r.reranker_score,
                "rrf_score":     r.rrf_score,
            })

        scores = [r.reranker_score for r in results]

        return {
            "results": results,
            "metadata": {
                "n_candidates":  len(candidates),
                "top_k":         top_k,
                "reranker_model": self.model_name,
                "rank_changes":  rank_changes,
                "avg_score":     float(np.mean(scores)) if scores else 0.0,
                "score_range":   (float(min(scores)), float(max(scores))) if scores else (0.0, 0.0),
            }
        }


class QueryExpander:
    """
    Generates multiple query variants to improve retrieval recall.

    Techniques:
        1. LLM-based expansion: ask the LLM to generate alternative phrasings
        2. Legal synonym injection: replace common terms with legal equivalents
        3. Hypothetical document (HyDE): generate a hypothetical answer, embed that

    For portfolio projects, method 1 (LLM expansion) is the most impressive
    because it shows you understand the query-document vocabulary gap problem.
    """

    # Legal term synonyms for rule-based expansion
    LEGAL_SYNONYMS = {
        "indemnification": ["indemnity", "hold harmless", "compensation", "reimbursement"],
        "termination":     ["cancellation", "expiration", "dissolution", "ending"],
        "confidentiality": ["non-disclosure", "NDA", "trade secret", "proprietary"],
        "liability":       ["responsibility", "obligation", "damages", "culpability"],
        "breach":          ["violation", "default", "non-compliance", "infringement"],
        "governing law":   ["jurisdiction", "applicable law", "choice of law", "venue"],
        "warranty":        ["guarantee", "representation", "assurance", "covenant"],
        "assignment":      ["transfer", "delegation", "conveyance", "novation"],
        "force majeure":   ["act of god", "unforeseeable", "extraordinary event"],
        "arbitration":     ["dispute resolution", "mediation", "ADR", "tribunal"],
    }

    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm

    def expand(self, query: str, n_variants: int = 3) -> List[str]:
        """
        Generate query variants for improved retrieval recall.

        Returns the original query + n_variants expansions.
        """
        variants = [query]

        # Rule-based: inject legal synonyms
        synonyms = self._synonym_expansion(query)
        if synonyms:
            variants.append(synonyms)

        # LLM-based expansion (optional, requires API call)
        if self.use_llm:
            llm_variants = self._llm_expansion(query, n_variants - 1)
            variants.extend(llm_variants)

        return variants[:n_variants + 1]

    def _synonym_expansion(self, query: str) -> Optional[str]:
        """Replace legal terms with synonyms."""
        query_lower = query.lower()
        expanded = query

        for term, synonyms in self.LEGAL_SYNONYMS.items():
            if term in query_lower:
                # Add the first 2 synonyms to the query
                synonym_str = " ".join(synonyms[:2])
                expanded = f"{query} {synonym_str}"
                break  # Only expand one term to avoid query drift

        return expanded if expanded != query else None

    def _llm_expansion(self, query: str, n: int = 2) -> List[str]:
        """
        Use the LLM to generate semantically similar query variants.
        This helps bridge the vocabulary gap between user questions
        and legal document language.
        """
        try:
            from src.generation import _call_openai, _call_groq
            from src.config import LLM_PROVIDER

            prompt = f"""Generate {n} alternative phrasings of this legal question.
Each variant should use different legal terminology but ask the same thing.
Return ONLY the variants, one per line, no numbering.

Original question: {query}"""

            if LLM_PROVIDER == "openai":
                response = _call_openai(
                    system_prompt="You are a legal query reformulation assistant.",
                    user_prompt=prompt,
                    max_tokens=200,
                )
            else:
                response = _call_groq(
                    system_prompt="You are a legal query reformulation assistant.",
                    user_prompt=prompt,
                    max_tokens=200,
                )

            variants = [
                line.strip() for line in response.strip().split("\n")
                if line.strip() and len(line.strip()) > 10
            ]
            return variants[:n]

        except Exception as e:
            print(f"LLM query expansion failed: {e}")
            return []


def build_enhanced_retrieval_pipeline(
    retriever,          # HybridRetriever
    reranker: Optional[CrossEncoderReranker] = None,
    query_expander: Optional[QueryExpander] = None,
):
    """
    Factory function: returns a retrieve() callable that chains
    query expansion → hybrid retrieval → re-ranking.

    This is the "enhanced pipeline" you reference in your README.
    """

    if reranker is None:
        reranker = CrossEncoderReranker()
    if query_expander is None:
        query_expander = QueryExpander(use_llm=False)

    def retrieve(
        query: str,
        top_k: int = TOP_K_FINAL,
        doc_type_filter: Optional[str] = None,
        use_reranker: bool = True,
        use_expansion: bool = True,
    ) -> List[RerankedResult]:
        """
        Full enhanced retrieval pipeline:
            1. Query expansion (synonym injection)
            2. Hybrid retrieval (dense + BM25 + RRF) over all variants
            3. Cross-encoder re-ranking

        Args:
            query:           Original user query
            top_k:           Final number of chunks to return
            doc_type_filter: Filter by document type
            use_reranker:    Enable/disable re-ranking (for ablation)
            use_expansion:   Enable/disable query expansion (for ablation)
        """
        # Step 1: Expand query
        if use_expansion:
            queries = query_expander.expand(query, n_variants=2)
        else:
            queries = [query]

        # Step 2: Retrieve from all query variants
        all_candidates = []
        seen_ids = set()

        for q in queries:
            results = retriever.retrieve(
                q,
                top_k_final=TOP_K_RERANK,
                doc_type_filter=doc_type_filter,
            )
            for r in results:
                if r.chunk.chunk_id not in seen_ids:
                    all_candidates.append(r)
                    seen_ids.add(r.chunk.chunk_id)

        # Step 3: Re-rank
        if use_reranker and len(all_candidates) > top_k:
            return reranker.rerank(query, all_candidates, top_k)
        else:
            # No re-ranking — just return top-k by RRF
            return [
                RerankedResult(
                    chunk=c.chunk,
                    reranker_score=0.0,
                    rrf_score=c.rrf_score,
                    dense_rank=c.dense_rank,
                    sparse_rank=c.sparse_rank,
                    final_rank=i + 1,
                )
                for i, c in enumerate(all_candidates[:top_k])
            ]

    return retrieve