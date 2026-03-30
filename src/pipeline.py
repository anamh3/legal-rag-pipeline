"""
pipeline.py — End-to-end RAG pipeline orchestrator.

Combines retrieval → generation → hallucination detection into a single
callable interface. This is the main entry point for the application.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.config import TOP_K_FINAL, VECTORSTORE_DIR
from src.ingestion import Chunk
from src.retrieval import HybridRetriever, RetrievalResult
from src.generation import GenerationResult, generate_answer
from src.hallucination import FaithfulnessResult, compute_faithfulness


@dataclass
class PipelineResult:
    """Complete result from a single RAG query."""
    query:            str
    retrieval_results: List[RetrievalResult]
    generation:       GenerationResult
    faithfulness:     FaithfulnessResult

    @property
    def answer(self) -> str:
        return self.generation.answer

    @property
    def faithfulness_score(self) -> float:
        return self.faithfulness.faithfulness_score

    @property
    def is_flagged(self) -> bool:
        return self.faithfulness.is_flagged

    @property
    def retrieved_chunks(self) -> List[Chunk]:
        return [r.chunk for r in self.retrieval_results]

    def format_for_display(self) -> dict:
        """Structured dict for Gradio / API responses."""
        return {
            "query":             self.query,
            "answer":            self.answer,
            "faithfulness_score": round(self.faithfulness_score, 3),
            "is_flagged":        self.is_flagged,
            "n_claims":          len(self.faithfulness.claims),
            "n_grounded":        self.faithfulness.n_grounded,
            "n_ungrounded":      self.faithfulness.n_ungrounded,
            "n_contradicted":    self.faithfulness.n_contradicted,
            "sources": [
                {
                    "chunk_id":    r.chunk.chunk_id,
                    "source_file": r.chunk.source_file,
                    "section":     r.chunk.section,
                    "doc_type":    r.chunk.doc_type,
                    "page_num":    r.chunk.page_num,
                    "rrf_score":   round(r.rrf_score, 4),
                    "text_preview": r.chunk.text[:300] + "..."
                        if len(r.chunk.text) > 300 else r.chunk.text,
                }
                for r in self.retrieval_results
            ],
            "claim_breakdown": [
                {
                    "claim":      c.claim[:200],
                    "grounded":   c.is_grounded,
                    "entailment": round(c.best_entailment_prob, 3),
                    "label":      c.best_label.value,
                    "status":     c.status_emoji,
                }
                for c in self.faithfulness.claims
            ],
            "model": self.generation.model,
        }


class RAGPipeline:
    """
    Main pipeline: retrieve → generate → verify faithfulness.

    Usage:
        pipeline = RAGPipeline.load()
        result = pipeline.query("What are the indemnification clauses?")
        print(result.answer)
        print(f"Faithfulness: {result.faithfulness_score:.2f}")
    """

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    @classmethod
    def load(cls, vectorstore_dir=VECTORSTORE_DIR) -> "RAGPipeline":
        """Load from persisted vector store."""
        retriever = HybridRetriever.load(vectorstore_dir)
        return cls(retriever)

    def query(
        self,
        question: str,
        top_k: int = TOP_K_FINAL,
        doc_type_filter: Optional[str] = None,
        temperature: float = 0.0,
        skip_hallucination_check: bool = False,
    ) -> PipelineResult:
        """
        Run the full RAG pipeline for a question.

        Args:
            question:                 The user's query
            top_k:                    Number of chunks to retrieve and pass to LLM
            doc_type_filter:          Restrict retrieval to a document type
            temperature:              LLM temperature
            skip_hallucination_check: Set True for faster inference (evaluation mode)

        Returns:
            PipelineResult with answer + faithfulness breakdown
        """
        # Step 1: Hybrid retrieval
        retrieval_results = self.retriever.retrieve(
            query           = question,
            top_k_final     = top_k,
            doc_type_filter = doc_type_filter,
        )
        chunks = [r.chunk for r in retrieval_results]

        # Step 2: Generate cited answer
        generation = generate_answer(
            query       = question,
            chunks      = chunks,
            temperature = temperature,
        )

        # Step 3: Faithfulness check
        if skip_hallucination_check:
            # Dummy result for speed
            from src.hallucination import FaithfulnessResult
            faithfulness = FaithfulnessResult(
                query=question, answer=generation.answer, chunks=chunks,
                claims=[], faithfulness_score=1.0, is_flagged=False,
                n_grounded=0, n_ungrounded=0, n_contradicted=0,
            )
        else:
            faithfulness = compute_faithfulness(
                query  = question,
                answer = generation.answer,
                chunks = chunks,
            )

        return PipelineResult(
            query             = question,
            retrieval_results = retrieval_results,
            generation        = generation,
            faithfulness      = faithfulness,
        )

    def batch_query(
        self,
        questions: List[str],
        skip_hallucination_check: bool = False,
        **kwargs,
    ) -> List[PipelineResult]:
        """Run pipeline on multiple questions. Used by evaluation."""
        results = []
        from tqdm import tqdm
        for q in tqdm(questions, desc="Running pipeline"):
            result = self.query(
                q,
                skip_hallucination_check=skip_hallucination_check,
                **kwargs,
            )
            results.append(result)
        return results
