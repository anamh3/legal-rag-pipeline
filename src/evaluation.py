"""
evaluation.py — Comprehensive evaluation harness.

Metrics computed:
  - Faithfulness (our NLI detector)
  - Context Recall (RAGAS)
  - Context Precision (RAGAS)
  - Answer Relevancy (RAGAS)
  - Hallucination Rate by doc type (custom breakdown)
  - Hallucination Rate by legal category (custom breakdown)
  - Ablation: dense vs sparse vs hybrid vs hybrid+reranker (MRR@k, Recall@k, NDCG@k, Precision@k)
  - Caught hallucination examples with NLI scores (portfolio showcase)
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from src.config import OPENAI_API_KEY, PROCESSED_DIR
from src.ingestion import Chunk, load_corpus
from src.pipeline import RAGPipeline, PipelineResult
from src.generation import generate_synthetic_question
from src.hallucination import faithfulness_score_only


console = Console()


# ── Test set generation ───────────────────────────────────────────────────────

@dataclass
class QAPair:
    question:       str
    ground_truth:   str   # the source chunk text (used as reference answer)
    source_chunk_id: str
    doc_type:       str
    category:       str = "general"  # legal category for analysis


def _categorize_chunk(text: str) -> str:
    """Categorize chunk by legal topic for analysis breakdown."""
    text_lower = text.lower()
    categories = {
        "indemnification": ["indemnif", "hold harmless", "defend"],
        "termination":     ["terminat", "expir", "cancel"],
        "confidentiality": ["confidential", "non-disclosure", "proprietary"],
        "liability":       ["liabilit", "limit of liabilit", "damages"],
        "governing_law":   ["governing law", "jurisdiction", "venue"],
        "warranty":        ["warrant", "guarantee", "representation"],
        "ip":              ["intellectual property", "patent", "copyright", "trademark"],
        "payment":         ["payment", "fee", "invoice", "compensation"],
    }
    for cat, keywords in categories.items():
        if any(kw in text_lower for kw in keywords):
            return cat
    return "general"


def generate_test_set(
    chunks: List[Chunk],
    n_questions: int = 50,
    seed: int = 42,
) -> List[QAPair]:
    """
    Synthesize a QA test set by generating questions from random chunks.

    For each selected chunk:
    - Generate a question whose answer is in the chunk (using the LLM)
    - Use the chunk text as the ground truth answer

    This gives us a labeled dataset without manual annotation.
    """
    random.seed(seed)

    # Stratify by doc type if possible
    doc_types = list({c.doc_type for c in chunks})
    per_type  = n_questions // max(len(doc_types), 1)

    selected: List[Chunk] = []
    for dt in doc_types:
        dt_chunks = [c for c in chunks if c.doc_type == dt and c.token_count > 80]
        selected.extend(random.sample(dt_chunks, min(per_type, len(dt_chunks))))

    # Fill remainder from all chunks
    remaining = n_questions - len(selected)
    pool = [c for c in chunks if c not in selected and c.token_count > 80]
    selected.extend(random.sample(pool, min(remaining, len(pool))))
    selected = selected[:n_questions]

    qa_pairs: List[QAPair] = []
    print(f"Generating {len(selected)} questions from corpus ...")
    for chunk in tqdm(selected, desc="Synthesizing QA pairs"):
        try:
            question = generate_synthetic_question(chunk)
            qa_pairs.append(QAPair(
                question        = question,
                ground_truth    = chunk.text,
                source_chunk_id = chunk.chunk_id,
                doc_type        = chunk.doc_type,
                category        = _categorize_chunk(chunk.text),
            ))
            time.sleep(0.3)  # rate limit buffer
        except Exception as e:
            tqdm.write(f"  Skipped chunk {chunk.chunk_id}: {e}")

    return qa_pairs


def save_test_set(qa_pairs: List[QAPair], path: Path) -> None:
    with open(path, "w") as f:
        json.dump(
            [{"question": q.question, "ground_truth": q.ground_truth,
              "source_chunk_id": q.source_chunk_id, "doc_type": q.doc_type,
              "category": q.category}
             for q in qa_pairs],
            f, indent=2
        )
    print(f"Test set saved: {path} ({len(qa_pairs)} pairs)")


def load_test_set(path: Path) -> List[QAPair]:
    with open(path) as f:
        data = json.load(f)
    # Backward compatibility: add category if missing
    for d in data:
        if "category" not in d:
            d["category"] = _categorize_chunk(d.get("ground_truth", ""))
    return [QAPair(**d) for d in data]


# ── RAGAS evaluation ──────────────────────────────────────────────────────────

def run_ragas_evaluation(
    qa_pairs: List[QAPair],
    pipeline_results: List[PipelineResult],
    delay: float = 0.5,
) -> Dict[str, float]:
    """
    Run RAGAS metrics on the pipeline results.

    RAGAS requires: question, answer, contexts, ground_truths
    It internally calls the OpenAI API for LLM-based metrics.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness as ragas_faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        )
        from datasets import Dataset
    except ImportError:
        console.print("[yellow]RAGAS not available. Skipping RAGAS metrics.[/yellow]")
        return {}

    data = {
        "question":      [qa.question for qa in qa_pairs],
        "answer":        [r.answer for r in pipeline_results],
        "contexts":      [[c.text for c in r.retrieved_chunks] for r in pipeline_results],
        "ground_truths": [[qa.ground_truth] for qa in qa_pairs],
    }
    dataset = Dataset.from_dict(data)

    console.print("Running RAGAS evaluation (calls OpenAI API) ...")
    try:
        result = evaluate(
            dataset,
            metrics=[
                ragas_faithfulness,
                answer_relevancy,
                context_recall,
                context_precision,
            ],
        )
        return {
            "ragas_faithfulness":      float(result["faithfulness"]),
            "ragas_answer_relevancy":  float(result["answer_relevancy"]),
            "ragas_context_recall":    float(result["context_recall"]),
            "ragas_context_precision": float(result["context_precision"]),
        }
    except Exception as e:
        console.print(f"[red]RAGAS evaluation failed: {e}[/red]")
        return {}


# ── Custom metrics ────────────────────────────────────────────────────────────

def compute_custom_metrics(
    qa_pairs: List[QAPair],
    pipeline_results: List[PipelineResult],
) -> Dict[str, float]:
    """
    Compute our custom NLI-based faithfulness + breakdown by doc type.
    """
    scores    = [r.faithfulness_score for r in pipeline_results]
    flagged   = [r.is_flagged for r in pipeline_results]
    doc_types = [qa.doc_type for qa in qa_pairs]

    # Overall
    metrics = {
        "nli_faithfulness":    sum(scores) / len(scores) if scores else 0.0,
        "hallucination_rate":  sum(flagged) / len(flagged) if flagged else 0.0,
    }

    # By doc type
    type_scores: Dict[str, List[float]] = {}
    for dt, score in zip(doc_types, scores):
        type_scores.setdefault(dt, []).append(score)

    for dt, type_score_list in type_scores.items():
        metrics[f"faithfulness_{dt}"] = (
            sum(type_score_list) / len(type_score_list)
        )

    return metrics


# ── Retrieval metrics (individual) ────────────────────────────────────────────

def _reciprocal_rank(retrieved_ids: List[str], target_id: str) -> float:
    """Reciprocal rank of a single target in retrieved list."""
    if target_id in retrieved_ids:
        return 1.0 / (retrieved_ids.index(target_id) + 1)
    return 0.0


def _ndcg_at_k(retrieved_ids: List[str], target_id: str, k: int) -> float:
    """NDCG@k for a single relevant document."""
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid == target_id:
            return 1.0 / np.log2(i + 2)  # DCG / ideal DCG (ideal = 1/log2(2) = 1.0)
    return 0.0


def _precision_at_k(retrieved_ids: List[str], target_id: str, k: int) -> float:
    """Precision@k: is the target in the top-k?"""
    return 1.0 / k if target_id in retrieved_ids[:k] else 0.0


# ── Ablation: retrieval strategies ────────────────────────────────────────────

def compute_retrieval_metrics(
    qa_pairs: List[QAPair],
    pipeline: RAGPipeline,
    top_k: int = 5,
    include_reranker: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Ablation: compare dense vs sparse vs hybrid vs hybrid+reranker.

    For each query, checks whether the ground-truth chunk appears
    in the top-k results of each retrieval strategy.

    Metrics:
    - Recall@k:    fraction of queries where ground-truth is in top-k
    - MRR@k:       mean reciprocal rank of ground-truth in top-k
    - NDCG@k:      normalized discounted cumulative gain
    - Precision@k: fraction of top-k that are relevant
    """
    strategies = ["dense", "sparse", "hybrid"]

    # Try to load re-ranker for 4th strategy
    reranker = None
    if include_reranker:
        try:
            from src.reranker import CrossEncoderReranker
            reranker = CrossEncoderReranker()
            strategies.append("hybrid_reranked")
            console.print("[green]Re-ranker loaded for ablation[/green]")
        except ImportError:
            console.print("[yellow]Re-ranker not available (src/reranker.py missing). "
                          "Running 3-strategy ablation.[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Re-ranker failed to load: {e}[/yellow]")

    hit_counts = {s: 0 for s in strategies}
    rr_sums    = {s: 0.0 for s in strategies}
    ndcg_sums  = {s: 0.0 for s in strategies}
    prec_sums  = {s: 0.0 for s in strategies}
    latencies  = {s: [] for s in strategies}
    n = 0

    for qa in tqdm(qa_pairs, desc="Ablation retrieval eval"):
        # Get results from all base strategies
        start = time.time()
        results = pipeline.retriever.retrieve_for_ablation(qa.question, top_k=top_k)
        base_latency = (time.time() - start) * 1000

        n += 1

        for strategy in ["dense", "sparse", "hybrid"]:
            chunk_ids = [c.chunk_id for c in results[strategy]]

            if qa.source_chunk_id in chunk_ids:
                hit_counts[strategy] += 1
                rr_sums[strategy] += _reciprocal_rank(chunk_ids, qa.source_chunk_id)
            ndcg_sums[strategy] += _ndcg_at_k(chunk_ids, qa.source_chunk_id, top_k)
            prec_sums[strategy] += _precision_at_k(chunk_ids, qa.source_chunk_id, top_k)
            latencies[strategy].append(base_latency)

        # Re-ranker strategy (4th)
        if reranker is not None and "hybrid_reranked" in strategies:
            start = time.time()
            # Get more candidates for re-ranking
            hybrid_candidates = pipeline.retriever.retrieve(
                qa.question, top_k_final=20
            )
            reranked = reranker.rerank(qa.question, hybrid_candidates, top_k=top_k)
            rerank_latency = (time.time() - start) * 1000

            chunk_ids = [r.chunk.chunk_id for r in reranked]

            if qa.source_chunk_id in chunk_ids:
                hit_counts["hybrid_reranked"] += 1
                rr_sums["hybrid_reranked"] += _reciprocal_rank(
                    chunk_ids, qa.source_chunk_id
                )
            ndcg_sums["hybrid_reranked"] += _ndcg_at_k(
                chunk_ids, qa.source_chunk_id, top_k
            )
            prec_sums["hybrid_reranked"] += _precision_at_k(
                chunk_ids, qa.source_chunk_id, top_k
            )
            latencies["hybrid_reranked"].append(rerank_latency)

    ablation: Dict[str, Dict[str, float]] = {}
    for s in strategies:
        avg_lat = np.mean(latencies[s]) if latencies[s] else 0.0
        ablation[s] = {
            f"Recall@{top_k}":    round(hit_counts[s] / n, 3) if n > 0 else 0.0,
            f"MRR@{top_k}":       round(rr_sums[s] / n, 3) if n > 0 else 0.0,
            f"NDCG@{top_k}":      round(ndcg_sums[s] / n, 3) if n > 0 else 0.0,
            f"Precision@{top_k}": round(prec_sums[s] / n, 3) if n > 0 else 0.0,
            "Latency (ms)":       round(float(avg_lat), 1),
        }

    return ablation


# ── Hallucination deep analysis ───────────────────────────────────────────────

@dataclass
class CaughtHallucination:
    """A caught hallucination example — portfolio showcase material."""
    query: str
    faithfulness_score: float
    answer_preview: str
    ungrounded_claims: List[Dict[str, str]]  # claim text + NLI label + score
    doc_type: str
    category: str


def run_hallucination_analysis(
    qa_pairs: List[QAPair],
    pipeline_results: List[PipelineResult],
) -> Dict:
    """
    Deep analysis of hallucination patterns.

    Returns breakdown by doc type, by legal category,
    and top caught hallucination examples with NLI scores.

    This is the data that makes your portfolio README stand out.
    """
    # By doc type
    by_doc_type: Dict[str, List[float]] = {}
    for qa, r in zip(qa_pairs, pipeline_results):
        by_doc_type.setdefault(qa.doc_type, []).append(r.faithfulness_score)

    doc_type_analysis = {
        dt: {
            "faithfulness": round(float(np.mean(scores)), 3),
            "hallucination_rate": round(1.0 - float(np.mean(scores)), 3),
            "n_queries": len(scores),
        }
        for dt, scores in by_doc_type.items()
    }

    # By legal category
    by_category: Dict[str, List[float]] = {}
    for qa, r in zip(qa_pairs, pipeline_results):
        by_category.setdefault(qa.category, []).append(r.faithfulness_score)

    category_analysis = {
        cat: {
            "faithfulness": round(float(np.mean(scores)), 3),
            "hallucination_rate": round(1.0 - float(np.mean(scores)), 3),
            "n_queries": len(scores),
        }
        for cat, scores in by_category.items()
    }

    # Caught hallucinations — examples with ungrounded claims
    caught: List[CaughtHallucination] = []
    for qa, r in zip(qa_pairs, pipeline_results):
        if r.is_flagged and hasattr(r, 'faithfulness') and r.faithfulness.claims:
            ungrounded = []
            for claim in r.faithfulness.claims:
                if not claim.is_grounded:
                    ungrounded.append({
                        "claim": claim.claim,
                        "nli_label": (
                            claim.best_label.value
                            if hasattr(claim.best_label, 'value')
                            else str(claim.best_label)
                        ),
                        "nli_score": round(claim.best_score, 3),
                    })

            if ungrounded:
                caught.append(CaughtHallucination(
                    query=qa.question,
                    faithfulness_score=r.faithfulness_score,
                    answer_preview=r.answer[:300],
                    ungrounded_claims=ungrounded,
                    doc_type=qa.doc_type,
                    category=qa.category,
                ))

    # Sort by worst faithfulness first
    caught.sort(key=lambda x: x.faithfulness_score)

    return {
        "by_doc_type": doc_type_analysis,
        "by_category": category_analysis,
        "caught_hallucinations": [
            {
                "query": c.query,
                "faithfulness_score": c.faithfulness_score,
                "answer_preview": c.answer_preview,
                "ungrounded_claims": c.ungrounded_claims,
                "doc_type": c.doc_type,
                "category": c.category,
            }
            for c in caught[:5]  # Top 5 worst
        ],
        "total_flagged": len(caught),
    }


# ── Display ───────────────────────────────────────────────────────────────────

def print_results_table(
    custom_metrics: Dict[str, float],
    ragas_metrics: Dict[str, float],
    ablation: Optional[Dict] = None,
    hallucination_analysis: Optional[Dict] = None,
) -> None:
    """Pretty-print all evaluation results using Rich."""

    # ── Main metrics table ────────────────────────────────────────────────
    table = Table(title="Evaluation Results", show_header=True, header_style="bold")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Score", justify="right")
    table.add_column("Source", style="dim")

    def add_row(name, val, source):
        if isinstance(val, float):
            color = "green" if val >= 0.75 else ("yellow" if val >= 0.5 else "red")
            table.add_row(name, f"[{color}]{val:.3f}[/{color}]", source)
        else:
            table.add_row(name, str(val), source)

    add_row("NLI Faithfulness",   custom_metrics.get("nli_faithfulness", 0),   "NLI detector")
    add_row("Hallucination Rate", custom_metrics.get("hallucination_rate", 0), "NLI detector")

    if ragas_metrics:
        add_row("RAGAS Faithfulness",      ragas_metrics.get("ragas_faithfulness", 0),      "RAGAS")
        add_row("Answer Relevancy",        ragas_metrics.get("ragas_answer_relevancy", 0),  "RAGAS")
        add_row("Context Recall",          ragas_metrics.get("ragas_context_recall", 0),    "RAGAS")
        add_row("Context Precision",       ragas_metrics.get("ragas_context_precision", 0), "RAGAS")

    # Doc-type breakdown
    for key, val in custom_metrics.items():
        if key.startswith("faithfulness_"):
            dt = key.replace("faithfulness_", "")
            add_row(f"  Faithfulness ({dt})", val, "NLI by type")

    console.print(table)

    # ── Ablation table ────────────────────────────────────────────────────
    if ablation:
        abl_table = Table(
            title="Retrieval Ablation",
            show_header=True,
            header_style="bold",
        )
        abl_table.add_column("Strategy", style="cyan")

        metric_names = list(next(iter(ablation.values())).keys())
        for mn in metric_names:
            abl_table.add_column(mn, justify="right")

        # Find best strategy by MRR
        mrr_key = [k for k in metric_names if k.startswith("MRR")][0]
        best_strategy = max(ablation, key=lambda s: ablation[s][mrr_key])

        for strategy, metrics in ablation.items():
            is_best = strategy == best_strategy
            style = "bold green" if is_best else ""
            marker = " ★" if is_best else ""

            vals = []
            for mn in metric_names:
                v = metrics[mn]
                if mn == "Latency (ms)":
                    vals.append(f"{v:.1f}ms")
                else:
                    vals.append(f"{v:.3f}")

            display_name = {
                "dense": "Dense only",
                "sparse": "Sparse (BM25)",
                "hybrid": "Hybrid (RRF)",
                "hybrid_reranked": "Hybrid + Re-ranker",
            }.get(strategy, strategy)

            abl_table.add_row(f"{display_name}{marker}", *vals, style=style)

        # Compute improvement
        baseline_mrr = ablation.get("dense", {}).get(mrr_key, 0)
        best_mrr = ablation[best_strategy][mrr_key]
        if baseline_mrr > 0:
            improvement = ((best_mrr - baseline_mrr) / baseline_mrr) * 100
            abl_table.add_section()
            abl_table.add_row(
                f"[dim]Improvement over dense-only: +{improvement:.1f}%[/dim]",
                *["" for _ in metric_names],
            )

        console.print(abl_table)

    # ── Hallucination analysis ────────────────────────────────────────────
    if hallucination_analysis:
        _print_hallucination_analysis(hallucination_analysis)


def _print_hallucination_analysis(analysis: Dict) -> None:
    """Pretty-print hallucination deep analysis."""

    # By doc type
    dt_table = Table(
        title="Hallucination by Document Type",
        show_header=True,
        header_style="bold",
    )
    dt_table.add_column("Doc Type", style="cyan")
    dt_table.add_column("Faithfulness", justify="right")
    dt_table.add_column("Hallucination Rate", justify="right")
    dt_table.add_column("N", justify="right", style="dim")

    for dt, data in analysis.get("by_doc_type", {}).items():
        faith = data["faithfulness"]
        hall = data["hallucination_rate"]
        color = "green" if faith >= 0.75 else ("yellow" if faith >= 0.5 else "red")
        dt_table.add_row(
            dt,
            f"[{color}]{faith:.1%}[/{color}]",
            f"{hall:.1%}",
            str(data["n_queries"]),
        )

    console.print(dt_table)

    # By category
    cat_table = Table(
        title="Hallucination by Legal Category",
        show_header=True,
        header_style="bold",
    )
    cat_table.add_column("Category", style="cyan")
    cat_table.add_column("Faithfulness", justify="right")
    cat_table.add_column("Hallucination Rate", justify="right")
    cat_table.add_column("N", justify="right", style="dim")

    for cat, data in analysis.get("by_category", {}).items():
        faith = data["faithfulness"]
        hall = data["hallucination_rate"]
        color = "green" if faith >= 0.75 else ("yellow" if faith >= 0.5 else "red")
        cat_table.add_row(
            cat,
            f"[{color}]{faith:.1%}[/{color}]",
            f"{hall:.1%}",
            str(data["n_queries"]),
        )

    console.print(cat_table)

    # Caught hallucinations
    caught = analysis.get("caught_hallucinations", [])
    if caught:
        console.print("\n[bold red]Caught Hallucinations (Portfolio Examples)[/bold red]")
        console.print("─" * 70)
        for i, h in enumerate(caught[:3], 1):
            console.print(f"\n[bold]Example {i}:[/bold] {h['query'][:70]}")
            console.print(f"  Faithfulness: [red]{h['faithfulness_score']:.1%}[/red]")
            console.print(f"  Doc type: {h['doc_type']} | Category: {h['category']}")
            for claim in h.get("ungrounded_claims", [])[:2]:
                console.print(f"  [red]✗ UNGROUNDED:[/red] \"{claim['claim'][:80]}...\"")
                console.print(
                    f"    NLI: {claim['nli_label']} "
                    f"(score: {claim['nli_score']:.3f})"
                )
        console.print("─" * 70)


# ── Save results ──────────────────────────────────────────────────────────────

def save_evaluation_results(
    custom_metrics: Dict[str, float],
    ragas_metrics: Dict[str, float],
    ablation: Optional[Dict],
    pipeline_results: List[PipelineResult],
    qa_pairs: List[QAPair],
    output_path: Path,
    hallucination_analysis: Optional[Dict] = None,
) -> None:
    results = {
        "summary": {**custom_metrics, **ragas_metrics},
        "ablation": ablation or {},
        "hallucination_analysis": hallucination_analysis or {},
        "per_query": [
            {
                "question":         qa.question,
                "doc_type":         qa.doc_type,
                "category":         qa.category,
                "faithfulness":     r.faithfulness_score,
                "is_flagged":       r.is_flagged,
                "n_claims":         len(r.faithfulness.claims),
                "n_grounded":       r.faithfulness.n_grounded,
                "answer_preview":   r.answer[:200],
            }
            for qa, r in zip(qa_pairs, pipeline_results)
        ]
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"[green]Results saved to {output_path}[/green]")