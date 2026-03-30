#!/usr/bin/env python3
"""
evaluate.py — Run the full evaluation suite.

Usage:
    python evaluate.py --generate-testset --n-questions 50
    python evaluate.py --testset test_set.json
    python evaluate.py --testset test_set.json --ablation
    python evaluate.py --testset test_set.json --skip-ragas
"""
import argparse
import sys
from pathlib import Path

from rich.console import Console

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the Legal RAG pipeline."
    )
    parser.add_argument(
        "--generate-testset", action="store_true",
        help="Generate a synthetic QA test set from the corpus."
    )
    parser.add_argument(
        "--n-questions", type=int, default=50,
        help="Number of QA pairs to generate (default: 50)."
    )
    parser.add_argument(
        "--testset", type=str, default="test_set.json",
        help="Path to existing test set JSON (default: test_set.json)."
    )
    parser.add_argument(
        "--ablation", action="store_true",
        help="Run retrieval ablation (dense vs sparse vs hybrid)."
    )
    parser.add_argument(
        "--skip-ragas", action="store_true",
        help="Skip RAGAS metrics (faster, no extra API calls)."
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Delay between API calls in seconds (default: 0.5)."
    )
    parser.add_argument(
        "--output", type=str, default="evaluation_results.json",
        help="Output path for results JSON."
    )
    args = parser.parse_args()

    from src.ingestion import load_corpus
    from src.pipeline import RAGPipeline
    from src.evaluation import (
        generate_test_set, save_test_set, load_test_set,
        run_ragas_evaluation, compute_custom_metrics,
        compute_retrieval_metrics, print_results_table,
        save_evaluation_results,
    )

    testset_path = Path(args.testset)
    output_path  = Path(args.output)

    # Load or generate test set
    if args.generate_testset or not testset_path.exists():
        console.rule("[bold]Generating test set")
        chunks = load_corpus()
        qa_pairs = generate_test_set(chunks, n_questions=args.n_questions)
        save_test_set(qa_pairs, testset_path)
    else:
        console.print(f"Loading test set from {testset_path}")
        qa_pairs = load_test_set(testset_path)

    console.print(f"Test set: [bold]{len(qa_pairs)}[/bold] QA pairs")

    # Load pipeline
    console.rule("[bold]Loading RAG pipeline")
    pipeline = RAGPipeline.load()

    # Run pipeline on all questions
    console.rule("[bold]Running pipeline on test set")
    pipeline_results = pipeline.batch_query(
        [qa.question for qa in qa_pairs],
        skip_hallucination_check=False,
    )

    # Compute custom metrics
    console.rule("[bold]Computing metrics")
    custom_metrics = compute_custom_metrics(qa_pairs, pipeline_results)

    # RAGAS metrics
    ragas_metrics = {}
    if not args.skip_ragas:
        ragas_metrics = run_ragas_evaluation(qa_pairs, pipeline_results, delay=args.delay)

    # Retrieval ablation
    ablation = None
    if args.ablation:
        console.rule("[bold]Running retrieval ablation")
        ablation = compute_retrieval_metrics(qa_pairs, pipeline)

    # Display results
    console.rule("[bold green]Results")
    print_results_table(custom_metrics, ragas_metrics, ablation)

    # Save
    save_evaluation_results(
        custom_metrics, ragas_metrics, ablation,
        pipeline_results, qa_pairs, output_path
    )


if __name__ == "__main__":
    main()
