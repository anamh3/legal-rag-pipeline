#!/usr/bin/env python3
"""
ingest.py — CLI to parse documents and build the vector store.

Usage:
    python ingest.py --input data/raw/              # ingest all files
    python ingest.py --input data/raw/contract.pdf  # ingest one file
    python ingest.py --input data/raw/ --reset      # wipe and rebuild
    python ingest.py --input data/raw/ --chunk-size 300 --workers 4
"""
import argparse
import shutil
import sys
from pathlib import Path

from rich.console import Console

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Ingest legal documents into the RAG vector store."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to a directory of documents, or a single file."
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Wipe the existing vector store and rebuild from scratch."
    )
    parser.add_argument(
        "--chunk-size", type=int, default=None,
        help="Max tokens per chunk (default from .env or 400)."
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel ingestion workers (default: 1)."
    )
    args = parser.parse_args()

    from src.config import VECTORSTORE_DIR, PROCESSED_DIR, CHUNK_SIZE, CHUNK_OVERLAP
    from src.ingestion import ingest_directory
    from src.embeddings import VectorStore
    from src.retrieval import HybridRetriever, BM25Index

    input_path  = Path(args.input)
    chunk_size  = args.chunk_size or CHUNK_SIZE

    if not input_path.exists():
        console.print(f"[red]Input path does not exist: {input_path}[/red]")
        sys.exit(1)

    # Handle single-file input
    if input_path.is_file():
        input_dir = input_path.parent
    else:
        input_dir = input_path

    # Optionally wipe existing indices
    if args.reset:
        console.print("[yellow]Resetting vector store ...[/yellow]")
        for directory in (VECTORSTORE_DIR, PROCESSED_DIR):
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)

    # Step 1: Parse + chunk documents
    console.rule("[bold]Step 1/3 — Parsing & chunking documents")
    chunks = ingest_directory(
        input_dir  = input_dir,
        output_dir = PROCESSED_DIR,
        chunk_size = chunk_size,
        overlap    = CHUNK_OVERLAP,
    )

    if not chunks:
        console.print("[red]No chunks produced. Check your documents.[/red]")
        sys.exit(1)

    # Step 2: Build FAISS index
    console.rule("[bold]Step 2/3 — Building FAISS (dense) index")
    vector_store = VectorStore()
    vector_store.build_from_corpus(chunks)

    # Step 3: Build BM25 index + save everything
    console.rule("[bold]Step 3/3 — Building BM25 (sparse) index & saving")
    bm25 = BM25Index(chunks)
    console.print(f"BM25 index built: {len(chunks)} documents")

    retriever = HybridRetriever(vector_store, bm25)
    retriever.save(VECTORSTORE_DIR)

    console.print()
    console.rule("[green bold]Ingestion complete")
    console.print(f"  Documents chunked:  {len({c.source_file for c in chunks})}")
    console.print(f"  Total chunks:       {len(chunks)}")
    console.print(f"  Vector store:       {VECTORSTORE_DIR}")
    console.print()
    console.print("Next step: [bold]python app.py[/bold]")


if __name__ == "__main__":
    main()
