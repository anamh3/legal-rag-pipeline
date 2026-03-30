"""
tests/test_ingestion.py — Unit tests for the ingestion module.
"""
import tempfile
from pathlib import Path

import pytest

from src.ingestion import (
    _clean_text, _split_into_clauses, _detect_doc_type,
    count_tokens, split_by_tokens, chunk_document, Chunk
)


# ── _clean_text ───────────────────────────────────────────────────────────────

def test_clean_text_removes_page_numbers():
    text = "Some legal text.\n\n42\n\nMore text."
    result = _clean_text(text)
    assert "42" not in result or "42" in result.replace("\n42\n", "")


def test_clean_text_collapses_blank_lines():
    text = "Para 1.\n\n\n\n\nPara 2."
    result = _clean_text(text)
    assert "\n\n\n" not in result


def test_clean_text_removes_repeated_headers():
    header = "CONFIDENTIAL"
    text = "\n".join([header] * 5 + ["Actual content."])
    result = _clean_text(text)
    # Header repeated 5 times should be removed
    assert result.count(header) < 3


# ── _split_into_clauses ───────────────────────────────────────────────────────

def test_split_detects_numbered_sections():
    text = (
        "1. Definitions\nHerein, 'Company' means Acme Corp.\n\n"
        "2. Payment Terms\nPayment is due within 30 days.\n\n"
        "3. Termination\nEither party may terminate with 30 days notice."
    )
    clauses = _split_into_clauses(text)
    assert len(clauses) >= 2


def test_split_returns_list_of_tuples():
    text = "Section 1. Introduction\nThis agreement governs..."
    clauses = _split_into_clauses(text)
    assert isinstance(clauses, list)
    assert all(isinstance(c, tuple) and len(c) == 2 for c in clauses)


def test_split_no_headers_returns_single_chunk():
    text = "This is a plain paragraph without any section headers."
    clauses = _split_into_clauses(text)
    assert len(clauses) == 1
    assert clauses[0][0] == ""


# ── _detect_doc_type ──────────────────────────────────────────────────────────

def test_detect_contract_from_filename():
    assert _detect_doc_type("service_agreement.pdf", "") == "contract"

def test_detect_contract_from_text():
    text = "WHEREAS the parties hereto agree to the following terms..."
    assert _detect_doc_type("document.pdf", text) == "contract"

def test_detect_case_law_from_text():
    text = "The plaintiff filed a motion for summary judgment. The defendant opposed..."
    assert _detect_doc_type("opinion.pdf", text) == "case_law"

def test_detect_other_fallback():
    assert _detect_doc_type("random.pdf", "Some random text.") == "other"


# ── Token counting ────────────────────────────────────────────────────────────

def test_count_tokens_returns_int():
    result = count_tokens("Hello world, this is a test.")
    assert isinstance(result, int)
    assert result > 0

def test_count_tokens_empty_string():
    assert count_tokens("") == 0

def test_split_by_tokens_respects_max():
    long_text = "word " * 1000
    chunks = split_by_tokens(long_text, max_tokens=100, overlap=10)
    for chunk in chunks:
        assert count_tokens(chunk) <= 110  # small tolerance for split boundaries


# ── chunk_document ────────────────────────────────────────────────────────────

def test_chunk_document_txt():
    """Test chunking a plain text file."""
    content = "\n\n".join([
        f"1.{i} Section {i}\nThis is the content of section {i}. " * 20
        for i in range(1, 6)
    ])
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write(content)
        tmp_path = Path(f.name)

    try:
        chunks = chunk_document(tmp_path, chunk_size=200, overlap=20)
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.token_count <= 220 for c in chunks)  # with tolerance
        assert all(c.source_file == tmp_path.name for c in chunks)
    finally:
        tmp_path.unlink()


def test_chunk_ids_are_unique():
    content = "Content section. " * 500
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write(content)
        tmp_path = Path(f.name)

    try:
        chunks = chunk_document(tmp_path, chunk_size=100, overlap=10)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"
    finally:
        tmp_path.unlink()
