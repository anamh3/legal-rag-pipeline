"""
tests/test_retrieval.py — Tests for BM25 and RRF retrieval.
tests/test_hallucination.py — Tests for claim extraction and NLI scoring.
"""

# ════════════════════════════════════════════════════════════════════════════════
# Retrieval tests
# ════════════════════════════════════════════════════════════════════════════════

import pytest
from src.ingestion import Chunk
from src.retrieval import BM25Index, reciprocal_rank_fusion


def _make_chunk(chunk_id: str, text: str, doc_type: str = "contract") -> Chunk:
    return Chunk(
        chunk_id    = chunk_id,
        text        = text,
        source_file = "test.pdf",
        section     = "Test Section",
        page_num    = 1,
        doc_type    = doc_type,
        char_start  = 0,
        token_count = len(text.split()),
    )


# ── BM25 ──────────────────────────────────────────────────────────────────────

SAMPLE_CHUNKS = [
    _make_chunk("c1", "indemnification clause protects the party from liability damages"),
    _make_chunk("c2", "termination of contract requires thirty days written notice"),
    _make_chunk("c3", "governing law shall be the state of california jurisdiction"),
    _make_chunk("c4", "payment terms are net thirty days from invoice date"),
    _make_chunk("c5", "confidentiality obligations survive termination of the agreement"),
]


def test_bm25_returns_results():
    idx = BM25Index(SAMPLE_CHUNKS)
    results = idx.search("indemnification liability", top_k=3)
    assert len(results) > 0


def test_bm25_top_result_relevant():
    idx = BM25Index(SAMPLE_CHUNKS)
    results = idx.search("termination notice", top_k=3)
    top_chunk, top_score = results[0]
    assert top_chunk.chunk_id in ("c2", "c5"), "Termination-related chunk should rank first"


def test_bm25_scores_are_positive():
    idx = BM25Index(SAMPLE_CHUNKS)
    results = idx.search("payment invoice", top_k=5)
    for _, score in results:
        assert score >= 0


def test_bm25_respects_top_k():
    idx = BM25Index(SAMPLE_CHUNKS)
    results = idx.search("contract", top_k=2)
    assert len(results) <= 2


def test_bm25_empty_query_doesnt_crash():
    idx = BM25Index(SAMPLE_CHUNKS)
    results = idx.search("", top_k=3)
    assert isinstance(results, list)


# ── RRF ───────────────────────────────────────────────────────────────────────

def test_rrf_merges_lists():
    list_a = [
        (SAMPLE_CHUNKS[0], 0.9),
        (SAMPLE_CHUNKS[1], 0.7),
        (SAMPLE_CHUNKS[2], 0.5),
    ]
    list_b = [
        (SAMPLE_CHUNKS[2], 0.95),   # c3 ranked higher in list B
        (SAMPLE_CHUNKS[0], 0.6),
        (SAMPLE_CHUNKS[3], 0.4),
    ]
    fused = reciprocal_rank_fusion([list_a, list_b], k=60)
    assert len(fused) > 0
    # All unique chunks should appear
    fused_ids = {c.chunk_id for c, _ in fused}
    assert "c1" in fused_ids
    assert "c3" in fused_ids


def test_rrf_scores_are_positive():
    list_a = [(SAMPLE_CHUNKS[0], 0.9)]
    list_b = [(SAMPLE_CHUNKS[0], 0.8)]
    fused = reciprocal_rank_fusion([list_a, list_b])
    _, score = fused[0]
    assert score > 0


def test_rrf_boosts_items_appearing_in_both_lists():
    """Item in both lists should outscore item in only one list."""
    shared = SAMPLE_CHUNKS[0]   # c1
    only_a = SAMPLE_CHUNKS[1]   # c2
    only_b = SAMPLE_CHUNKS[2]   # c3

    list_a = [(shared, 0.5), (only_a, 0.9)]
    list_b = [(shared, 0.5), (only_b, 0.9)]

    fused = reciprocal_rank_fusion([list_a, list_b])
    fused_dict = {c.chunk_id: score for c, score in fused}

    # shared appears in both — should beat items appearing in only one
    assert fused_dict["c1"] > fused_dict.get("c2", 0)
    assert fused_dict["c1"] > fused_dict.get("c3", 0)


def test_rrf_empty_lists():
    fused = reciprocal_rank_fusion([[], []])
    assert fused == []


# ════════════════════════════════════════════════════════════════════════════════
# Hallucination tests (no model loading — unit tests only)
# ════════════════════════════════════════════════════════════════════════════════

from src.hallucination import extract_claims


def test_extract_claims_splits_sentences():
    answer = (
        "The agreement requires 30 days notice for termination [Chunk 1]. "
        "Payment is due within 30 days of invoice [Chunk 2]. "
        "Confidentiality obligations survive termination [Chunk 3]."
    )
    claims = extract_claims(answer)
    assert len(claims) >= 2


def test_extract_claims_removes_citations():
    answer = "Indemnification is required [Chunk 1]. Liability is capped at $1M [Chunk 2]."
    claims = extract_claims(answer)
    for claim in claims:
        assert "[Chunk" not in claim


def test_extract_claims_handles_bullets():
    answer = (
        "Key points:\n"
        "- The party must provide notice\n"
        "- Payment terms are net 30\n"
        "- Governing law is California"
    )
    claims = extract_claims(answer)
    assert len(claims) >= 2


def test_extract_claims_filters_short_fragments():
    answer = "Yes. The indemnification clause requires that each party hold harmless the other."
    claims = extract_claims(answer)
    # "Yes." should be filtered (too short)
    for claim in claims:
        assert len(claim) >= 20


def test_extract_claims_single_sentence():
    answer = "The contract requires written consent for assignment of rights."
    claims = extract_claims(answer)
    assert len(claims) == 1
    assert claims[0] == answer


def test_extract_claims_empty_string():
    claims = extract_claims("")
    # Should return something (the full string) rather than crashing
    assert isinstance(claims, list)
