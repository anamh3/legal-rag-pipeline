"""
test_legal_rag.py — Comprehensive test suite for the Legal RAG pipeline.

Tests cover:
    1. Ingestion & chunking (5 tests)
    2. BM25 retrieval (5 tests)
    3. Reciprocal Rank Fusion (4 tests)
    4. Re-ranker (4 tests)
    5. Hallucination / claim extraction (5 tests)
    6. Evaluation metrics (5 tests)
    7. API endpoints (5 tests)

Run with:
    pytest tests/test_legal_rag.py -v
"""

import pytest
import json
import numpy as np
from pathlib import Path

from src.ingestion import Chunk
from src.retrieval import BM25Index, reciprocal_rank_fusion


# ── Test helpers ──────────────────────────────────────────────────────────────

def _make_chunk(chunk_id: str, text: str, doc_type: str = "contract",
                source_file: str = "test.pdf", section: str = "Test Section") -> Chunk:
    """Create a test chunk with minimal required fields."""
    return Chunk(
        chunk_id    = chunk_id,
        text        = text,
        source_file = source_file,
        section     = section,
        page_num    = 1,
        doc_type    = doc_type,
        char_start  = 0,
        token_count = len(text.split()),
    )


# Sample chunks representing different legal document content
LEGAL_CHUNKS = [
    _make_chunk("c1",
        "The Licensee shall indemnify and hold harmless the Licensor from any claims, "
        "damages, losses, liabilities, costs and expenses including reasonable attorneys fees "
        "arising out of the Licensee's use of the Software in violation of this Agreement.",
        doc_type="contract", section="Article 7 — Indemnification"),
    _make_chunk("c2",
        "Either party may terminate this Agreement immediately upon written notice if "
        "the other party materially breaches this Agreement and fails to cure such breach "
        "within thirty days after receiving written notice thereof.",
        doc_type="contract", section="Article 9 — Termination"),
    _make_chunk("c3",
        "This Agreement shall be governed by and construed in accordance with the laws "
        "of the State of Delaware without regard to its conflict of laws provisions.",
        doc_type="contract", section="Article 10 — Governing Law"),
    _make_chunk("c4",
        "All fees are due and payable within thirty days of the date of the applicable "
        "invoice. Late payments shall accrue interest at the rate of one and one-half "
        "percent per month or the maximum rate permitted by applicable law.",
        doc_type="contract", section="Article 3 — Payment"),
    _make_chunk("c5",
        "The obligations of confidentiality shall survive the termination or expiration "
        "of this Agreement for a period of five years provided that obligations with "
        "respect to trade secrets shall survive for as long as such information qualifies.",
        doc_type="contract", section="Article 5 — Confidentiality"),
    _make_chunk("c6",
        "During Employee's employment and for a period of twelve months following the "
        "termination of employment for any reason Employee shall not directly or indirectly "
        "engage in any business that competes with the Company's business.",
        doc_type="contract", source_file="employment.txt",
        section="Article 4 — Non-Competition"),
    _make_chunk("c7",
        "The Receiving Party shall protect the Confidential Information using at least "
        "the same degree of care that it uses to protect its own confidential information "
        "of a similar nature but in no event less than reasonable care.",
        doc_type="contract", source_file="nda.txt",
        section="Article 2 — Obligations"),
    _make_chunk("c8",
        "Employee shall be entitled to an annual base salary of Two Hundred Fifty Thousand "
        "Dollars payable in accordance with the Company's standard payroll practices and "
        "subject to applicable withholdings and deductions.",
        doc_type="contract", source_file="employment.txt",
        section="Article 2 — Compensation"),
]


# ════════════════════════════════════════════════════════════════════════════════
# 1. Ingestion & chunking tests
# ════════════════════════════════════════════════════════════════════════════════

class TestIngestion:
    """Test document parsing and chunk creation."""

    def test_chunk_has_required_fields(self):
        chunk = _make_chunk("test1", "This is a test chunk.")
        assert chunk.chunk_id == "test1"
        assert chunk.text == "This is a test chunk."
        assert chunk.source_file == "test.pdf"
        assert chunk.doc_type == "contract"

    def test_chunk_to_dict_roundtrip(self):
        chunk = _make_chunk("test2", "Roundtrip test chunk.", doc_type="case_law")
        d = chunk.to_dict()
        restored = Chunk.from_dict(d)
        assert restored.chunk_id == chunk.chunk_id
        assert restored.text == chunk.text
        assert restored.doc_type == chunk.doc_type

    def test_chunk_token_count_is_positive(self):
        chunk = _make_chunk("test3", "This chunk has several words in it.")
        assert chunk.token_count > 0

    def test_chunk_metadata_default(self):
        chunk = _make_chunk("test4", "Metadata test.")
        assert isinstance(chunk.metadata, dict)

    def test_token_counter_works(self):
        from src.ingestion import count_tokens
        text = "This is a test sentence with eight words."
        count = count_tokens(text)
        assert count > 0
        assert count < 50  # sanity check


# ════════════════════════════════════════════════════════════════════════════════
# 2. BM25 retrieval tests
# ════════════════════════════════════════════════════════════════════════════════

class TestBM25:
    """Test BM25 sparse retrieval."""

    def test_bm25_returns_results(self):
        idx = BM25Index(LEGAL_CHUNKS)
        results = idx.search("indemnification liability damages", top_k=3)
        assert len(results) > 0

    def test_bm25_top_result_is_relevant(self):
        idx = BM25Index(LEGAL_CHUNKS)
        results = idx.search("termination breach notice thirty days", top_k=3)
        top_chunk, _ = results[0]
        assert top_chunk.chunk_id == "c2", "Termination chunk should rank first"

    def test_bm25_scores_are_positive(self):
        idx = BM25Index(LEGAL_CHUNKS)
        results = idx.search("payment invoice fees", top_k=5)
        for _, score in results:
            assert score >= 0

    def test_bm25_respects_top_k(self):
        idx = BM25Index(LEGAL_CHUNKS)
        results = idx.search("agreement", top_k=3)
        assert len(results) <= 3

    def test_bm25_empty_query_doesnt_crash(self):
        idx = BM25Index(LEGAL_CHUNKS)
        results = idx.search("", top_k=3)
        assert isinstance(results, list)


# ════════════════════════════════════════════════════════════════════════════════
# 3. Reciprocal Rank Fusion tests
# ════════════════════════════════════════════════════════════════════════════════

class TestRRF:
    """Test Reciprocal Rank Fusion merging."""

    def test_rrf_merges_lists(self):
        list_a = [(LEGAL_CHUNKS[0], 0.9), (LEGAL_CHUNKS[1], 0.7)]
        list_b = [(LEGAL_CHUNKS[2], 0.95), (LEGAL_CHUNKS[0], 0.6)]
        fused = reciprocal_rank_fusion([list_a, list_b], k=60)
        assert len(fused) > 0
        fused_ids = {c.chunk_id for c, _ in fused}
        assert "c1" in fused_ids
        assert "c3" in fused_ids

    def test_rrf_scores_are_positive(self):
        list_a = [(LEGAL_CHUNKS[0], 0.9)]
        list_b = [(LEGAL_CHUNKS[0], 0.8)]
        fused = reciprocal_rank_fusion([list_a, list_b])
        _, score = fused[0]
        assert score > 0

    def test_rrf_boosts_shared_items(self):
        """Item appearing in both lists should score higher than items in one."""
        shared = LEGAL_CHUNKS[0]
        only_a = LEGAL_CHUNKS[1]
        only_b = LEGAL_CHUNKS[2]

        list_a = [(shared, 0.5), (only_a, 0.9)]
        list_b = [(shared, 0.5), (only_b, 0.9)]

        fused = reciprocal_rank_fusion([list_a, list_b])
        fused_dict = {c.chunk_id: score for c, score in fused}

        assert fused_dict["c1"] > fused_dict.get("c2", 0)
        assert fused_dict["c1"] > fused_dict.get("c3", 0)

    def test_rrf_empty_lists(self):
        fused = reciprocal_rank_fusion([[], []])
        assert fused == []


# ════════════════════════════════════════════════════════════════════════════════
# 4. Re-ranker tests (no model loading — structure tests only)
# ════════════════════════════════════════════════════════════════════════════════

class TestReranker:
    """Test re-ranker data structures and logic (no model needed)."""

    def test_reranker_imports(self):
        from src.reranker import CrossEncoderReranker, QueryExpander, RerankedResult
        assert CrossEncoderReranker is not None

    def test_query_expander_synonym_expansion(self):
        from src.reranker import QueryExpander
        expander = QueryExpander(use_llm=False)
        variants = expander.expand("What are the indemnification obligations?")
        assert len(variants) >= 2
        # Should include synonyms like "hold harmless"
        combined = " ".join(variants).lower()
        assert "indemnity" in combined or "hold harmless" in combined

    def test_query_expander_no_match(self):
        from src.reranker import QueryExpander
        expander = QueryExpander(use_llm=False)
        variants = expander.expand("Tell me about the weather today")
        assert len(variants) >= 1  # At least the original

    def test_reranked_result_structure(self):
        from src.reranker import RerankedResult
        result = RerankedResult(
            chunk=LEGAL_CHUNKS[0],
            reranker_score=0.85,
            rrf_score=0.03,
            dense_rank=1,
            sparse_rank=3,
            final_rank=1,
        )
        assert result.reranker_score == 0.85
        assert result.final_rank == 1


# ════════════════════════════════════════════════════════════════════════════════
# 5. Hallucination detection / claim extraction tests
# ════════════════════════════════════════════════════════════════════════════════

class TestHallucination:
    """Test claim extraction logic (no NLI model needed)."""

    def test_extract_claims_splits_sentences(self):
        from src.hallucination import extract_claims
        answer = (
            "The agreement requires 30 days notice for termination [Chunk 1]. "
            "Payment is due within 30 days of invoice [Chunk 2]. "
            "Confidentiality obligations survive termination [Chunk 3]."
        )
        claims = extract_claims(answer)
        assert len(claims) >= 2

    def test_extract_claims_removes_citations(self):
        from src.hallucination import extract_claims
        answer = "Indemnification is required [Chunk 1]. Liability is capped at fees paid [Chunk 2]."
        claims = extract_claims(answer)
        for claim in claims:
            assert "[Chunk" not in claim

    def test_extract_claims_handles_bullets(self):
        from src.hallucination import extract_claims
        answer = (
            "Key provisions include:\n"
            "- The party must provide written notice\n"
            "- Payment terms are net 30 days\n"
            "- Governing law is the State of Delaware"
        )
        claims = extract_claims(answer)
        assert len(claims) >= 1

    def test_extract_claims_filters_short_fragments(self):
        from src.hallucination import extract_claims
        answer = "Yes. The indemnification clause requires that each party hold harmless the other."
        claims = extract_claims(answer)
        # "Yes." alone should be filtered out as too short
        for claim in claims:
            assert len(claim) > 10

    def test_extract_claims_empty_input(self):
        from src.hallucination import extract_claims
        claims = extract_claims("")
        assert isinstance(claims, list)


# ════════════════════════════════════════════════════════════════════════════════
# 6. Evaluation metrics tests
# ════════════════════════════════════════════════════════════════════════════════

class TestEvaluationMetrics:
    """Test retrieval evaluation metric calculations."""

    def test_reciprocal_rank_found(self):
        from src.evaluation import _reciprocal_rank
        rr = _reciprocal_rank(["c1", "c2", "c3"], "c2")
        assert rr == 0.5  # Found at position 2 → 1/2

    def test_reciprocal_rank_not_found(self):
        from src.evaluation import _reciprocal_rank
        rr = _reciprocal_rank(["c1", "c2", "c3"], "c99")
        assert rr == 0.0

    def test_reciprocal_rank_first_position(self):
        from src.evaluation import _reciprocal_rank
        rr = _reciprocal_rank(["c1", "c2", "c3"], "c1")
        assert rr == 1.0

    def test_ndcg_at_k(self):
        from src.evaluation import _ndcg_at_k
        ndcg = _ndcg_at_k(["c1", "c2", "c3"], "c1", k=5)
        assert ndcg > 0
        # First position should give highest NDCG
        ndcg_first = _ndcg_at_k(["c1", "c2", "c3"], "c1", k=5)
        ndcg_third = _ndcg_at_k(["c1", "c2", "c3"], "c3", k=5)
        assert ndcg_first > ndcg_third

    def test_precision_at_k(self):
        from src.evaluation import _precision_at_k
        p = _precision_at_k(["c1", "c2", "c3"], "c1", k=5)
        assert p == 1.0 / 5  # 1 relevant in top 5
        p_miss = _precision_at_k(["c1", "c2", "c3"], "c99", k=5)
        assert p_miss == 0.0


# ════════════════════════════════════════════════════════════════════════════════
# 7. API endpoint tests
# ════════════════════════════════════════════════════════════════════════════════

class TestAPI:
    """Test FastAPI endpoints (requires server to NOT be running — uses TestClient)."""

    @pytest.fixture
    def client(self):
        """Create a test client. Skips if server dependencies missing."""
        try:
            from fastapi.testclient import TestClient
            from server import app
            return TestClient(app)
        except Exception:
            pytest.skip("FastAPI test client not available")

    def test_health_endpoint(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["status"] == "ok"

    def test_documents_list(self, client):
        resp = client.get("/api/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert "documents" in data
        assert "total" in data

    def test_history_endpoint(self, client):
        resp = client.get("/api/history")
        assert resp.status_code == 200
        data = resp.json()
        assert "history" in data
        assert isinstance(data["history"], list)

    def test_upload_rejects_invalid_extension(self, client):
        """Uploading a .exe file should be rejected."""
        from io import BytesIO
        resp = client.post(
            "/api/documents/upload",
            files={"file": ("malware.exe", BytesIO(b"fake"), "application/octet-stream")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"
        assert "Unsupported" in data["message"]

    def test_upload_accepts_txt(self, client):
        """Uploading a .txt file should succeed."""
        from io import BytesIO
        content = b"This is a test legal document about indemnification."
        resp = client.post(
            "/api/documents/upload",
            files={"file": ("test_doc.txt", BytesIO(content), "text/plain")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


# ════════════════════════════════════════════════════════════════════════════════
# 8. Categorization tests
# ════════════════════════════════════════════════════════════════════════════════

class TestCategorization:
    """Test legal topic categorization."""

    def test_categorize_indemnification(self):
        from src.evaluation import _categorize_chunk
        cat = _categorize_chunk("The indemnification clause requires hold harmless provisions.")
        assert cat == "indemnification"

    def test_categorize_termination(self):
        from src.evaluation import _categorize_chunk
        cat = _categorize_chunk("Either party may terminate upon thirty days notice.")
        assert cat == "termination"

    def test_categorize_confidentiality(self):
        from src.evaluation import _categorize_chunk
        cat = _categorize_chunk("All proprietary and confidential information must be protected.")
        assert cat == "confidentiality"

    def test_categorize_general(self):
        from src.evaluation import _categorize_chunk
        cat = _categorize_chunk("The sky is blue and the grass is green.")
        assert cat == "general"