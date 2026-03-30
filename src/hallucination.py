"""
hallucination.py — NLI-based faithfulness scoring and hallucination detection.

Pipeline:
  1. Extract atomic claims from the generated answer
  2. For each claim, run NLI against every retrieved chunk
  3. A claim is "grounded" if ANY chunk entails it (prob > threshold)
  4. Faithfulness = grounded_claims / total_claims

Model: cross-encoder/nli-deberta-v3-large
  - DeBERTa-v3 fine-tuned on MNLI+SNLI+FEVER
  - Output labels: [contradiction, entailment, neutral] (index order varies by model)
  - Best open-source NLI model as of 2024 for faithfulness evaluation
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np

from src.config import NLI_MODEL, NLI_BATCH_SIZE, NLI_THRESHOLD, FAITHFULNESS_WARN_THRESHOLD
from src.ingestion import Chunk


# ── NLI label mapping ─────────────────────────────────────────────────────────

class NLILabel(str, Enum):
    CONTRADICTION = "contradiction"
    NEUTRAL       = "neutral"
    ENTAILMENT    = "entailment"


# DeBERTa NLI label order: [contradiction, entailment, neutral]
# (confirmed from model card — don't assume alphabetical)
_DEBERTA_LABEL_ORDER = [NLILabel.CONTRADICTION, NLILabel.ENTAILMENT, NLILabel.NEUTRAL]


# ── NLI model (lazy-loaded singleton) ─────────────────────────────────────────

_nli_model = None

def _get_nli_model():
    global _nli_model
    if _nli_model is None:
        print(f"Loading NLI model: {NLI_MODEL} ...")
        from sentence_transformers import CrossEncoder
        _nli_model = CrossEncoder(NLI_MODEL)
        print("NLI model loaded.")
    return _nli_model


# ── Claim extraction ──────────────────────────────────────────────────────────

def extract_claims(answer: str) -> List[str]:
    """
    Split an answer into atomic claims (one per sentence, roughly).

    Handles:
    - Sentence boundaries (period, !, ?)
    - Parenthetical citations [Chunk N] removed
    - Bullet points / numbered lists → individual claims
    - Filters out very short fragments (<20 chars)
    """
    # Remove citation markers for cleaner claim text
    text = re.sub(r"\[Chunk\s*\d+\]", "", answer, flags=re.IGNORECASE)

    # Split on sentence boundaries
    # Regex: split on . ! ? followed by whitespace and uppercase (or end of string)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\'(])|(?<=[.!?])\s*$", text)

    # Also handle bullet/numbered lists
    bullets = []
    for sent in sentences:
        # If a sentence contains newlines (bullet list), split further
        sub = re.split(r"\n+[-•*\d]+[.)]\s*", sent)
        bullets.extend(sub)

    claims = []
    for claim in bullets:
        claim = claim.strip()
        # Remove trailing punctuation oddities
        claim = re.sub(r"^[-•*\d.)\s]+", "", claim).strip()
        if len(claim) >= 20:  # skip fragments
            claims.append(claim)

    return claims if claims else [answer.strip()]


# ── Per-claim result ──────────────────────────────────────────────────────────

@dataclass
class ClaimVerification:
    claim:             str
    is_grounded:       bool
    best_label:        NLILabel
    best_entailment_prob: float
    best_chunk_id:     Optional[str]    # chunk_id of the most entailing chunk
    all_scores:        List[dict] = field(default_factory=list)  # [{chunk_id, probs}]

    @property
    def status_emoji(self) -> str:
        if self.is_grounded:
            return "✓"
        elif self.best_label == NLILabel.CONTRADICTION:
            return "✗"
        return "?"


@dataclass
class FaithfulnessResult:
    query:              str
    answer:             str
    chunks:             List[Chunk]
    claims:             List[ClaimVerification]
    faithfulness_score: float       # 0.0 – 1.0
    is_flagged:         bool        # True if below FAITHFULNESS_WARN_THRESHOLD
    n_grounded:         int
    n_ungrounded:       int
    n_contradicted:     int

    @property
    def summary(self) -> str:
        flag = " [HALLUCINATION RISK]" if self.is_flagged else ""
        return (
            f"Faithfulness: {self.faithfulness_score:.2f}{flag} | "
            f"Grounded: {self.n_grounded}/{len(self.claims)} claims"
        )


# ── NLI scoring ───────────────────────────────────────────────────────────────

def _run_nli_batch(
    premise_hypothesis_pairs: List[tuple[str, str]],
) -> np.ndarray:
    """
    Run NLI model on a batch of (premise, hypothesis) pairs.
    Returns softmax probabilities of shape (n, 3).
    Labels: [contradiction, entailment, neutral]
    """
    model = _get_nli_model()
    scores = model.predict(
        premise_hypothesis_pairs,
        apply_softmax=True,
        batch_size=NLI_BATCH_SIZE,
        show_progress_bar=False,
    )
    return np.array(scores)


def verify_claim(
    claim: str,
    chunks: List[Chunk],
    threshold: float = NLI_THRESHOLD,
) -> ClaimVerification:
    """
    Check a single claim against all retrieved chunks.
    A claim is grounded if ANY chunk entails it with prob > threshold.
    """
    if not chunks:
        return ClaimVerification(
            claim=claim,
            is_grounded=False,
            best_label=NLILabel.NEUTRAL,
            best_entailment_prob=0.0,
            best_chunk_id=None,
        )

    # Build (premise=chunk_text, hypothesis=claim) pairs
    pairs = [(chunk.text, claim) for chunk in chunks]
    probs = _run_nli_batch(pairs)  # shape: (n_chunks, 3)

    # DeBERTa label order: [contradiction=0, entailment=1, neutral=2]
    contradiction_probs = probs[:, 0]
    entailment_probs    = probs[:, 1]
    neutral_probs       = probs[:, 2]

    best_idx = int(np.argmax(entailment_probs))
    best_entailment_prob = float(entailment_probs[best_idx])

    # Determine the dominant label for the best chunk
    best_probs = probs[best_idx]
    dominant_label_idx = int(np.argmax(best_probs))
    best_label = _DEBERTA_LABEL_ORDER[dominant_label_idx]

    is_grounded = best_entailment_prob >= threshold

    all_scores = [
        {
            "chunk_id": chunks[i].chunk_id,
            "entailment":    float(entailment_probs[i]),
            "contradiction": float(contradiction_probs[i]),
            "neutral":       float(neutral_probs[i]),
        }
        for i in range(len(chunks))
    ]

    return ClaimVerification(
        claim                = claim,
        is_grounded          = is_grounded,
        best_label           = best_label,
        best_entailment_prob = best_entailment_prob,
        best_chunk_id        = chunks[best_idx].chunk_id,
        all_scores           = all_scores,
    )


# ── Full faithfulness evaluation ──────────────────────────────────────────────

def compute_faithfulness(
    query: str,
    answer: str,
    chunks: List[Chunk],
    threshold: float = NLI_THRESHOLD,
    warn_threshold: float = FAITHFULNESS_WARN_THRESHOLD,
) -> FaithfulnessResult:
    """
    Full faithfulness evaluation for a generated answer.

    Steps:
    1. Extract atomic claims from the answer
    2. Verify each claim against retrieved chunks via NLI
    3. Compute faithfulness score

    Args:
        query:         Original user query
        answer:        LLM-generated answer
        chunks:        Retrieved context chunks
        threshold:     Entailment prob threshold to consider a claim grounded
        warn_threshold: Faithfulness score below this triggers a hallucination flag

    Returns:
        FaithfulnessResult with per-claim breakdown and aggregate score
    """
    claims_text = extract_claims(answer)

    if not claims_text:
        return FaithfulnessResult(
            query=query, answer=answer, chunks=chunks, claims=[],
            faithfulness_score=0.0, is_flagged=True,
            n_grounded=0, n_ungrounded=0, n_contradicted=0,
        )

    # Verify each claim
    verified_claims: List[ClaimVerification] = []
    for claim in claims_text:
        result = verify_claim(claim, chunks, threshold)
        verified_claims.append(result)

    # Aggregate
    n_grounded     = sum(1 for c in verified_claims if c.is_grounded)
    n_contradicted = sum(
        1 for c in verified_claims
        if not c.is_grounded and c.best_label == NLILabel.CONTRADICTION
    )
    n_ungrounded   = len(verified_claims) - n_grounded

    faithfulness_score = n_grounded / len(verified_claims) if verified_claims else 0.0
    is_flagged = faithfulness_score < warn_threshold

    return FaithfulnessResult(
        query              = query,
        answer             = answer,
        chunks             = chunks,
        claims             = verified_claims,
        faithfulness_score = faithfulness_score,
        is_flagged         = is_flagged,
        n_grounded         = n_grounded,
        n_ungrounded       = n_ungrounded,
        n_contradicted     = n_contradicted,
    )


def faithfulness_score_only(
    answer: str,
    chunks: List[Chunk],
) -> float:
    """Lightweight version — returns just the float score."""
    result = compute_faithfulness("", answer, chunks)
    return result.faithfulness_score
