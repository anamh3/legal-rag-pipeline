"""
generation.py — LLM generation with citation-aware prompting.

Prompts the LLM to ground every claim in the retrieved context and
explicitly cite source chunks. Supports OpenAI and Groq (Llama-3).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.config import (
    LLM_PROVIDER, OPENAI_API_KEY, GROQ_API_KEY,
    OPENAI_LLM_MODEL, GROQ_LLM_MODEL, get_llm_model
)
from src.ingestion import Chunk


# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise legal research assistant. Your task is to answer
questions based strictly on the provided legal document excerpts.

RULES YOU MUST FOLLOW:
1. Base every claim ONLY on the provided context chunks. Do not use outside knowledge.
2. After each claim or sentence, cite the source chunk using [Chunk N] notation.
3. If the context does not contain enough information to answer, say so explicitly.
4. Do not speculate, extrapolate, or add information not present in the chunks.
5. Use precise legal language. Do not paraphrase in ways that change meaning.
6. If chunks contradict each other, note the contradiction explicitly.

Your answer will be automatically verified for faithfulness. Unsupported claims
will be flagged as hallucinations."""


def _build_context_block(chunks: List[Chunk]) -> str:
    """Format retrieved chunks into a numbered context block."""
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[Chunk {i}] Source: {chunk.source_file} | "
            f"Section: {chunk.section} | "
            f"Type: {chunk.doc_type}\n"
            f"{chunk.text}\n"
        )
    return "\n---\n".join(lines)


def _build_user_message(query: str, chunks: List[Chunk]) -> str:
    context = _build_context_block(chunks)
    return (
        f"CONTEXT CHUNKS:\n\n{context}\n\n"
        f"{'='*60}\n\n"
        f"QUESTION: {query}\n\n"
        f"Answer based only on the context chunks above. "
        f"Cite each chunk you use with [Chunk N]."
    )


# ── LLM clients ───────────────────────────────────────────────────────────────

def _call_openai(messages: list, model: str, temperature: float) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=1500,
    )
    return response.choices[0].message.content.strip()


def _call_groq(messages: list, model: str, temperature: float) -> str:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=1500,
    )
    return response.choices[0].message.content.strip()


# ── Generation result ─────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    answer:        str
    query:         str
    chunks_used:   List[Chunk]
    model:         str
    cited_indices: List[int]   # Which chunk numbers appear in the answer

    def get_cited_chunks(self) -> List[Chunk]:
        """Return only the chunks actually cited in the answer."""
        return [
            self.chunks_used[i - 1]
            for i in self.cited_indices
            if 1 <= i <= len(self.chunks_used)
        ]


def _extract_cited_indices(answer: str, n_chunks: int) -> List[int]:
    """Parse [Chunk N] references from the answer text."""
    import re
    refs = re.findall(r"\[Chunk\s+(\d+)\]", answer, re.IGNORECASE)
    return sorted(set(
        int(r) for r in refs if 1 <= int(r) <= n_chunks
    ))


# ── Main generation function ──────────────────────────────────────────────────

def generate_answer(
    query: str,
    chunks: List[Chunk],
    temperature: float = 0.0,
    model: Optional[str] = None,
) -> GenerationResult:
    """
    Generate a grounded, cited answer from retrieved chunks.

    Args:
        query:       The user's question
        chunks:      Retrieved context chunks (ordered by relevance)
        temperature: LLM temperature (0.0 = deterministic, best for factual tasks)
        model:       Override the model from config

    Returns:
        GenerationResult with answer, cited chunks, and metadata
    """
    if not chunks:
        return GenerationResult(
            answer="No relevant documents were found to answer this question.",
            query=query,
            chunks_used=[],
            model=model or get_llm_model(),
            cited_indices=[],
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": _build_user_message(query, chunks)},
    ]

    resolved_model = model or get_llm_model()

    if LLM_PROVIDER == "openai":
        answer = _call_openai(messages, resolved_model, temperature)
    elif LLM_PROVIDER == "groq":
        answer = _call_groq(messages, resolved_model, temperature)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")

    cited_indices = _extract_cited_indices(answer, len(chunks))

    return GenerationResult(
        answer        = answer,
        query         = query,
        chunks_used   = chunks,
        model         = resolved_model,
        cited_indices = cited_indices,
    )


def generate_synthetic_question(chunk: Chunk, model: Optional[str] = None) -> str:
    """
    Generate a plausible question from a chunk.
    Used to build the evaluation test set.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a legal researcher. Given a legal document excerpt, "
                "generate ONE specific, answerable question whose answer is "
                "contained in the excerpt. Output only the question, nothing else."
            )
        },
        {
            "role": "user",
            "content": f"EXCERPT:\n{chunk.text[:800]}\n\nGenerate one question:"
        }
    ]

    resolved_model = model or get_llm_model()
    if LLM_PROVIDER == "openai":
        return _call_openai(messages, resolved_model, temperature=0.7)
    elif LLM_PROVIDER == "groq":
        return _call_groq(messages, resolved_model, temperature=0.7)
    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")
