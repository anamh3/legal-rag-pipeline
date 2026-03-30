"""
ingestion.py — Document parsing, cleaning, and clause-level chunking.

Supports PDF (via pdfplumber) and HTML (via BeautifulSoup).
Chunks at clause/section boundaries first, falls back to token-count splitting.
Each chunk gets rich metadata for filtering and citation generation.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import pdfplumber
import tiktoken
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, PROCESSED_DIR


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """A single text chunk with metadata."""
    chunk_id:    str
    text:        str
    source_file: str
    section:     str          # e.g. "Section 4.2 — Indemnification"
    page_num:    Optional[int]
    doc_type:    str          # "contract" | "case_law" | "statute" | "other"
    char_start:  int          # character offset in original doc
    token_count: int
    metadata:    dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)


# ── Tokenizer (shared, loaded once) ──────────────────────────────────────────

_TOKENIZER = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))

def split_by_tokens(text: str, max_tokens: int, overlap: int) -> List[str]:
    """Fallback splitter: sliding window over tokens."""
    tokens = _TOKENIZER.encode(text)
    chunks, start = [], 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(_TOKENIZER.decode(tokens[start:end]))
        start += max_tokens - overlap
    return chunks


# ── Clause boundary detection ─────────────────────────────────────────────────

# Legal section header patterns — ordered most-specific to most-general
_CLAUSE_PATTERNS = [
    r"^\s*(ARTICLE|Article)\s+[IVXLCDM\d]+[\.\s]",
    r"^\s*\d+\.\d+(?:\.\d+)?\s+[A-Z]",          # 4.2.1 Title
    r"^\s*\d+\.\s+[A-Z][A-Za-z\s]{3,}",          # 4. Title
    r"^\s*(Section|SECTION)\s+\d+",
    r"^\s*(WHEREAS|NOW, THEREFORE|IN WITNESS)",
    r"^\s*[A-Z]{3,}(?:\s+[A-Z]{2,}){0,3}\s*\n", # ALL CAPS HEADING
]
_CLAUSE_RE = re.compile("|".join(_CLAUSE_PATTERNS), re.MULTILINE)


def _detect_doc_type(filename: str, text: str) -> str:
    fname = filename.lower()
    if any(k in fname for k in ("contract", "agreement", "nda", "mou", "lease")):
        return "contract"
    if any(k in fname for k in ("case", "opinion", "judgment", "v.", "vs")):
        return "case_law"
    if any(k in fname for k in ("statute", "act", "regulation", "code", "usc")):
        return "statute"
    # Heuristic: case law usually mentions "plaintiff" or "defendant"
    if re.search(r"\b(plaintiff|defendant|appellant|appellee)\b", text[:2000], re.I):
        return "case_law"
    if re.search(r"\b(party|parties|hereinafter|whereas)\b", text[:2000], re.I):
        return "contract"
    return "other"


def _split_into_clauses(text: str) -> List[tuple[str, str]]:
    """
    Split text at legal clause boundaries.
    Returns list of (section_header, clause_text) tuples.
    """
    matches = list(_CLAUSE_RE.finditer(text))
    if not matches:
        return [("", text)]

    clauses = []
    for i, m in enumerate(matches):
        header = m.group(0).strip()
        start  = m.start()
        end    = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        clauses.append((header, text[start:end]))
    # Text before first match
    if matches[0].start() > 0:
        clauses.insert(0, ("Preamble", text[:matches[0].start()]))
    return clauses


# ── PDF Parser ────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Remove common PDF artifacts while preserving legal structure."""
    # Normalize whitespace but keep paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    # Remove page numbers (standalone digits on a line)
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)
    # Remove repeated headers/footers (lines appearing 3+ times)
    lines = text.split("\n")
    line_counts: dict[str, int] = {}
    for line in lines:
        stripped = line.strip()
        if stripped:
            line_counts[stripped] = line_counts.get(stripped, 0) + 1
    repeated = {l for l, c in line_counts.items() if c >= 3 and len(l) < 120}
    lines = [l for l in lines if l.strip() not in repeated]
    text = "\n".join(lines)
    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_pdf(filepath: Path) -> List[tuple[str, int]]:
    """
    Parse PDF, return list of (page_text, page_num) tuples.
    Strips headers/footers and joins hyphenated words across lines.
    """
    pages = []
    with pdfplumber.open(filepath) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            # Join hyphenated line breaks
            text = re.sub(r"-\n(\w)", r"\1", text)
            pages.append((text, i))
    return pages


def parse_html(filepath: Path) -> List[tuple[str, int]]:
    """Parse HTML (e.g., SEC filings, court opinions from web)."""
    html = filepath.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "lxml")
    # Remove boilerplate tags
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    return [(text, None)]


# ── Main chunker ──────────────────────────────────────────────────────────────

def chunk_document(
    filepath: Path,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Chunk]:
    """
    Full pipeline: parse → clean → clause-split → token-split → Chunk objects.
    """
    suffix = filepath.suffix.lower()
    if suffix == ".pdf":
        pages = parse_pdf(filepath)
    elif suffix in (".html", ".htm"):
        pages = parse_html(filepath)
    elif suffix == ".txt":
        pages = [(filepath.read_text(encoding="utf-8", errors="replace"), None)]
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    # Combine all pages into single string (keep page boundary info)
    full_text = _clean_text("\n\n".join(t for t, _ in pages))
    doc_type  = _detect_doc_type(filepath.name, full_text)

    chunks: List[Chunk] = []
    char_offset = 0

    clauses = _split_into_clauses(full_text)
    for section_header, clause_text in clauses:
        clause_text = clause_text.strip()
        if not clause_text:
            continue

        # If clause fits in one chunk, keep it whole
        if count_tokens(clause_text) <= chunk_size:
            sub_texts = [clause_text]
        else:
            sub_texts = split_by_tokens(clause_text, chunk_size, overlap)

        for sub in sub_texts:
            sub = sub.strip()
            if not sub or count_tokens(sub) < 20:  # skip tiny fragments
                continue

            # Approximate page number from char offset
            cumulative = 0
            approx_page = 1
            for txt, pnum in pages:
                cumulative += len(txt)
                if char_offset <= cumulative:
                    approx_page = pnum or 1
                    break

            chunk_id = hashlib.md5(
                f"{filepath.name}:{char_offset}".encode()
            ).hexdigest()[:12]

            chunks.append(Chunk(
                chunk_id    = chunk_id,
                text        = sub,
                source_file = filepath.name,
                section     = section_header or "General",
                page_num    = approx_page,
                doc_type    = doc_type,
                char_start  = char_offset,
                token_count = count_tokens(sub),
                metadata    = {
                    "filepath": str(filepath),
                    "file_hash": hashlib.md5(filepath.read_bytes()).hexdigest(),
                }
            ))
            char_offset += len(sub)

    return chunks


# ── Batch ingestion ───────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".txt"}


def ingest_directory(
    input_dir: Path,
    output_dir: Path = PROCESSED_DIR,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    workers: int = 1,
) -> List[Chunk]:
    """
    Ingest all supported documents from input_dir.
    Saves chunks as JSON to output_dir.
    Returns all chunks.
    """
    files = [
        f for f in input_dir.rglob("*")
        if f.suffix.lower() in SUPPORTED_EXTENSIONS and f.is_file()
    ]
    if not files:
        raise FileNotFoundError(f"No supported documents found in {input_dir}")

    all_chunks: List[Chunk] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in tqdm(files, desc="Ingesting documents"):
        try:
            chunks = chunk_document(filepath, chunk_size, overlap)
            all_chunks.extend(chunks)

            # Save per-document chunk file
            out_path = output_dir / f"{filepath.stem}_chunks.json"
            with open(out_path, "w") as f:
                json.dump([c.to_dict() for c in chunks], f, indent=2)

            tqdm.write(f"  {filepath.name}: {len(chunks)} chunks")
        except Exception as e:
            tqdm.write(f"  [ERROR] {filepath.name}: {e}")

    # Save combined corpus
    corpus_path = output_dir / "corpus.json"
    with open(corpus_path, "w") as f:
        json.dump([c.to_dict() for c in all_chunks], f)

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Saved to: {output_dir}")
    return all_chunks


def load_corpus(processed_dir: Path = PROCESSED_DIR) -> List[Chunk]:
    """Load previously processed chunks from disk."""
    corpus_path = processed_dir / "corpus.json"
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"No corpus found at {corpus_path}. Run ingestion first."
        )
    with open(corpus_path) as f:
        return [Chunk.from_dict(d) for d in json.load(f)]
