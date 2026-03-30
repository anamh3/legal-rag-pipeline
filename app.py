#!/usr/bin/env python3
"""
app.py — Gradio web interface for the Legal RAG pipeline.

Upgrades over original:
  - Re-ranker integration for better answer quality
  - Redesigned UI with professional styling
  - Color-coded claim verification cards
  - Source chunks with relevance indicators
  - Query expansion indicator
  - Re-ranker toggle for A/B comparison
"""
import sys
import time
import gradio as gr

# ── Load pipeline ─────────────────────────────────────────────────────────────

def load_pipeline():
    try:
        from src.pipeline import RAGPipeline
        return RAGPipeline.load()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nRun ingestion first:")
        print("  python ingest.py --input data/raw/ --reset\n")
        sys.exit(1)


def load_reranker():
    """Try to load the cross-encoder re-ranker. Returns None if unavailable."""
    try:
        from src.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()
        # Warm up the model on first call (lazy loading)
        return reranker
    except ImportError:
        print("[INFO] Re-ranker not available (src/reranker.py missing). Using base retrieval.")
        return None
    except Exception as e:
        print(f"[WARN] Re-ranker failed to load: {e}")
        return None


pipeline = load_pipeline()
reranker = load_reranker()


# ── Custom CSS ────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* ── Global ─────────────────────────────────────────── */
.gradio-container {
    max-width: 1100px !important;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important;
}

/* ── Header ─────────────────────────────────────────── */
.app-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 20px;
    border: 1px solid #334155;
}
.app-header h1 {
    color: #f8fafc;
    font-size: 24px;
    font-weight: 700;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.app-header p {
    color: #94a3b8;
    font-size: 14px;
    margin: 0;
    line-height: 1.5;
}
.app-header .badge-row {
    display: flex;
    gap: 8px;
    margin-top: 12px;
    flex-wrap: wrap;
}
.app-header .badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.badge-retrieval { background: #1e3a5f; color: #60a5fa; }
.badge-nli       { background: #3b1f2b; color: #f472b6; }
.badge-reranker  { background: #1a3a2a; color: #4ade80; }

/* ── Faith bar ──────────────────────────────────────── */
.faith-container {
    background: #0f172a;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 8px 0 12px;
    border: 1px solid #1e293b;
}
.faith-header {
    display: flex;
    align-items: baseline;
    gap: 10px;
    margin-bottom: 8px;
}
.faith-label {
    font-size: 13px;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.faith-score {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -1px;
}
.faith-status {
    font-size: 12px;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 4px;
}
.faith-bar-bg {
    background: #1e293b;
    border-radius: 6px;
    height: 8px;
    width: 100%;
    overflow: hidden;
}
.faith-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.5s ease;
}
.faith-stats {
    display: flex;
    gap: 16px;
    margin-top: 10px;
    font-size: 12px;
    color: #64748b;
}
.faith-stats span {
    display: inline-flex;
    align-items: center;
    gap: 4px;
}
.stat-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
}

/* ── Source cards ────────────────────────────────────── */
.source-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
    transition: border-color 0.2s;
}
.source-card:hover {
    border-color: #334155;
}
.source-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.source-tag {
    font-size: 11px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 0.3px;
}
.tag-contract  { background: #1e3a5f; color: #60a5fa; }
.tag-case_law  { background: #3b2f1e; color: #fbbf24; }
.tag-statute   { background: #1a3a2a; color: #4ade80; }
.tag-unknown   { background: #1e293b; color: #94a3b8; }
.source-score {
    font-size: 11px;
    color: #64748b;
    font-family: 'Cascadia Code', 'Fira Code', monospace;
}
.source-meta {
    font-size: 11px;
    color: #475569;
    margin-bottom: 6px;
}
.source-text {
    font-size: 13px;
    color: #cbd5e1;
    line-height: 1.6;
    white-space: pre-wrap;
    background: #1e293b;
    border-radius: 6px;
    padding: 10px 12px;
    max-height: 200px;
    overflow-y: auto;
}

/* ── Claim cards ────────────────────────────────────── */
.claim-card {
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 8px;
    border-left: 4px solid;
    font-size: 13px;
    line-height: 1.5;
}
.claim-grounded {
    background: #0a2618;
    border-color: #22c55e;
    color: #bbf7d0;
}
.claim-ungrounded {
    background: #1a1a0a;
    border-color: #eab308;
    color: #fef08a;
}
.claim-contradicted {
    background: #2a0a0a;
    border-color: #ef4444;
    color: #fecaca;
}
.claim-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.claim-score {
    font-family: 'Cascadia Code', 'Fira Code', monospace;
    font-size: 11px;
}
"""


# ── Helper formatters ─────────────────────────────────────────────────────────

def format_sources(result, used_reranker: bool = False) -> str:
    """Format retrieved chunks as styled source cards."""
    lines = []

    for i, r in enumerate(result.retrieval_results, 1):
        c = r.chunk

        # Tag color by doc type
        tag_class = f"tag-{c.doc_type}" if c.doc_type in ("contract", "case_law", "statute") else "tag-unknown"

        # Score display
        score_label = f"RRF: {r.rrf_score:.4f}"
        if used_reranker and hasattr(r, 'reranker_score') and r.reranker_score:
            score_label = f"Re-rank: {r.reranker_score:.3f} | {score_label}"

        # Truncate text for display
        display_text = c.text[:500] + ("..." if len(c.text) > 500 else "")

        lines.append(f"""
<div class="source-card">
  <div class="source-header">
    <div>
      <span class="source-tag {tag_class}">{c.doc_type or 'document'}</span>
      <span style="color:#e2e8f0; font-weight:600; font-size:14px; margin-left:8px;">Chunk {i}</span>
    </div>
    <span class="source-score">{score_label}</span>
  </div>
  <div class="source-meta">
    📄 {c.source_file} &nbsp;|&nbsp; 📑 {c.section or 'N/A'} &nbsp;|&nbsp; 📃 Page {c.page_num or 'N/A'} &nbsp;|&nbsp; 🔤 {c.token_count} tokens
  </div>
  <div class="source-text">{display_text}</div>
</div>
""")

    return "\n".join(lines) if lines else "<p style='color:#64748b;'>No sources retrieved.</p>"


def format_claim_cards(result) -> str:
    """Format per-claim verification as color-coded cards."""
    claims = result.faithfulness.claims
    if not claims:
        return "<p style='color:#64748b;'>No claims extracted from the answer.</p>"

    lines = []
    for i, c in enumerate(claims, 1):
        if c.is_grounded:
            card_class = "claim-grounded"
            icon = "✓"
            status = "GROUNDED"
        elif hasattr(c.best_label, 'value') and c.best_label.value == "contradiction":
            card_class = "claim-contradicted"
            icon = "✗"
            status = "CONTRADICTED"
        else:
            card_class = "claim-ungrounded"
            icon = "?"
            status = "UNGROUNDED"

        score_display = f"entailment: {c.best_entailment_prob:.2f}"

        lines.append(f"""
<div class="claim-card {card_class}">
  <div class="claim-header">
    <span>{icon} Claim {i} — {status}</span>
    <span class="claim-score">{score_display}</span>
  </div>
  <div>{c.claim}</div>
</div>
""")

    return "\n".join(lines)


def format_faithfulness_bar(score: float, is_flagged: bool, result=None, latency_ms: float = 0) -> str:
    """Render the faithfulness indicator with stats."""
    pct = int(score * 100)

    if score >= 0.85:
        color, label, bg = "#22c55e", "Highly Grounded", "#0a2618"
    elif score >= 0.70:
        color, label, bg = "#84cc16", "Mostly Grounded", "#1a2a0a"
    elif score >= 0.50:
        color, label, bg = "#eab308", "Partially Grounded", "#2a2a0a"
    else:
        color, label, bg = "#ef4444", "High Hallucination Risk", "#2a0a0a"

    flag_html = ""
    if is_flagged:
        flag_html = """
        <span style="font-size:11px; color:#ef4444; font-weight:600;
                      background:#2a0a0a; padding:2px 8px; border-radius:4px;
                      margin-left:8px;">
            ⚠ REVIEW CAREFULLY
        </span>"""

    # Stats row
    stats_html = ""
    if result:
        f = result.faithfulness
        stats_html = f"""
        <div class="faith-stats">
            <span><span class="stat-dot" style="background:#22c55e;"></span> {f.n_grounded} grounded</span>
            <span><span class="stat-dot" style="background:#eab308;"></span> {f.n_ungrounded} ungrounded</span>
            <span><span class="stat-dot" style="background:#ef4444;"></span> {f.n_contradicted} contradicted</span>
            <span>⏱ {latency_ms:.0f}ms</span>
            <span>📊 {len(f.claims)} claims extracted</span>
        </div>"""

    return f"""
<div class="faith-container">
  <div class="faith-header">
    <span class="faith-label">Faithfulness</span>
    <span class="faith-score" style="color:{color};">{pct}%</span>
    <span class="faith-status" style="background:{bg}; color:{color};">{label}</span>
    {flag_html}
  </div>
  <div class="faith-bar-bg">
    <div class="faith-bar-fill" style="background:{color}; width:{pct}%;"></div>
  </div>
  {stats_html}
</div>
"""


def format_answer(result, used_reranker: bool = False) -> str:
    """Format the answer with model info."""
    model_name = result.generation.model
    n_chunks = len(result.retrieval_results)
    retrieval_method = "Hybrid + Re-ranker" if used_reranker else "Hybrid (RRF)"

    header = f"""
<div style="display:flex; gap:8px; margin-bottom:12px; flex-wrap:wrap;">
    <span style="font-size:11px; padding:3px 8px; border-radius:4px;
                 background:#1e293b; color:#94a3b8;">
        🤖 {model_name}
    </span>
    <span style="font-size:11px; padding:3px 8px; border-radius:4px;
                 background:#1e293b; color:#94a3b8;">
        📚 {n_chunks} chunks
    </span>
    <span style="font-size:11px; padding:3px 8px; border-radius:4px;
                 background:#1e3a5f; color:#60a5fa;">
        🔍 {retrieval_method}
    </span>
</div>
"""
    return header + "\n\n" + result.answer


# ── Core query function ───────────────────────────────────────────────────────

def run_query(
    question: str,
    doc_type_filter: str,
    top_k: int,
    use_reranker: bool,
):
    """Called by Gradio on submit."""
    if not question.strip():
        return (
            "_Please enter a question._",
            "<p style='color:#64748b;'>No query.</p>",
            "<p style='color:#64748b;'>No query.</p>",
            "<p style='color:#64748b;'>No query.</p>",
        )

    filter_val = None if doc_type_filter == "All" else doc_type_filter.lower().replace(" ", "_")

    start_time = time.time()

    try:
        # If re-ranker is enabled and available, use enhanced retrieval
        if use_reranker and reranker is not None:
            result = _query_with_reranker(question, int(top_k), filter_val)
            used_reranker = True
        else:
            result = pipeline.query(
                question        = question,
                top_k           = int(top_k),
                doc_type_filter = filter_val,
                temperature     = 0.0,
            )
            used_reranker = False

    except Exception as e:
        error_html = f"""
        <div style="background:#2a0a0a; border:1px solid #ef4444; border-radius:8px;
                    padding:16px; color:#fecaca;">
            <strong>Error:</strong> {str(e)}
        </div>"""
        return error_html, "", "<p>Error occurred.</p>", ""

    latency_ms = (time.time() - start_time) * 1000

    answer_html   = format_answer(result, used_reranker)
    sources_html  = format_sources(result, used_reranker)
    claims_html   = format_claim_cards(result)
    faith_html    = format_faithfulness_bar(
        result.faithfulness_score, result.is_flagged, result, latency_ms
    )

    return answer_html, sources_html, faith_html, claims_html


def _query_with_reranker(question: str, top_k: int, doc_type_filter=None):
    """
    Enhanced query pipeline: Hybrid retrieval → Re-ranking → Generation → NLI check.

    Pulls 20 candidates from hybrid retrieval, re-ranks with cross-encoder,
    then sends the top-k to the LLM for generation.
    """
    # Step 1: Get more candidates from hybrid retrieval
    from src.retrieval import HybridRetriever
    retrieval_results = pipeline.retriever.retrieve(
        question,
        top_k_final=20,  # more candidates for re-ranking
        doc_type_filter=doc_type_filter,
    )

    # Step 2: Re-rank with cross-encoder
    reranked = reranker.rerank(question, retrieval_results, top_k=top_k)

    # Step 3: Build chunks for generation
    chunks = [r.chunk for r in reranked]

    # Step 4: Generate answer using re-ranked chunks
    from src.generation import generate_answer
    gen_result = generate_answer(question, chunks, temperature=0.0)

    # Step 5: Check faithfulness
    from src.hallucination import compute_faithfulness
    faith_result = compute_faithfulness(question, gen_result.answer, chunks)

    # Step 6: Package as PipelineResult
    from src.pipeline import PipelineResult

    # Convert RerankedResults back to RetrievalResult-like objects for display
    from src.retrieval import RetrievalResult
    display_results = []
    for r in reranked:
        display_results.append(RetrievalResult(
            chunk       = r.chunk,
            dense_score = None,
            sparse_score= None,
            rrf_score   = r.rrf_score,
            dense_rank  = r.dense_rank,
            sparse_rank = r.sparse_rank,
        ))

    return PipelineResult(
        query             = question,
        retrieval_results = display_results,
        generation        = gen_result,
        faithfulness      = faith_result,
    )


# ── Example queries ───────────────────────────────────────────────────────────

EXAMPLE_QUERIES = [
    ["What are the indemnification obligations of the parties?", "All", 5, True],
    ["What is the governing law and jurisdiction for disputes?", "Contract", 5, True],
    ["What are the termination conditions and notice requirements?", "All", 5, True],
    ["What confidentiality obligations apply after termination?", "Contract", 5, True],
    ["What damages or remedies are available for breach?", "All", 5, True],
]


# ── Gradio UI ─────────────────────────────────────────────────────────────────

HEADER_HTML = """
<div class="app-header">
    <h1>⚖️ Legal RAG Pipeline</h1>
    <p>Ask questions about ingested legal documents. Every answer is grounded in retrieved context,
       cited with chunk references, and verified via NLI-based faithfulness scoring.</p>
    <div class="badge-row">
        <span class="badge badge-retrieval">🔍 Hybrid Retrieval + RRF</span>
        <span class="badge badge-reranker">🎯 Cross-Encoder Re-ranking</span>
        <span class="badge badge-nli">🧠 NLI Hallucination Detection</span>
    </div>
</div>
"""


with gr.Blocks(css=CUSTOM_CSS) as demo:

    gr.HTML(HEADER_HTML)

    # ── Input row ─────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=3):
            question_input = gr.Textbox(
                label="Question",
                placeholder="e.g., What are the indemnification obligations of the parties?",
                lines=2,
                show_label=True,
            )
        with gr.Column(scale=1):
            doc_type_filter = gr.Dropdown(
                label="Document Type",
                choices=["All", "Contract", "Case law", "Statute"],
                value="All",
            )
            with gr.Row():
                top_k_slider = gr.Slider(
                    label="Chunks",
                    minimum=2,
                    maximum=10,
                    value=5,
                    step=1,
                )
                use_reranker_toggle = gr.Checkbox(
                    label="Re-ranker",
                    value=reranker is not None,  # auto-enable if available
                    interactive=reranker is not None,
                )

    submit_btn = gr.Button(
        "🔍 Search & Analyze",
        variant="primary",
        size="lg",
    )

    # ── Faithfulness bar ──────────────────────────────────────────────────
    faithfulness_display = gr.HTML()

    # ── Tabbed results ────────────────────────────────────────────────────
    with gr.Tabs():
        with gr.Tab("💬 Answer"):
            answer_display = gr.Markdown()

        with gr.Tab("📚 Sources"):
            sources_display = gr.HTML()

        with gr.Tab("🔬 Claim Verification"):
            claims_display = gr.HTML()

    # ── Examples ──────────────────────────────────────────────────────────
    gr.Examples(
        examples=EXAMPLE_QUERIES,
        inputs=[question_input, doc_type_filter, top_k_slider, use_reranker_toggle],
        label="Example Queries",
    )

    # ── Wire up events ────────────────────────────────────────────────────
    submit_btn.click(
        fn=run_query,
        inputs=[question_input, doc_type_filter, top_k_slider, use_reranker_toggle],
        outputs=[answer_display, sources_display, faithfulness_display, claims_display],
    )
    question_input.submit(
        fn=run_query,
        inputs=[question_input, doc_type_filter, top_k_slider, use_reranker_toggle],
        outputs=[answer_display, sources_display, faithfulness_display, claims_display],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )