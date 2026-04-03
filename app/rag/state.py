"""LangGraph state schema for the RAG pipeline."""

from typing import Any, Required, TypedDict


class StepStatus(TypedDict):
    """Human-readable progress label written by every node.
    Consumed by the SSE route to push live stage events to the client.
    """
    title: str        # short label  e.g. "Keywords Extracted"
    description: str  # detail line  e.g. "Identified 5 key terms: neural network, ..."


class RAGState(TypedDict, total=False):
    """Full pipeline state.

    ``question`` and ``max_iterations`` are marked ``Required`` — they MUST be
    provided when invoking the graph; all other keys are populated by nodes.
    """

    # ── Required inputs (must be present at graph invocation) ─────────────────
    question: Required[str]
    max_iterations: Required[int]

    # ── Stage progress (written by EVERY node, read by SSE route) ────────────
    step: StepStatus   # {"title": "...", "description": "..."}

    # ── Keyword & query rewriting ─────────────────────────────────────────────
    keywords: list[str]
    optimized_query: str
    search_queries_tried: list[str]  # all queries issued so far

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k: int                              # LLM-estimated chunks per pass (clamped 3-20)
    retrieved_chunks: list[dict[str, Any]]  # chunks for CURRENT iteration
    all_chunk_ids: list[str]               # dedup across all iterations

    # ── Generation ────────────────────────────────────────────────────────────
    answer: str
    references: list[str]  # unique document/source names cited

    # ── Sufficiency check (post-retrieve LLM gate) ───────────────────────────
    is_sufficient: bool     # True = context is enough → proceed to generate
    sufficiency_reason: str # LLM explanation of what is/isn't covered

    # ── Next-retrieval hint set by refine_query_node ──────────────────────────
    # "keyword"  → exact-match $contains search
    # "semantic" → dense-vector similarity search
    # "hybrid"   → both (default)
    search_mode: str

    # ── Evaluation (terminal — log + store only) ──────────────────────────────
    ragas_scores: dict[str, float]  # metric_name -> score

    # ── Loop control ──────────────────────────────────────────────────────────
    iteration: int  # 0-based, incremented by refine_query_node
