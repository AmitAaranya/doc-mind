"""LangGraph RAG pipeline graph.

Flow
----
START ──► extract_keywords ──► rewrite_query ──► dispatch_retrieve ──► retrieve_semantic ──┐
                                                        ▲                                   ├──► merge_retrieve ──► check_sufficiency
                                                        │               ──► retrieve_bm25  ──┘              │
                                                   refine_query ◄──────────────────────────────────── sufficient? NO
                                                                                                            │
                                                                                                       sufficient? YES
                                                                                                            │
                                                                                                            ▼
                                                                                                        generate ──► evaluate ──► END

Parallel retrieve
-----------------
  retrieve_semantic  – dense-vector similarity search (ChromaDB)
  retrieve_bm25      – BM25 keyword search (SQLite corpus)
  Both fan out from rewrite_query / refine_query and fan in to merge_retrieve.

Loop gate
---------
  check_sufficiency  – LLM analyses retrieved chunks vs the question.
                       If the context is enough → proceed to generate.
                       If information is missing → refine_query → retrieve (loop).
                       Hard ceiling: max_iterations forces proceed after N loops.

Evaluation (terminal, no routing)
----------------------------------
  evaluate           – Runs RAGAS metrics (faithfulness, answer_relevancy,
                       context_precision), logs them, and appends a record to
                       ``rag_evaluations.jsonl`` in the project root.
"""

from __future__ import annotations

import io

from langgraph.graph import END, START, StateGraph
from PIL import Image

from app.core.logging import get_logger
from app.rag.evaluator import evaluate_node
from app.rag.nodes import (
    check_sufficiency_node,
    dispatch_retrieve_node,
    extract_keywords_node,
    generate_node,
    merge_retrieve_node,
    refine_query_node,
    retrieve_bm25_node,
    retrieve_semantic_node,
    rewrite_query_node,
)
from app.rag.state import RAGState

logger = get_logger(__name__)


# ── Conditional edge ──────────────────────────────────────────────────────────


def _decide_after_sufficiency(state: RAGState) -> str:
    """After check_sufficiency: loop back to retrieve or proceed to generate."""
    if state.get("is_sufficient", False):
        logger.info("Context sufficient – proceeding to generate.")
        return "generate"
    logger.info("Context insufficient – refining query (iter=%d).", state.get("iteration", 0))
    return "refine"


# ── Graph assembly ────────────────────────────────────────────────────────────


def build_rag_graph():
    workflow = StateGraph(RAGState)

    # Register nodes
    workflow.add_node("extract_keywords", extract_keywords_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("dispatch_retrieve", dispatch_retrieve_node)
    workflow.add_node("retrieve_semantic", retrieve_semantic_node)
    workflow.add_node("retrieve_bm25", retrieve_bm25_node)
    workflow.add_node("merge_retrieve", merge_retrieve_node)
    workflow.add_node("check_sufficiency", check_sufficiency_node)
    workflow.add_node("refine_query", refine_query_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("evaluate", evaluate_node)

    # Static edges
    workflow.add_edge(START, "extract_keywords")
    workflow.add_edge("extract_keywords", "rewrite_query")
    # Both rewrite_query and refine_query converge here before the fan-out
    workflow.add_edge("rewrite_query", "dispatch_retrieve")
    # Fan-out: both retrieve branches start after dispatch_retrieve
    workflow.add_edge("dispatch_retrieve", "retrieve_semantic")
    workflow.add_edge("dispatch_retrieve", "retrieve_bm25")
    # Fan-in: merge waits for both branches to complete
    workflow.add_edge("retrieve_semantic", "merge_retrieve")
    workflow.add_edge("retrieve_bm25", "merge_retrieve")
    workflow.add_edge("merge_retrieve", "check_sufficiency")

    # Conditional edge: check_sufficiency gates the loop
    workflow.add_conditional_edges(
        "check_sufficiency",
        _decide_after_sufficiency,
        {
            "refine": "refine_query",  # missing context → loop
            "generate": "generate",  # context is enough → answer
        },
    )

    # Loop-back: refine_query → dispatch_retrieve → fan-out (same path as initial)
    workflow.add_edge("refine_query", "dispatch_retrieve")

    # Terminal path: generate → evaluate → END
    workflow.add_edge("generate", "evaluate")
    workflow.add_edge("evaluate", END)

    return workflow.compile()


# Singleton graph compiled once at import time
rag_graph = build_rag_graph()
try:
    png_bytes = rag_graph.get_graph().draw_mermaid_png()  # warm up Mermaid renderer at startup
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    img.save("rag_graph.jpg", format="JPEG", quality=95)
except Exception as exc:
    logger.warning("Failed to render RAG graph visualization: %s", exc)

__all__ = ["rag_graph"]
