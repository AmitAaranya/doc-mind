"""LangGraph agent pipeline with tool routing and human-in-the-loop clarification.

Flow
----
START ──► classify_intent ──┬──► "rag"        → check_clarification ──┬──► needs_clarification? YES → END (sends clarification to user)
                            │                                          └──► needs_clarification? NO  → [full RAG pipeline]
                            ├──► "web_search"  → web_search → tool_answer → END
                            ├──► "weather"     → weather → tool_answer → END
                            ├──► "datetime"    → datetime → tool_answer → END
                            └──► "general"     → direct_answer → END

Clarification can happen at **two** points:
  1. **At start** (check_clarification) — the question is too vague to even begin retrieval
  2. **At sufficiency** (check_sufficiency) — retrieved context reveals ambiguity

When the user responds, the frontend re-invokes the pipeline with
``clarification_response`` set.  The pipeline merges the response into the
question and reruns from the appropriate point.

RAG sub-pipeline
----------------
  extract_keywords → rewrite_query → dispatch_retrieve → retrieve_semantic ──┐
                                            ▲                                 ├─► merge_retrieve → check_sufficiency
                                            │              retrieve_bm25 ────┘         │
                                       refine_query ◄─────────────────────────── sufficient? NO (& no clarification)
                                                                                       │
                                                                              needs_clarification? YES → END
                                                                                       │
                                                                                  sufficient? YES
                                                                                       │
                                                                                       ▼
                                                                                  generate → evaluate → END
"""

from __future__ import annotations

import io

from langgraph.graph import END, START, StateGraph
from PIL import Image

from app.core.logging import get_logger
from app.rag.evaluator import evaluate_node
from app.rag.nodes import (
    check_clarification_node,
    check_sufficiency_node,
    classify_intent_node,
    datetime_node,
    direct_answer_node,
    dispatch_retrieve_node,
    extract_keywords_node,
    generate_node,
    merge_retrieve_node,
    refine_query_node,
    retrieve_bm25_node,
    retrieve_semantic_node,
    rewrite_query_node,
    tool_answer_node,
    weather_node,
    web_search_node,
)
from app.rag.state import RAGState

logger = get_logger(__name__)


# ── Conditional edges ─────────────────────────────────────────────────────────


def _route_by_tool(state: RAGState) -> str:
    """After classify_intent: route to the selected tool pipeline."""
    tool = state.get("tool_name", "general")
    logger.info("Routing to tool: %s", tool)
    return tool


def _decide_after_clarification(state: RAGState) -> str:
    """After check_clarification: pause for user input or continue the RAG pipeline."""
    if state.get("needs_clarification", False):
        logger.info("Clarification needed — pausing pipeline for user input.")
        return "needs_clarification"
    logger.info("No clarification needed — proceeding to extract_keywords.")
    return "proceed"


def _decide_after_sufficiency(state: RAGState) -> str:
    """After check_sufficiency: loop back, ask user, or proceed to generate."""
    if state.get("needs_clarification", False):
        logger.info("Sufficiency check → clarification needed — pausing for user input.")
        return "clarify"
    if state.get("is_sufficient", False):
        logger.info("Context sufficient – proceeding to generate.")
        return "generate"
    logger.info("Context insufficient – refining query (iter=%d).", state.get("iteration", 0))
    return "refine"


# ── Graph assembly ────────────────────────────────────────────────────────────


def build_rag_graph():
    workflow = StateGraph(RAGState)

    # ── Agent router ──────────────────────────────────────────────────────────
    workflow.add_node("classify_intent", classify_intent_node)

    # ── Tool nodes ────────────────────────────────────────────────────────────
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("weather", weather_node)
    workflow.add_node("datetime", datetime_node)
    workflow.add_node("tool_answer", tool_answer_node)
    workflow.add_node("direct_answer", direct_answer_node)

    # ── RAG pipeline nodes ──────────────────────────────────────────────────
    workflow.add_node("check_clarification", check_clarification_node)
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

    # ── Entry point: classify intent first ────────────────────────────────────
    workflow.add_edge(START, "classify_intent")

    # ── Route by tool ─────────────────────────────────────────────────────────
    workflow.add_conditional_edges(
        "classify_intent",
        _route_by_tool,
        {
            "rag": "check_clarification",
            "web_search": "web_search",
            "weather": "weather",
            "datetime": "datetime",
            "general": "direct_answer",
        },
    )

    # ── Tool paths converge to tool_answer → END ─────────────────────────────
    workflow.add_edge("web_search", "tool_answer")
    workflow.add_edge("weather", "tool_answer")
    workflow.add_edge("datetime", "tool_answer")
    workflow.add_edge("tool_answer", END)
    workflow.add_edge("direct_answer", END)

    # ── Clarification gate: ask user or proceed to RAG ────────────────────────
    workflow.add_conditional_edges(
        "check_clarification",
        _decide_after_clarification,
        {
            "needs_clarification": END,  # pipeline pauses; UI shows clarification
            "proceed": "extract_keywords",  # question is clear; continue RAG
        },
    )

    # ── RAG pipeline edges ────────────────────────────────────────────────────
    workflow.add_edge("extract_keywords", "rewrite_query")
    workflow.add_edge("rewrite_query", "dispatch_retrieve")
    workflow.add_edge("dispatch_retrieve", "retrieve_semantic")
    workflow.add_edge("dispatch_retrieve", "retrieve_bm25")
    workflow.add_edge("retrieve_semantic", "merge_retrieve")
    workflow.add_edge("retrieve_bm25", "merge_retrieve")
    workflow.add_edge("merge_retrieve", "check_sufficiency")

    # Conditional edge: check_sufficiency gates the loop (3-way)
    workflow.add_conditional_edges(
        "check_sufficiency",
        _decide_after_sufficiency,
        {
            "refine": "refine_query",       # missing context → loop
            "generate": "generate",          # context is enough → answer
            "clarify": END,                  # ambiguity detected → ask user
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
