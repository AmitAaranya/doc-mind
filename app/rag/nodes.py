"""LangGraph node implementations for the RAG pipeline.

Nodes
-----
1. extract_keywords_node    – extract key terms from the raw question
2. rewrite_query_node       – optimise query for dense-vector retrieval
3. retrieve_node            – hybrid semantic + keyword search in ChromaDB
4. check_sufficiency_node   – LLM decides if retrieved context covers the question
                              (loop gate: insufficient → refine_query, sufficient → generate)
5. generate_node            – build answer from retrieved context
6. refine_query_node        – LLM generates a targeted follow-up query and decides
                              whether to run keyword, semantic, or hybrid retrieval
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from typing import Any

from app.core.logging import get_logger
from app.database.chroma import ChromaVectorStore
from app.llm import embeddings, llm_chat
from app.rag.state import RAGState

logger = get_logger(__name__)

_vector_store = ChromaVectorStore()

# ── Thread-local token streaming callback ─────────────────────────────────────
# The streaming HTTP route sets this before running the graph so that
# generate_node can push LLM tokens into the SSE queue in real time.

_tl = threading.local()


def set_token_callback(cb: Callable[[str], None]) -> None:
    """Register a per-thread callback invoked with each LLM output token."""
    _tl.token_callback = cb


def clear_token_callback() -> None:
    """Remove the token callback for the current thread."""
    _tl.token_callback = None


# ── System prompts ────────────────────────────────────────────────────────────

_KEYWORDS_SYSTEM = """\
You are a keyword-extraction and retrieval-planning expert.
Given a user question:
1. Extract the most important search keywords, entities, and noun phrases.
2. Estimate how many document chunks (top_k) are needed to fully answer the question.

Retrieval depth guide:
  - Simple factual / single-concept question → top_k: 3-5
  - Multi-part or comparative question        → top_k: 6-10
  - Broad research / open-ended question      → top_k: 11-20

Respond with ONLY a valid JSON object — no markdown, no explanation:
{
  "keywords": ["<term1>", "<term2>", ...],
  "top_k": <integer between 3 and 20>,
  "reasoning": "<one sentence why you chose this depth>"
}\
"""

_REWRITE_SYSTEM = """\
You are an expert at optimising questions for dense-vector semantic retrieval.
Rewrite the given question into a concise, descriptive declarative statement that maximises
recall against a document collection.  Expand abbreviations, add synonyms, and focus on the
core information need.
Return ONLY the rewritten query — no markdown, no explanation.\
"""

_GENERATE_SYSTEM = """\
You are a precise document-analyst assistant.
Answer the user's question based ONLY on the provided numbered context passages.
- Be factual and cite every claim with the source index, e.g. [1], [2].
- If the context is insufficient to answer fully, clearly state what is missing.
- End your answer with a "References" section listing the document names used.\
"""

_SUFFICIENCY_SYSTEM = """\
You are a retrieval-quality analyst.
Given a user question, a list of key terms, and a set of retrieved document passages,
decide whether the retrieved context contains enough information to answer the question fully.

Respond with a JSON object using EXACTLY this schema (no markdown, no extra keys):
{
  "sufficient": true | false,
  "reason": "<one sentence explaining what is covered or what is still missing>",
  "missing_aspects": ["<aspect1>", "<aspect2>"]   // empty list if sufficient
}\
"""

_REFINE_SYSTEM = """\
You are an expert at iterative information retrieval.
Given the original question, the keywords identified, what is still missing, and all
queries already tried, generate ONE new targeted search query.

Also decide the best retrieval strategy for this query:
  - "keyword"  → when the missing information requires exact term matching
  - "semantic" → when the missing information is conceptual / paraphrased
  - "hybrid"   → when both are needed

Respond with a JSON object (no markdown, no extra keys):
{
  "query": "<new search query>",
  "search_mode": "keyword" | "semantic" | "hybrid"
}\
"""

# ── Utility ───────────────────────────────────────────────────────────────────


def _stream_and_collect(iterator) -> str:
    """Drain a streaming iterator, firing the thread-local token callback for every token.

    Every node that calls the LLM goes through this helper so that the SSE
    stream receives live tokens regardless of which pipeline stage is active.
    """
    token_cb: Callable[[str], None] | None = getattr(_tl, "token_callback", None)
    parts: list[str] = []
    for tok in iterator:
        if tok:
            parts.append(tok)
            if token_cb:
                token_cb(tok)
    return "".join(parts)


def _chromadb_rows(raw: dict) -> list[dict[str, Any]]:
    """Convert a raw ChromaDB query response dict into a list of chunk dicts."""
    rows: list[dict[str, Any]] = []
    ids = raw.get("ids", [[]])[0]
    docs = raw.get("documents", [[]])[0]
    metas = raw.get("metadatas", [[]])[0]
    dists = raw.get("distances", [[]])[0]
    for doc_id, doc, meta, dist in zip(ids, docs, metas, dists):
        rows.append(
            {
                "id": doc_id,
                "document": doc,
                "metadata": meta or {},
                "distance": dist,
            }
        )
    return rows


# ── Node 1: extract_keywords ──────────────────────────────────────────────────


def extract_keywords_node(state: RAGState) -> dict[str, Any]:
    question = state["question"]

    keywords: list[str] = []
    top_k: int = 8  # fallback default
    reasoning: str = ""

    try:
        raw = _stream_and_collect(
            llm_chat.stream_text(
                prompt=f"User question: {question}",
                system_instruction=_KEYWORDS_SYSTEM,
                temperature=0.2,
            )
        )
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else {}
        keywords = [str(k).strip() for k in parsed.get("keywords", []) if k and str(k).strip()]
        raw_k = parsed.get("top_k", 8)
        top_k = max(3, min(20, int(raw_k)))  # clamp 3-20
        reasoning = parsed.get("reasoning", "")
    except Exception:
        # Fallback: simple word extraction, conservative top_k
        stop = {
            "what",
            "how",
            "why",
            "when",
            "where",
            "who",
            "is",
            "are",
            "the",
            "a",
            "an",
            "in",
            "on",
            "of",
            "for",
            "to",
            "and",
            "or",
            "does",
            "do",
            "can",
            "could",
            "would",
            "should",
            "will",
        }
        keywords = [w for w in re.findall(r"\b\w{3,}\b", question.lower()) if w not in stop]
        top_k = 8

    logger.info("Extracted %d keywords, top_k=%d: %s", len(keywords), top_k, keywords)
    kw_preview = ", ".join(keywords[:6]) + (
        f" (+{len(keywords) - 6} more)" if len(keywords) > 6 else ""
    )
    desc = (
        f"Identified {len(keywords)} key term(s): {kw_preview}. "
        f"Retrieval depth: {top_k} chunks" + (f" ({reasoning})" if reasoning else "")
        if keywords
        else f"No key terms found — using full question. Retrieval depth: {top_k} chunks."
    )
    return {
        "keywords": keywords,
        "top_k": top_k,
        "search_queries_tried": [],
        "all_chunk_ids": [],
        "iteration": 0,
        "step": {
            "title": "Keywords Extracted",
            "description": desc,
        },
    }


# ── Node 2: rewrite_query ─────────────────────────────────────────────────────


def rewrite_query_node(state: RAGState) -> dict[str, Any]:
    question = state["question"]
    keywords = state.get("keywords", [])

    prompt = f"Original question: {question}\nKey terms identified: {', '.join(keywords)}"

    optimized_query = _stream_and_collect(
        llm_chat.stream_text(
            prompt=prompt,
            system_instruction=_REWRITE_SYSTEM,
            temperature=0.3,
        )
    ).strip()

    if not optimized_query:
        optimized_query = question  # safe fallback

    logger.info("Optimised query: %s", optimized_query)
    return {
        "optimized_query": optimized_query,
        "step": {
            "title": "Query Optimised",
            "description": f'Rewritten for dense-vector retrieval: "{optimized_query}"',
        },
    }


# ── Node 3: retrieve ──────────────────────────────────────────────────────────


# Cosine distance threshold — distance < 0.5 ↔ cosine similarity > 0.5 (>50%)
_SIMILARITY_THRESHOLD = 0.5


def retrieve_node(state: RAGState) -> dict[str, Any]:
    iteration = state.get("iteration", 0)
    all_chunk_ids: list[str] = state.get("all_chunk_ids", [])
    keywords: list[str] = state.get("keywords", [])
    queries_tried: list[str] = state.get("search_queries_tried", [])
    top_k: int = state.get("top_k", 8)

    # Choose which query to run this iteration
    if iteration == 0:
        active_query = state.get("optimized_query") or state["question"]
    else:
        # refine_query_node appended the new query to search_queries_tried
        active_query = (
            queries_tried[-1] if queries_tried else state.get("optimized_query", state["question"])
        )

    if active_query not in queries_tried:
        queries_tried = queries_tried + [active_query]

    query_vec = embeddings.embed_documents([active_query])[0]

    # ── Semantic search (pure vector similarity) ──────────────────────────────
    # Over-fetch so we still have top_k after the similarity cut.
    semantic_chunks: list[dict[str, Any]] = _vector_store.query(
        query_embedding=query_vec, n_results=top_k * 2
    )

    # ── Keyword search (exact-text containment, independent of semantic) ──────
    # Each keyword runs its own ChromaDB query; results are unioned before dedup.
    keyword_chunks: list[dict[str, Any]] = []
    kw_fetch = max(3, top_k // 2)  # candidates per keyword
    for kw in keywords[:5]:
        try:
            raw = _vector_store.collection.query(
                query_embeddings=[query_vec],
                n_results=kw_fetch,
                where_document={"$contains": kw},
                include=["documents", "metadatas", "distances"],
            )
            keyword_chunks.extend(_chromadb_rows(raw))
        except Exception as exc:
            logger.debug("Keyword search for %r skipped: %s", kw, exc)

    # ── Merge & deduplicate (semantic first so lower-distance entry wins) ─────
    seen: set[str] = set(all_chunk_ids)
    candidates: list[dict[str, Any]] = []
    for chunk in semantic_chunks + keyword_chunks:
        if chunk["id"] not in seen:
            seen.add(chunk["id"])
            candidates.append(chunk)

    # ── Similarity filter: keep only chunks with >50% cosine similarity ───────
    # ChromaDB cosine distance: 0 = identical, 2 = opposite (normalised vectors)
    # distance < 0.5  ↔  cosine_similarity > 0.5
    relevant = [c for c in candidates if c.get("distance", 1.0) < _SIMILARITY_THRESHOLD]

    # Sort by distance ascending (most similar first) and cap at top_k
    relevant.sort(key=lambda c: c.get("distance", 1.0))
    top_chunks = relevant[:top_k]

    updated_ids = all_chunk_ids + [c["id"] for c in top_chunks]

    # ── Extract unique source / document names ────────────────────────────────
    references = sorted(
        {
            c["metadata"].get("source_file")
            or c["metadata"].get("source")
            or c["metadata"].get("filename")
            or c["metadata"].get("file_name")
            or "unknown"
            for c in top_chunks
        }
    )

    discarded = len(candidates) - len(relevant)
    logger.info(
        "Retrieve iter=%d: semantic=%d keyword=%d candidates=%d relevant(>50%%)=%d kept=%d discarded=%d refs=%s",
        iteration,
        len(semantic_chunks),
        len(keyword_chunks),
        len(candidates),
        len(relevant),
        len(top_chunks),
        discarded,
        references,
    )
    active_q = queries_tried[-1] if queries_tried else ""
    doc_part = f"{len(references)} document(s)" if references else "no matching documents"
    return {
        "retrieved_chunks": top_chunks,
        "all_chunk_ids": updated_ids,
        "references": references,
        "search_queries_tried": queries_tried,
        "step": {
            "title": "Documents Retrieved",
            "description": (
                f"Found {len(top_chunks)} passage(s) from {doc_part} "
                f"(semantic + keyword, >50% similarity"
                + (f", {discarded} low-similarity discarded" if discarded else "")
                + ")"
                + (f' — "{active_q}"' if active_q else "")
            ),
        },
    }


# ── Node 4: check_sufficiency ─────────────────────────────────────────────────


def check_sufficiency_node(state: RAGState) -> dict[str, Any]:
    """LLM gate: analyse retrieved chunks against the question.

    Sets ``is_sufficient`` (bool) and ``sufficiency_reason`` (str) in state.
    The graph uses ``is_sufficient`` to decide whether to loop back through
    refine_query → retrieve or to proceed to generate.
    """
    question = state["question"]
    keywords = state.get("keywords", [])
    chunks = state.get("retrieved_chunks", [])
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)

    # If we've hit the ceiling, just proceed regardless
    if iteration >= max_iterations - 1 or not chunks:
        reason = (
            "max iterations reached" if iteration >= max_iterations - 1 else "no chunks retrieved"
        )
        logger.info("Sufficiency check skipped (%s) — forcing proceed to generate.", reason)
        return {
            "is_sufficient": True,
            "sufficiency_reason": reason,
            "step": {
                "title": "Context Check Skipped",
                "description": f"Proceeding to answer ({reason}).",
            },
        }

    # Build a compact context summary for the LLM (first 300 chars per chunk)
    snippet_lines = [f"[{i + 1}] {c['document'][:300]}" for i, c in enumerate(chunks[:6])]
    context_preview = "\n".join(snippet_lines)

    prompt = (
        f"Question: {question}\n"
        f"Key terms: {', '.join(keywords)}\n\n"
        f"Retrieved passages (truncated):\n{context_preview}\n\n"
        "Analyse whether these passages together contain enough information to fully answer the question."
    )

    raw = _stream_and_collect(
        llm_chat.stream_text(
            prompt=prompt,
            system_instruction=_SUFFICIENCY_SYSTEM,
            temperature=0.1,
        )
    )

    is_sufficient = False
    reason = "could not parse sufficiency response"
    try:
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else {}
        is_sufficient = bool(parsed.get("sufficient", False))
        reason = parsed.get("reason", reason)
        missing = parsed.get("missing_aspects", [])
        if missing:
            reason += f" Missing: {', '.join(missing)}"
    except Exception as exc:
        logger.debug("Sufficiency JSON parse error: %s | raw=%r", exc, raw[:200])

    logger.info(
        "Sufficiency check iter=%d: sufficient=%s — %s",
        iteration,
        is_sufficient,
        reason,
    )
    if is_sufficient:
        su_title = "Context Complete"
        su_desc = (
            f"All required information found — proceeding to generate answer. {reason}".strip()
        )
    else:
        su_title = "Context Incomplete"
        su_desc = f"More information needed — will refine search. {reason}".strip()
    return {
        "is_sufficient": is_sufficient,
        "sufficiency_reason": reason,
        "step": {"title": su_title, "description": su_desc},
    }


# ── Node 5: generate ─────────────────────────────────────────────────────────


def generate_node(state: RAGState) -> dict[str, Any]:
    question = state["question"]
    chunks = state.get("retrieved_chunks", [])
    references = state.get("references", [])

    if not chunks:
        return {
            "answer": "No relevant document passages were found to answer this question.",
            "references": [],
            "step": {
                "title": "Generating Answer",
                "description": "No passages retrieved — cannot generate an answer.",
            },
        }

    # Build numbered context block
    context_parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        src = (
            chunk["metadata"].get("source_file")
            or chunk["metadata"].get("source")
            or chunk["metadata"].get("filename")
            or chunk["metadata"].get("file_name")
            or "unknown"
        )
        page = chunk["metadata"].get("page", "")
        loc = f"{src}" + (f" (p.{page})" if page else "")
        context_parts.append(f"[{i}] {loc}\n{chunk['document']}")

    context = "\n\n---\n\n".join(context_parts)

    prompt = (
        f"Question: {question}\n\n"
        f"Context passages:\n{context}\n\n"
        "Answer the question comprehensively using the passages above. "
        "Cite each claim with its passage number, e.g. [1]. "
        "List all referenced document names at the end under 'References:'."
    )

    answer = _stream_and_collect(
        llm_chat.stream_text(
            prompt=prompt,
            system_instruction=_GENERATE_SYSTEM,
            temperature=0.4,
        )
    ).strip()

    # Ensure references section is appended even if LLM skipped it
    if references and "References:" not in answer:
        answer += "\n\nReferences:\n" + "\n".join(f"- {r}" for r in references)

    logger.info("Generated answer: %d chars from %d chunks", len(answer), len(chunks))
    return {
        "answer": answer,
        "references": references,
        "step": {
            "title": "Answer Generated",
            "description": f"Synthesised a {len(answer):,}-character answer citing {len(references)} document(s).",
        },
    }


# ── Node 6: refine_query ──────────────────────────────────────────────────────


def refine_query_node(state: RAGState) -> dict[str, Any]:
    """Generate a new targeted search query and decide the retrieval strategy.

    Reads ``sufficiency_reason`` to understand what is missing, then asks the
    LLM to produce a query + ``search_mode`` (keyword | semantic | hybrid).
    """
    question = state["question"]
    keywords = state.get("keywords", [])
    queries_tried = state.get("search_queries_tried", [])
    reason = state.get("sufficiency_reason", "")
    iteration = state.get("iteration", 0)

    prompt = (
        f"Original question: {question}\n"
        f"Key terms: {', '.join(keywords)}\n"
        f"What is still missing / insufficient: {reason}\n"
        f"Queries already tried:\n" + "\n".join(f"  - {q}" for q in queries_tried) + "\n\n"
        "Generate a new query targeting the missing information, and select the best search mode."
    )

    raw = _stream_and_collect(
        llm_chat.stream_text(
            prompt=prompt,
            system_instruction=_REFINE_SYSTEM,
            temperature=0.6,
        )
    ).strip()

    new_query: str = ""
    search_mode: str = "hybrid"
    try:
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else {}
        new_query = parsed.get("query", "").strip()
        search_mode = parsed.get("search_mode", "hybrid")
        if search_mode not in ("keyword", "semantic", "hybrid"):
            search_mode = "hybrid"
    except Exception as exc:
        logger.debug("Refine JSON parse error: %s | raw=%r", exc, raw[:200])

    if not new_query or new_query in queries_tried:
        fallback_kw = keywords[iteration % len(keywords)] if keywords else "details"
        new_query = f"{question} {fallback_kw}"
        search_mode = "hybrid"

    updated_queries = queries_tried + [new_query]
    logger.info(
        "Refined query (iter %d→%d) mode=%s: %s",
        iteration,
        iteration + 1,
        search_mode,
        new_query,
    )
    mode_label = {
        "keyword": "keyword (exact match)",
        "semantic": "semantic (conceptual)",
        "hybrid": "hybrid",
    }.get(search_mode, search_mode)
    return {
        "search_queries_tried": updated_queries,
        "search_mode": search_mode,
        "iteration": iteration + 1,
        "step": {
            "title": f"Search Refined (loop {iteration + 1})",
            "description": f'Switching to {mode_label} search — new query: "{new_query}"',
        },
    }
