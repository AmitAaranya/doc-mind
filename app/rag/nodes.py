"""LangGraph node implementations for the RAG pipeline.

Nodes
-----
0.  classify_intent_node      – LLM classifies query intent and routes to the right tool
0.5 check_clarification_node  – LLM decides if the question is too vague and asks
                                 the user for clarification (human-in-the-loop)
1.  extract_keywords_node     – extract key terms from the raw question
2.  rewrite_query_node        – optimise query for dense-vector retrieval
3.  retrieve_node             – hybrid semantic + keyword search in ChromaDB
4.  check_sufficiency_node    – LLM decides if retrieved context covers the question
                                 (three-way gate: sufficient → generate,
                                  insufficient → refine_query,
                                  ambiguous → ask user for clarification)
5.  generate_node             – build answer from retrieved context
6.  refine_query_node         – LLM generates a targeted follow-up query
7.  web_search_node           – search the internet via DuckDuckGo
8.  weather_node              – fetch current weather for a location
9.  datetime_node             – return current date/time
10. direct_answer_node        – LLM answers general/casual questions directly
11. tool_answer_node          – LLM generates answer from tool results

Human-in-the-Loop
-----------------
Clarification can be triggered at two points:
  1. **check_clarification** — before retrieval, when the query is too vague
  2. **check_sufficiency**   — after retrieval, when context reveals ambiguity

When more info is needed, the pipeline sends a `clarification` SSE event
(with a question + optional selectable options) and terminates.  The user
responds and the frontend re-invokes the pipeline with
``clarification_response`` set.  ``check_clarification_node`` enriches the
question, and the full pipeline re-runs — including a fresh sufficiency
check on the new, better-focused retrieval results.

To prevent infinite loops, ``check_sufficiency`` will only ask for
clarification once (it checks ``clarification_response`` to see if the user
already clarified).
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from typing import Any

from langchain_community.retrievers import BM25Retriever

from app.core.logging import get_logger
from app.database.bm25_store import BM25CorpusStore
from app.database.chroma import ChromaVectorStore
from app.llm import embeddings, llm_chat
from app.rag.prompts import (
    CLARIFICATION_SYSTEM,
    DIRECT_ANSWER_SYSTEM,
    GENERATE_SYSTEM,
    KEYWORDS_SYSTEM,
    REFINE_SYSTEM,
    REWRITE_SYSTEM,
    SUFFICIENCY_SYSTEM,
    TOOL_ANSWER_SYSTEM,
)
from app.rag.state import RAGState
from app.rag.tools import (
    FUNCTION_TO_TOOL,
    get_current_datetime,
    get_gemini_tool_declarations,
    get_weather,
    web_search,
)

logger = get_logger(__name__)

_vector_store = ChromaVectorStore()
_bm25_corpus = BM25CorpusStore()

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


def set_current_node(name: str) -> None:
    """Track which pipeline node is currently executing (thread-local)."""
    _tl.current_node = name


def get_current_node() -> str:
    """Return the name of the currently executing pipeline node."""
    return getattr(_tl, "current_node", "unknown")


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


# ── Node 0.5: check_clarification (human-in-the-loop) ─────────────────────────


def check_clarification_node(state: RAGState) -> dict[str, Any]:
    """Decide whether the user's question needs clarification before proceeding.

    If ``clarification_response`` is already present in state (user replied to
    a previous clarification request), the original question is enriched with
    the user's response and the pipeline continues without re-asking.

    Otherwise the LLM evaluates the question's clarity.  When clarification is
    needed the node sets ``needs_clarification=True`` and provides a question
    + optional selection options.  The graph routes to END so the SSE stream
    can deliver the clarification request to the client.
    """
    set_current_node("check_clarification")
    question = state["question"]
    clarification_response = state.get("clarification_response", "")

    # ── User already responded to a clarification → enrich question & proceed ─
    if clarification_response:
        enriched = f"{question}\nUser clarification: {clarification_response}"
        logger.info("Clarification received — enriched question: %s", enriched)
        return {
            "question": enriched,
            "needs_clarification": False,
            "clarification_question": "",
            "clarification_options": [],
            "clarification_source": "",
            "step": {
                "title": "Clarification Received",
                "description": f"User clarified: \"{clarification_response}\" — proceeding.",
            },
        }

    # ── Ask the LLM whether clarification is needed ───────────────────────────
    prompt = f"User question: {question}"

    raw = _stream_and_collect(
        llm_chat.stream_text(
            prompt=prompt,
            system_instruction=CLARIFICATION_SYSTEM,
            temperature=0.1,
        )
    )

    needs = False
    clar_question = ""
    clar_options: list[dict[str, str]] = []
    reason = ""

    try:
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else {}
        needs = bool(parsed.get("needs_clarification", False))
        clar_question = parsed.get("question", "").strip()
        raw_options = parsed.get("options", [])
        # Sanitise options — keep only well-formed entries
        for opt in raw_options:
            if isinstance(opt, dict) and "label" in opt and "value" in opt:
                clar_options.append(
                    {"label": str(opt["label"]), "value": str(opt["value"])}
                )
        reason = parsed.get("reason", "")
    except Exception as exc:
        logger.debug("Clarification JSON parse error: %s | raw=%r", exc, raw[:200])

    if needs and clar_question:
        logger.info("Clarification needed: %s (options=%d)", clar_question, len(clar_options))
        return {
            "needs_clarification": True,
            "clarification_question": clar_question,
            "clarification_options": clar_options,
            "clarification_source": "start",
            "step": {
                "title": "Clarification Needed",
                "description": reason or "The question is ambiguous — asking the user.",
            },
        }

    logger.info("No clarification needed — proceeding to keyword extraction.")
    return {
        "needs_clarification": False,
        "clarification_question": "",
        "clarification_options": [],
        "clarification_source": "",
        "step": {
            "title": "Question Clear",
            "description": reason or "Question is well-formed — proceeding to retrieval.",
        },
    }


# ── Node 1: extract_keywords ──────────────────────────────────────────────────


def extract_keywords_node(state: RAGState) -> dict[str, Any]:
    set_current_node("extract_keywords")
    question = state["question"]

    keywords: list[str] = []
    top_k: int = 8  # fallback default
    reasoning: str = ""

    try:
        raw = _stream_and_collect(
            llm_chat.stream_text(
                prompt=f"User question: {question}",
                system_instruction=KEYWORDS_SYSTEM,
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
    set_current_node("rewrite_query")
    question = state["question"]
    keywords = state.get("keywords", [])

    prompt = f"Original question: {question}\nKey terms identified: {', '.join(keywords)}"

    optimized_query = _stream_and_collect(
        llm_chat.stream_text(
            prompt=prompt,
            system_instruction=REWRITE_SYSTEM,
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
_SIMILARITY_THRESHOLD = 0.8


# ── Node 3: dispatch_retrieve ─────────────────────────────────────────────────


def dispatch_retrieve_node(state: RAGState) -> dict[str, Any]:
    """Pass-through dispatcher that fans out to the parallel retrieve branches.

    Acts as a single convergence point so both ``rewrite_query`` and
    ``refine_query`` only need one outgoing edge.  The node resolves and
    records the active query, then LangGraph fans out to
    ``retrieve_semantic`` and ``retrieve_bm25`` in parallel.
    """
    active_q, queries_tried = _active_query(state)
    iteration = state.get("iteration", 0)
    logger.info(
        "Dispatch retrieve iter=%d — query: %r (mode=%s)",
        iteration,
        active_q,
        state.get("search_mode", "hybrid"),
    )
    return {
        "search_queries_tried": queries_tried,
        "step": {
            "title": "Dispatching Retrieval",
            "description": (f'Launching parallel semantic + BM25 search — "{active_q}"'),
        },
    }


def _active_query(state: RAGState) -> tuple[str, list[str]]:
    """Return (active_query, updated_queries_tried) for the current iteration."""
    iteration = state.get("iteration", 0)
    queries_tried: list[str] = state.get("search_queries_tried", [])
    if iteration == 0:
        query = state.get("optimized_query") or state["question"]
    else:
        query = (
            queries_tried[-1] if queries_tried else state.get("optimized_query", state["question"])
        )
    if query not in queries_tried:
        queries_tried = queries_tried + [query]
    return query, queries_tried


# ── Node 3a: retrieve_semantic ────────────────────────────────────────────────


def retrieve_semantic_node(state: RAGState) -> dict[str, Any]:
    """Dense-vector similarity search against ChromaDB (runs in parallel with BM25)."""
    top_k: int = state.get("top_k", 8)
    active_q, queries_tried = _active_query(state)
    user_id = state["user_id"]

    query_vec = embeddings.embed_documents([active_q])[0]
    # Over-fetch so we still have top_k after the similarity cut.
    chunks: list[dict[str, Any]] = _vector_store.query(
        query_embedding=query_vec,
        n_results=top_k * 2,
        where={"user_id": user_id},
    )

    logger.debug("Semantic search: query=%r hits=%d", active_q, len(chunks))
    return {
        "semantic_chunks": chunks,
        "search_queries_tried": queries_tried,
        "step": {
            "title": "Semantic Search",
            "description": (
                f'Vector similarity search returned {len(chunks)} candidate(s) — "{active_q}"'
            ),
        },
    }


# ── Node 3b: retrieve_bm25 ────────────────────────────────────────────────────


def retrieve_bm25_node(state: RAGState) -> dict[str, Any]:
    """BM25 keyword search over the SQLite corpus (runs in parallel with semantic)."""
    keywords: list[str] = state.get("keywords", [])
    top_k: int = state.get("top_k", 8)
    active_q, _ = _active_query(state)
    user_id = state["user_id"]

    kw_fetch = max(3, top_k // 2)
    bm25_chunks: list[dict[str, Any]] = []
    corpus = _bm25_corpus.get_all(user_id=user_id)

    if corpus and keywords:
        try:
            bm25_query = " ".join(keywords)
            bm25 = BM25Retriever.from_texts(
                texts=[c["document"] for c in corpus],
                metadatas=[{**c["metadata"], "_chunk_id": c["id"]} for c in corpus],
                k=kw_fetch,
            )
            tokens = bm25.preprocess_func(bm25_query)
            scores = bm25.vectorizer.get_scores(tokens)
            max_score = float(max(scores)) if len(scores) > 0 and max(scores) > 0 else 1.0
            id_to_score = {corpus[i]["id"]: float(scores[i]) for i in range(len(corpus))}

            for doc in bm25.invoke(bm25_query):
                cid = doc.metadata.get("_chunk_id", "")
                if not cid:
                    continue
                raw_score = id_to_score.get(cid, 0.0)
                norm_score = raw_score / max_score if max_score > 0 else 0.0
                distance = 1.0 - norm_score
                meta = {k: v for k, v in doc.metadata.items() if k != "_chunk_id"}
                bm25_chunks.append(
                    {
                        "id": cid,
                        "document": doc.page_content,
                        "metadata": meta,
                        "distance": distance,
                    }
                )
            logger.debug(
                "BM25 search: corpus=%d query=%r hits=%d",
                len(corpus),
                bm25_query,
                len(bm25_chunks),
            )
        except Exception as exc:
            logger.debug("BM25 search failed: %s", exc)

    return {
        "bm25_chunks": bm25_chunks,
        "step": {
            "title": "BM25 Search",
            "description": (
                f'BM25 keyword search returned {len(bm25_chunks)} candidate(s) — "{active_q}"'
            ),
        },
    }


# ── Node 3c: merge_retrieve ───────────────────────────────────────────────────


def merge_retrieve_node(state: RAGState) -> dict[str, Any]:
    """Merge, dedup, filter, and rank results from the two parallel retrieve branches."""
    iteration = state.get("iteration", 0)
    all_chunk_ids: list[str] = state.get("all_chunk_ids", [])
    queries_tried: list[str] = state.get("search_queries_tried", [])
    top_k: int = state.get("top_k", 8)
    semantic_chunks: list[dict[str, Any]] = state.get("semantic_chunks", [])
    bm25_chunks: list[dict[str, Any]] = state.get("bm25_chunks", [])

    # ── Merge & deduplicate (semantic first so lower-distance entry wins) ─────
    seen: set[str] = set(all_chunk_ids)
    candidates: list[dict[str, Any]] = []
    for chunk in semantic_chunks + bm25_chunks:
        if chunk["id"] not in seen:
            seen.add(chunk["id"])
            candidates.append(chunk)

    # ── Similarity filter: keep only chunks with cosine similarity ───────
    relevant = [c for c in candidates if c.get("distance", 1.0) < _SIMILARITY_THRESHOLD]

    # Sort by distance ascending (most similar first) and cap at top_k
    relevant.sort(key=lambda c: c.get("distance", 1.0))
    top_chunks = relevant[:top_k]

    updated_ids = all_chunk_ids + [c["id"] for c in top_chunks]
    all_retrieved_chunks = state.get("all_retrieved_chunks", []) + top_chunks

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
        "Merge retrieve iter=%d: semantic=%d bm25=%d candidates=%d"
        " relevant=%d kept=%d discarded=%d refs=%s",
        iteration,
        len(semantic_chunks),
        len(bm25_chunks),
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
        "all_retrieved_chunks": all_retrieved_chunks,
        "references": references,
        # Clear intermediates so stale data never bleeds into the next loop
        "semantic_chunks": [],
        "bm25_chunks": [],
        "step": {
            "title": "Documents Retrieved",
            "description": (
                f"Found {len(top_chunks)} passage(s) from {doc_part} "
                f"(semantic + BM25"
                + (f", {discarded} low-similarity discarded" if discarded else "")
                + ")"
                + (f' — "{active_q}"' if active_q else "")
            ),
        },
    }


# ── Node 4: check_sufficiency ─────────────────────────────────────────────────


def check_sufficiency_node(state: RAGState) -> dict[str, Any]:
    """LLM gate: analyse retrieved chunks against the question.

    Three possible outcomes:
      1. **sufficient** → proceed to generate
      2. **insufficient** → refine query and re-retrieve
      3. **needs_clarification** → pause pipeline, ask the user

    The clarification path is only available when no prior clarification has
    been provided (``clarification_response`` is empty).  This prevents
    infinite clarification loops — once the user has already clarified, the
    sufficiency check sticks to sufficient/insufficient decisions.

    After the user responds to a clarification, the pipeline re-runs from
    scratch with the enriched question, producing better keywords, better
    retrieval, and then this node re-evaluates sufficiency with the new
    context.
    """
    set_current_node("check_sufficiency")
    question = state["question"]
    keywords = state.get("keywords", [])
    chunks = state.get("retrieved_chunks", [])
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)
    already_clarified = bool(state.get("clarification_response", ""))

    # If we've hit the ceiling, just proceed regardless
    if iteration >= max_iterations - 1 or not chunks:
        reason = (
            "max iterations reached" if iteration >= max_iterations - 1 else "no chunks retrieved"
        )
        logger.info("Sufficiency check skipped (%s) — forcing proceed to generate.", reason)
        return {
            "is_sufficient": True,
            "needs_clarification": False,
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
            system_instruction=SUFFICIENCY_SYSTEM,
            temperature=0.1,
        )
    )

    is_sufficient = False
    needs_clarification = False
    clar_question = ""
    clar_options: list[dict[str, str]] = []
    reason = "could not parse sufficiency response"
    try:
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else {}
        is_sufficient = bool(parsed.get("sufficient", False))
        needs_clarification = bool(parsed.get("needs_clarification", False))
        clar_question = parsed.get("clarification_question", "").strip()
        raw_options = parsed.get("clarification_options", [])
        for opt in raw_options:
            if isinstance(opt, dict) and "label" in opt and "value" in opt:
                clar_options.append(
                    {"label": str(opt["label"]), "value": str(opt["value"])}
                )
        reason = parsed.get("reason", reason)
        missing = parsed.get("missing_aspects", [])
        if missing:
            reason += f" Missing: {', '.join(missing)}"
    except Exception as exc:
        logger.debug("Sufficiency JSON parse error: %s | raw=%r", exc, raw[:200])

    # ── Clarification path (only if user hasn't clarified yet) ────────────────
    # After one clarification round, we never ask again — we either proceed to
    # generate with what we have or refine the query automatically.
    if needs_clarification and clar_question and not is_sufficient and not already_clarified:
        logger.info(
            "Sufficiency → clarification needed (iter=%d): %s",
            iteration,
            clar_question,
        )
        return {
            "is_sufficient": False,
            "needs_clarification": True,
            "clarification_question": clar_question,
            "clarification_options": clar_options,
            "clarification_source": "sufficiency",
            "sufficiency_reason": reason,
            "step": {
                "title": "Clarification Needed",
                "description": reason or "Retrieved context is ambiguous — asking the user.",
            },
        }

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
        "needs_clarification": False,
        "sufficiency_reason": reason,
        "step": {"title": su_title, "description": su_desc},
    }


# ── Adjacent-chunk merger ─────────────────────────────────────────────────────


def _merge_adjacent_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-order retrieved chunks by document position and merge adjacent ones.

    Two chunks are considered *adjacent* when they originate from the same
    source document and sit next to each other in reading order:

    * **Same block** (same ``source_file`` + ``page_number`` + ``block_order``):
      the chunks were produced by splitting one text block with overlap.
      We remove the duplicated overlap region so the merged text is clean.

    * **Consecutive blocks** (same ``source_file`` + ``page_number``,
      ``block_order`` differs by 1): neighbouring paragraphs / sections.
      We join them with a blank line — no overlap to strip.

    All other combinations remain as independent entries.

    The merged chunk inherits the lowest (most similar) distance of its parts
    and the metadata of the first (positionally earliest) chunk.
    """
    if len(chunks) <= 1:
        return chunks

    def _pos(c: dict) -> tuple:
        m = c.get("metadata", {})
        return (
            m.get("source_file", ""),
            m.get("page_number", 0),
            m.get("block_order", 0),
            m.get("char_start", 0),
        )

    sorted_chunks = sorted(chunks, key=_pos)
    merged: list[dict[str, Any]] = []
    current = dict(sorted_chunks[0])  # shallow copy so we don't mutate state

    for nxt in sorted_chunks[1:]:
        cm = current.get("metadata", {})
        nm = nxt.get("metadata", {})

        same_source = cm.get("source_file") == nm.get("source_file")
        same_page = cm.get("page_number") == nm.get("page_number")
        same_block = cm.get("block_order") == nm.get("block_order")
        consec_block = (
            same_source
            and same_page
            and (
                isinstance(cm.get("block_order"), int)
                and isinstance(nm.get("block_order"), int)
                and nm["block_order"] == cm["block_order"] + 1
            )
        )

        if same_source and same_page and same_block and "char_start" in cm and "char_start" in nm:
            # ── Same block split: remove overlapping prefix ───────────────
            cur_text: str = current["document"]
            nxt_text: str = nxt["document"]
            cur_start: int = cm["char_start"]
            nxt_start: int = nm["char_start"]
            overlap = (cur_start + len(cur_text)) - nxt_start
            if overlap > 0:
                merged_text = cur_text + nxt_text[overlap:]
            else:
                merged_text = cur_text + " " + nxt_text
            current = dict(current)
            current["document"] = merged_text
            current["distance"] = min(current.get("distance", 1.0), nxt.get("distance", 1.0))

        elif consec_block:
            # ── Consecutive blocks on the same page: join with blank line ─
            current = dict(current)
            current["document"] = current["document"].rstrip() + "\n\n" + nxt["document"].lstrip()
            current["distance"] = min(current.get("distance", 1.0), nxt.get("distance", 1.0))

        else:
            merged.append(current)
            current = dict(nxt)

    merged.append(current)
    logger.debug("Chunk merge: %d chunks → %d merged passages", len(chunks), len(merged))
    return merged


# ── Node 5: generate ─────────────────────────────────────────────────────────


def generate_node(state: RAGState) -> dict[str, Any]:
    set_current_node("generate")
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

    # Merge adjacent / overlapping chunks before building the context prompt
    chunks = _merge_adjacent_chunks(chunks)

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
        page = chunk["metadata"].get("page_number") or chunk["metadata"].get("page", "")
        loc = f"{src}" + (f", Page {page}" if page else "")
        context_parts.append(f"[{i}] Source: {loc}\n{chunk['document']}")

    context = "\n\n---\n\n".join(context_parts)

    prompt = (
        f"Question: {question}\n\n"
        f"Context passages:\n{context}\n\n"
        "Answer the question comprehensively using the passages above."
    )

    answer = _stream_and_collect(
        llm_chat.stream_text(
            prompt=prompt,
            system_instruction=GENERATE_SYSTEM,
            temperature=0.4,
        )
    ).strip()

    # Strip any References section the LLM may append (we keep Sources)
    answer = re.split(r"\n+(?:References):\s*", answer, maxsplit=1, flags=re.IGNORECASE)[
        0
    ].strip()

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
    set_current_node("refine_query")
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
            system_instruction=REFINE_SYSTEM,
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


# ══════════════════════════════════════════════════════════════════════════════
# Agent tool nodes — classify intent and dispatch to the right tool
# ══════════════════════════════════════════════════════════════════════════════

# ── Node 0: classify_intent ──────────────────────────────────────────────────


def classify_intent_node(state: RAGState) -> dict[str, Any]:
    """Use Gemini native function calling to classify intent and pick a tool."""
    set_current_node("classify_intent")
    question = state["question"]

    tool_name = "general"
    tool_input = ""
    fn_name = ""

    try:
        tools = get_gemini_tool_declarations()
        result = llm_chat.call_tool(
            prompt=question,
            tools=tools,
            system_instruction=(
                "You are a document QA assistant with access to tools. "
                "The user has uploaded documents. For ANY question that "
                "could be answered from documents (including personal "
                "questions like 'what is my name', 'what is my salary', "
                "etc.), ALWAYS use search_documents. Only use "
                "direct_answer for greetings like 'hi' or 'hello'. "
                "Use get_current_datetime for date/time questions."
            ),
            temperature=0.0,
        )
        fn_name = result.get("name", "direct_answer")
        args = result.get("args", {})
        tool_name = FUNCTION_TO_TOOL.get(fn_name, "general")
        tool_input = (
            args.get("query", "")
            or args.get("location", "")
            or args.get("topic", "")
        )
    except Exception as exc:
        logger.warning("Native tool call failed, defaulting to general: %s", exc)

    logger.info("Intent classified (native): fn=%s → tool=%s input=%r", fn_name, tool_name, tool_input)
    return {
        "tool_name": tool_name,
        "tool_input": tool_input,
        "step": {
            "title": "Intent Classified",
            "description": f"Routing to: {tool_name} (via {fn_name})",
        },
    }


# ── Tool execution nodes ─────────────────────────────────────────────────────


def web_search_node(state: RAGState) -> dict[str, Any]:
    """Execute a web search and store the raw results."""
    set_current_node("web_search")
    question = state["question"]
    tool_input = state.get("tool_input", "") or question

    logger.info("Web search: %r", tool_input)
    result = web_search(tool_input, max_results=5)

    return {
        "tool_result": result,
        "step": {
            "title": "Web Search Complete",
            "description": f'Searched the web for: "{tool_input}"',
        },
    }


def weather_node(state: RAGState) -> dict[str, Any]:
    """Fetch weather for the extracted location."""
    set_current_node("weather")
    location = state.get("tool_input", "").strip()
    if not location:
        # Try to extract location from the question
        location = state["question"]

    logger.info("Weather lookup: %r", location)
    result = get_weather(location)

    return {
        "tool_result": result,
        "step": {
            "title": "Weather Retrieved",
            "description": f"Fetched weather for: {location}",
        },
    }


def datetime_node(state: RAGState) -> dict[str, Any]:
    """Return the current date and time."""
    set_current_node("datetime")
    result = get_current_datetime()

    return {
        "tool_result": result,
        "step": {
            "title": "Date/Time Retrieved",
            "description": "Fetched current date and time.",
        },
    }


def tool_answer_node(state: RAGState) -> dict[str, Any]:
    """Generate a natural-language answer from tool output using the LLM."""
    set_current_node("tool_answer")
    question = state["question"]
    tool_name = state.get("tool_name", "unknown")
    tool_result = state.get("tool_result", "")

    prompt = (
        f"User question: {question}\n\n"
        f"Tool used: {tool_name}\n"
        f"Tool output:\n{tool_result}\n\n"
        "Provide a helpful answer to the user based on the tool output above."
    )

    answer = _stream_and_collect(
        llm_chat.stream_text(
            prompt=prompt,
            system_instruction=TOOL_ANSWER_SYSTEM,
            temperature=0.4,
        )
    ).strip()

    logger.info("Tool answer generated: %d chars (tool=%s)", len(answer), tool_name)
    return {
        "answer": answer,
        "references": [f"Source: {tool_name}"],
        "step": {
            "title": "Answer Generated",
            "description": f"Synthesised answer from {tool_name} results.",
        },
    }


def direct_answer_node(state: RAGState) -> dict[str, Any]:
    """LLM answers general/casual questions directly without any tool."""
    set_current_node("direct_answer")
    question = state["question"]

    answer = _stream_and_collect(
        llm_chat.stream_text(
            prompt=question,
            system_instruction=DIRECT_ANSWER_SYSTEM,
            temperature=0.7,
        )
    ).strip()

    logger.info("Direct answer generated: %d chars", len(answer))
    return {
        "answer": answer,
        "references": [],
        "step": {
            "title": "Answer Generated",
            "description": "Answered directly without external tools.",
        },
    }
