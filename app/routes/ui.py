"""Server-rendered UI for upload/query/stream and corpus inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from app.database.bm25_store import BM25CorpusStore

ui_route = APIRouter(tags=["ui"])

_bm25_store = BM25CorpusStore()
_DUMMY_USER_ID = "dummy-user"
_UI_HTML_PATH = Path(__file__).resolve().parents[1] / "templates" / "ui.html"


def _as_str_map(meta: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in (meta or {}).items():
        out[str(k)] = "" if v is None else str(v)
    return out


@ui_route.get("/app", response_class=HTMLResponse, summary="QA Assistant web UI")
async def ui_page() -> str:
    """Serve a single-page UI for upload/query/stream/data inspection."""
    return _UI_HTML_PATH.read_text(encoding="utf-8")


@ui_route.get("/ui/data", summary="Get stored docs and chunks for UI")
async def ui_data() -> dict[str, Any]:
    """Return all stored document/chunk data for the fixed dummy user."""
    docs = _bm25_store.get_all_documents(user_id=_DUMMY_USER_ID)
    chunks = _bm25_store.get_all(user_id=_DUMMY_USER_ID)

    docs_out = [
        {
            "user_id": str(d.get("user_id", "")),
            "source_file": str(d.get("source_file", "")),
            "content": str(d.get("content", "")),
            "content_length": len(str(d.get("content", ""))),
            "metadata": _as_str_map(d.get("metadata", {})),
            "ingested_at": str(d.get("ingested_at", "")),
        }
        for d in docs
    ]

    chunks_out = [
        {
            "id": str(c.get("id", "")),
            "document": str(c.get("document", "")),
            "document_length": len(str(c.get("document", ""))),
            "metadata": _as_str_map(c.get("metadata", {})),
        }
        for c in chunks
    ]

    return {
        "user_id": _DUMMY_USER_ID,
        "documents": docs_out,
        "chunks": chunks_out,
    }


@ui_route.get("/ui/history", summary="List all previous pipeline run threads (MemorySaver)")
async def ui_history() -> dict[str, Any]:
    """Return every thread stored in the in-memory checkpointer, newest first.

    Each entry contains the final state snapshot values so the UI can display
    the question, answer, tool used, status, and timestamp without needing to
    replay the graph.
    """
    from app.rag import rag_graph  # local import to avoid circular deps at module load

    checkpointer = rag_graph.checkpointer
    threads: list[dict[str, Any]] = []

    for thread_id in list(checkpointer.storage.keys()):  # type: ignore[union-attr]
        try:
            config = {"configurable": {"thread_id": thread_id}}
            snapshot = rag_graph.get_state(config)  # type: ignore[arg-type]
            if snapshot is None:
                continue
            values = snapshot.values
            is_interrupted = bool(snapshot.next)
            threads.append(
                {
                    "thread_id": thread_id,
                    "question": str(values.get("question", "")),
                    "answer": str(values.get("answer", "")),
                    "tool_name": str(values.get("tool_name", "rag")),
                    "keywords": values.get("keywords", []),
                    "references": values.get("references", []),
                    "iteration": int(values.get("iteration", 0)),
                    "status": "interrupted" if is_interrupted else "completed",
                    "next_nodes": list(snapshot.next),
                    "clarification_question": (
                        str(values.get("clarification_question", "")) if is_interrupted else ""
                    ),
                    "created_at": str(snapshot.created_at or ""),
                }
            )
        except Exception:
            continue

    # Newest first (ISO timestamps sort lexicographically)
    threads.sort(key=lambda t: t["created_at"], reverse=True)
    return {"threads": threads}


@ui_route.get(
    "/ui/history/{thread_id}",
    summary="Get full step-by-step timeline for a single pipeline run",
)
async def ui_history_detail(thread_id: str) -> dict[str, Any]:
    """Replay all checkpoints for *thread_id* and return an ordered list of steps.

    Each step is one pipeline node that ran, with its title, description, and
    timestamp — exactly the data needed to render the timeline in the UI.
    """
    from app.rag import rag_graph  # local import

    config = {"configurable": {"thread_id": thread_id}}

    # get_state_history yields newest-first; we reverse to get oldest-first.
    try:
        history = list(reversed(list(rag_graph.get_state_history(config))))  # type: ignore[arg-type]
    except Exception as exc:
        return {"thread_id": thread_id, "steps": [], "error": str(exc)}

    steps: list[dict[str, Any]] = []
    final_values: dict[str, Any] = {}

    for i, snapshot in enumerate(history):
        source = (snapshot.metadata or {}).get("source", "")
        values = snapshot.values or {}

        # Track final state for the summary at top
        if values:
            final_values = values

        # "input" = before any node ran; "loop" = after a node ran
        if source != "loop":
            continue

        step_val = values.get("step") or {}
        title = str(step_val.get("title", ""))
        description = str(step_val.get("description", ""))

        # Infer which node(s) ran: the previous snapshot's .next gives us that
        prev = history[i - 1] if i > 0 else None
        node_names: list[str] = list(prev.next) if prev else []

        # Skip empty / duplicate steps (parallel nodes share the same step slot)
        if not title:
            continue
        # Deduplicate consecutive identical steps produced by parallel branches
        if steps and steps[-1]["title"] == title and steps[-1]["description"] == description:
            if node_names:
                # Merge the parallel node name in
                existing = steps[-1]["node"]
                if isinstance(existing, list):
                    for n in node_names:
                        if n not in existing:
                            existing.append(n)
                elif node_names[0] != existing:
                    steps[-1]["node"] = [existing, *node_names]
            continue

        steps.append(
            {
                "node": node_names[0] if len(node_names) == 1 else node_names,
                "title": title,
                "description": description,
                "ts": str(snapshot.created_at or ""),
            }
        )

    return {
        "thread_id": thread_id,
        "question": str(final_values.get("question", "")),
        "answer": str(final_values.get("answer", "")),
        "references": final_values.get("references", []),
        "tool_name": str(final_values.get("tool_name", "rag")),
        "steps": steps,
    }
