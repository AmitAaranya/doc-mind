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

    for thread_id in list(checkpointer.storage.keys()):
        try:
            config = {"configurable": {"thread_id": thread_id}}
            snapshot = rag_graph.get_state(config)
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
