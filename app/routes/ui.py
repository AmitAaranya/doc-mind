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
