"""Query route — streaming Server-Sent Events RAG pipeline endpoint.

Every pipeline stage emits an SSE event so the client sees live progress.
The LLM answer is streamed token-by-token as it is generated.

Why thread + queue?
-------------------
`rag_graph.stream()` is a blocking sync call.  While a node's LLM call is in
progress the outer generator is frozen — it cannot `yield` a token until the
entire node returns.  A background thread lets the token callback fire tokens
into an asyncio queue the moment they arrive, so the client receives them in
real time, not batched at node completion.

SSE event shapes
----------------
  stage  : {"type":"stage",  "node":"<id>", "title":"<str>", "description":"<str>"}
  token  : {"type":"token",  "content":"<str>"}   ← every LLM token, every node, real-time
  result : {"type":"result", "answer":"...", "references":[...], "ragas_scores":{...},
             "optimized_query":"...", "keywords":[...], "iterations":N,
             "search_queries_tried":[...]}
  done   : {"type":"done"}
  error  : {"type":"error",  "detail":"<str>"}
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.rag import rag_graph
from app.rag.nodes import clear_token_callback, set_token_callback
from app.rag.state import StepStatus

query_route = APIRouter(prefix="/query", tags=["query"])
logger = get_logger(__name__)


# ── Request model ─────────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    user_id: str = Field(
        ..., min_length=1, description="Raw user identifier used for tenant filtering"
    )
    question: str = Field(
        ..., min_length=3, description="Question to answer from uploaded documents"
    )
    max_iterations: int = Field(
        default=3, ge=1, le=5, description="Max retrieval-refinement loop iterations (1-5)"
    )


# ── SSE helper ────────────────────────────────────────────────────────────────


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


# ── Core streaming generator (thread + asyncio.Queue) ─────────────────────────
# rag_graph.stream() is a BLOCKING call running in a background thread.
# The token callback fires loop.call_soon_threadsafe → queue.put_nowait for
# EVERY token the instant it is produced — zero buffering, zero bottleneck.
# The async generator simply drains the queue and yields each SSE frame live.
#
#   LLM produces token → callback → queue (thread-safe) → yield SSE  (real-time)


async def _stream_rag(question: str, user_id: str, max_iterations: int) -> AsyncGenerator[str, None]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Any] = asyncio.Queue()
    _SENTINEL = object()

    def _put(event: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def _run() -> None:
        # Each token is pushed to the queue the instant the LLM yields it
        set_token_callback(lambda tok: _put({"type": "token", "content": tok}))
        accumulated: dict[str, Any] = {}
        try:
            for chunk in rag_graph.stream(
                {"question": question, "user_id": user_id, "max_iterations": max_iterations},
                stream_mode="updates",
            ):
                for node_name, node_output in chunk.items():
                    if node_name.startswith("__"):
                        continue

                    if node_name == "generate":
                        accumulated.update({k: v for k, v in node_output.items() if k != "answer"})
                        accumulated["answer"] = node_output.get("answer", "")
                    else:
                        accumulated.update(node_output)

                    # Stage event arrives AFTER all tokens for this node are already in the queue
                    step: StepStatus = node_output.get("step", {})
                    _put(
                        {
                            "type": "stage",
                            "node": node_name,
                            "title": step.get("title", node_name.replace("_", " ").title()),
                            "description": step.get("description", ""),
                        }
                    )

            _put(
                {
                    "type": "result",
                    "answer": accumulated.get("answer", ""),
                    "references": accumulated.get("references", []),
                    "ragas_scores": accumulated.get("ragas_scores", {}),
                    "optimized_query": accumulated.get("optimized_query", ""),
                    "keywords": accumulated.get("keywords", []),
                    "iterations": accumulated.get("iteration", 0),
                    "search_queries_tried": accumulated.get("search_queries_tried", []),
                }
            )

        except Exception as exc:
            logger.exception("RAG pipeline error: %s", exc)
            _put({"type": "error", "detail": str(exc)})
        finally:
            clear_token_callback()
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    try:
        while True:
            item = await queue.get()
            if item is _SENTINEL:
                break
            yield _sse(item)
        yield _sse({"type": "done"})
    except asyncio.CancelledError:
        pass  # client disconnected
    finally:
        thread.join(timeout=5)


# ── Endpoint ──────────────────────────────────────────────────────────────────


@query_route.post("", summary="Stream RAG pipeline via Server-Sent Events")
async def query_documents(request: QueryRequest) -> StreamingResponse:
    """
    Run the full LangGraph RAG pipeline and stream every stage as an SSE event.

    **Event stream protocol** (`text/event-stream`):

    | `type`   | Fields                                             | When                         |
    |----------|----------------------------------------------------|------------------------------|
    | `stage`  | `node`, `title`, `description`                     | After each pipeline node     |
    | `token`  | `content`                                          | Each LLM token while writing |
    | `result` | `answer`, `references`, `ragas_scores`, `keywords` | Pipeline complete            |
    | `done`   | —                                                  | Stream closed                |
    | `error`  | `detail`                                           | On failure                   |

    **JavaScript example:**
    ```js
    const es = new EventSource('/query');
    es.onmessage = e => {
      const evt = JSON.parse(e.data);
      if (evt.type === 'token') appendToken(evt.content);
      if (evt.type === 'stage') showStage(evt.title, evt.description);
      if (evt.type === 'result') showFinalResult(evt);
      if (evt.type === 'done') es.close();
    };
    ```
    """
    user_id = request.user_id.strip()
    if not user_id:
        raise HTTPException(status_code=422, detail="user_id must not be blank.")

    logger.info(
        "SSE query: user_id=%r question=%r (max_iter=%d)",
        user_id,
        request.question,
        request.max_iterations,
    )
    return StreamingResponse(
        _stream_rag(request.question, user_id, request.max_iterations),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
