"""RAGAS evaluation node for the RAG pipeline.

This node runs AFTER generation and is a terminal step — it does NOT influence
the retrieval loop.  Its sole purpose is:

  1. Compute RAGAS reference-free metrics against the final answer.
  2. Log the scores.
  3. Append a record to ``rag_evaluations.jsonl`` in the project root so results
     accumulate across requests for offline analysis.

Metrics
-------
  • faithfulness               – answer claims grounded in retrieved context
  • answer_relevancy            – answer addresses the question
  • context_precision           – retrieved passages are relevant to the question
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.logging import get_logger
from app.rag.state import RAGState

logger = get_logger(__name__)

EVAL_LOG_PATH = Path("rag_evaluations.jsonl")


# ── LangChain adapters (avoid extra google-specific langchain packages) ────────


def _make_langchain_llm():
    """Return a minimal BaseChatModel wrapping our GoogleLLMModel."""
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    from app.llm import llm_chat as _llm

    class _Adapter(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "doc-mind-google-genai"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
            prompt = "\n".join(str(m.content) for m in messages)
            text = "".join(chunk for chunk in _llm.stream_text(prompt=prompt, temperature=0) if chunk)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    return _Adapter()


def _make_langchain_embeddings():
    """Return a minimal Embeddings wrapper around our FastEmbedEmbeddings."""
    from langchain_core.embeddings import Embeddings

    from app.llm import embeddings as _emb

    class _EmbAdapter(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return _emb.embed_documents(texts)

        def embed_query(self, text: str) -> list[float]:
            return _emb.embed_documents([text])[0]

    return _EmbAdapter()


# ── RAGAS evaluation ──────────────────────────────────────────────────────────


def _run_ragas(
    question: str,
    answer: str,
    contexts: list[str],
) -> dict[str, float]:
    """Call RAGAS metrics and return a dict of metric_name -> float score."""
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithoutReference,
        Faithfulness,
    )

    ragas_llm = LangchainLLMWrapper(_make_langchain_llm())
    ragas_emb = LangchainEmbeddingsWrapper(_make_langchain_embeddings())

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
    )
    dataset = EvaluationDataset(samples=[sample])

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
        ContextPrecisionWithoutReference(llm=ragas_llm),
    ]

    result = evaluate(dataset=dataset, metrics=metrics)

    # Newer RAGAS returns a Results object; .scores is a list[dict]
    raw_scores: dict[str, Any] = {}
    try:
        df = result.to_pandas()
        raw_scores = df.iloc[0].to_dict()
    except Exception:
        if hasattr(result, "scores") and result.scores:
            raw_scores = result.scores[0]

    return {
        k: float(v)
        for k, v in raw_scores.items()
        if isinstance(v, (int, float)) and not k.startswith("_")
    }


# ── evaluate_node (terminal — log + store only) ───────────────────────────────


def evaluate_node(state: RAGState) -> dict[str, Any]:
    """Compute RAGAS metrics, log them, and persist to ``rag_evaluations.jsonl``.

    This node does NOT affect the retrieval loop.  It is the final step of the
    graph, always followed by END.
    """
    question = state["question"]
    answer = state.get("answer", "")
    chunks = state.get("retrieved_chunks", [])
    iteration = state.get("iteration", 0)
    references = state.get("references", [])
    queries_tried = state.get("search_queries_tried", [])

    contexts = [c["document"] for c in chunks]

    scores: dict[str, float] = {}
    if answer and contexts:
        try:
            scores = _run_ragas(question, answer, contexts)
        except Exception as exc:
            logger.warning("RAGAS evaluation error: %s", exc)

    avg = sum(scores.values()) / len(scores) if scores else 0.0

    logger.info(
        "RAGAS evaluation — iterations=%d  avg_score=%.3f  scores=%s",
        iteration, avg, scores,
    )

    # ── Persist to local JSONL ────────────────────────────────────────────────
    record = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "question": question,
        "answer_length": len(answer),
        "iterations": iteration,
        "search_queries_tried": queries_tried,
        "references": references,
        "ragas_scores": scores,
        "avg_ragas_score": round(avg, 4),
    }
    try:
        with EVAL_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.debug("Evaluation record appended to %s", EVAL_LOG_PATH)
    except Exception as exc:
        logger.warning("Could not write evaluation log: %s", exc)

    if scores:
        score_parts = [f"{k.replace('_', ' ').title()}: {v:.2f}" for k, v in sorted(scores.items())]
        eval_desc = f"avg {avg:.2f} — {', '.join(score_parts)}"
    else:
        eval_desc = "RAGAS scores unavailable (LLM or data error)."

    return {
        "ragas_scores": scores,
        "step": {
            "title": "Quality Evaluated",
            "description": eval_desc,
        },
    }
