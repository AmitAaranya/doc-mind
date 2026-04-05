"""SQLite-backed BM25 corpus store.

Persists chunk texts and metadata so that ``BM25Retriever`` can be built at
query-time over the full document corpus.  The corpus is updated incrementally
whenever new documents are ingested — existing chunks for the same ``chunk_id``
are replaced (upsert semantics), so re-ingesting a document is safe.

Schema
------
Table ``bm25_corpus``:
  chunk_id    TEXT  PRIMARY KEY   — matches the ChromaDB chunk ID
  document    TEXT                — raw chunk text
  metadata    TEXT                — JSON-serialised metadata dict
  source_file TEXT                — fast per-document filter / deletion

Table ``documents``:
  source_file  TEXT  PRIMARY KEY  — original filename / document identifier
  content      TEXT               — full document text (all chunks joined)
  metadata     TEXT               — JSON-serialised document-level metadata
  ingested_at  TEXT               — ISO-8601 timestamp (UTC, set by SQLite)
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from app.core import SETTING
from app.core.logging import get_logger

logger = get_logger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS bm25_corpus (
    chunk_id    TEXT PRIMARY KEY,
    document    TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}',
    source_file TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_bm25_source ON bm25_corpus (source_file);

CREATE TABLE IF NOT EXISTS documents (
    source_file TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}',
    ingested_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
"""


class BM25CorpusStore:
    """Thread-safe SQLite store for the BM25 text corpus."""

    def __init__(self, db_path: str | None = None) -> None:
        self._path = db_path or str(Path(SETTING.CHROMA_PATH) / "bm25_corpus.db")
        self._lock = threading.Lock()
        self._ensure_schema()

    # ── Internal ───────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(_DDL)
                conn.commit()
            finally:
                conn.close()

    # ── Public API ─────────────────────────────────────────────────────────

    def upsert(self, chunks: list[dict[str, Any]]) -> None:
        """Insert or replace chunks into the corpus.

        Each chunk dict must have keys ``id``, ``document``, and optionally
        ``metadata`` (a dict).  Existing rows with the same ``chunk_id`` are
        replaced, so re-ingesting a document is idempotent.
        """
        if not chunks:
            return
        rows = [
            (
                c["id"],
                c["document"],
                json.dumps(c.get("metadata") or {}),
                (c.get("metadata") or {}).get("source_file", ""),
            )
            for c in chunks
        ]
        with self._lock:
            conn = self._connect()
            try:
                conn.executemany(
                    "INSERT OR REPLACE INTO bm25_corpus "
                    "(chunk_id, document, metadata, source_file) VALUES (?, ?, ?, ?)",
                    rows,
                )
                conn.commit()
                logger.debug("BM25 corpus: upserted %d chunk(s).", len(rows))
            finally:
                conn.close()

    def get_all(self) -> list[dict[str, Any]]:
        """Return every corpus chunk as a dict: ``{id, document, metadata}``."""
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute("SELECT chunk_id, document, metadata FROM bm25_corpus")
                rows = cur.fetchall()
            finally:
                conn.close()
        result: list[dict[str, Any]] = []
        for row in rows:
            try:
                meta = json.loads(row["metadata"])
            except Exception:
                meta = {}
            result.append({"id": row["chunk_id"], "document": row["document"], "metadata": meta})
        return result

    def delete_by_source(self, source_file: str) -> int:
        """Remove all chunks belonging to *source_file*. Returns the count removed."""
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "DELETE FROM bm25_corpus WHERE source_file = ?", (source_file,)
                )
                conn.commit()
                removed = cur.rowcount
            finally:
                conn.close()
        logger.debug("BM25 corpus: removed %d chunk(s) for %r.", removed, source_file)
        return removed

    @property
    def size(self) -> int:
        """Total number of chunks currently stored in the corpus."""
        with self._lock:
            conn = self._connect()
            try:
                (n,) = conn.execute("SELECT COUNT(*) FROM bm25_corpus").fetchone()
            finally:
                conn.close()
        return n

    # ── Full-document store ────────────────────────────────────────────────

    def upsert_document(
        self, source_file: str, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Insert or replace a full document record.

        ``content`` should be the complete text of the document (e.g. all
        chunk texts joined in reading order).  Re-ingesting the same
        ``source_file`` overwrites the previous record and refreshes
        ``ingested_at``.
        """
        meta_json = json.dumps(metadata or {})
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO documents "
                    "(source_file, content, metadata, ingested_at) "
                    "VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))",
                    (source_file, content, meta_json),
                )
                conn.commit()
                logger.debug("Documents store: upserted %r (%d chars).", source_file, len(content))
            finally:
                conn.close()

    def get_all_documents(self) -> list[dict[str, Any]]:
        """Return every stored document as a dict.

        Keys: ``source_file``, ``content``, ``metadata``, ``ingested_at``.
        Suitable for building a document-level BM25 index when chunk-level
        granularity is not required.
        """
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "SELECT source_file, content, metadata, ingested_at FROM documents"
                )
                rows = cur.fetchall()
            finally:
                conn.close()
        result: list[dict[str, Any]] = []
        for row in rows:
            try:
                meta = json.loads(row["metadata"])
            except Exception:
                meta = {}
            result.append(
                {
                    "source_file": row["source_file"],
                    "content": row["content"],
                    "metadata": meta,
                    "ingested_at": row["ingested_at"],
                }
            )
        return result

    def delete_document(self, source_file: str) -> int:
        """Remove a document record and all its chunks.

        Returns the total number of rows removed across both tables.
        """
        chunk_count = self.delete_by_source(source_file)
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "DELETE FROM documents WHERE source_file = ?", (source_file,)
                )
                conn.commit()
                doc_count = cur.rowcount
            finally:
                conn.close()
        logger.debug(
            "Documents store: removed %r (%d doc row, %d chunk rows).",
            source_file,
            doc_count,
            chunk_count,
        )
        return doc_count + chunk_count
