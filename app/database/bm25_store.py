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
    user_id     TEXT                — tenant filter used during retrieval

Table ``documents``:
    document_key TEXT  PRIMARY KEY  — internal unique key (user_id + source_file)
    user_id      TEXT               — tenant owner of the document
    source_file  TEXT               — original filename / document identifier
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
    source_file TEXT NOT NULL DEFAULT '',
    user_id     TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_bm25_source ON bm25_corpus (source_file);

CREATE TABLE IF NOT EXISTS documents (
    document_key TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL DEFAULT '',
    source_file  TEXT NOT NULL,
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
                self._migrate_schema(conn)
                conn.commit()
            finally:
                conn.close()

    def _table_columns(self, conn: sqlite3.Connection, table_name: str) -> set[str]:
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        return {str(row["name"]) for row in rows}

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        bm25_columns = self._table_columns(conn, "bm25_corpus")
        if "user_id" not in bm25_columns:
            conn.execute(
                "ALTER TABLE bm25_corpus ADD COLUMN user_id TEXT NOT NULL DEFAULT ''"
            )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_bm25_user ON bm25_corpus (user_id)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_bm25_user_source ON bm25_corpus (user_id, source_file)"
        )

        document_columns = self._table_columns(conn, "documents")
        if "document_key" in document_columns and "user_id" in document_columns:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_user ON documents (user_id)")
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_user_source ON documents (user_id, source_file)"
            )
            return

        conn.execute("ALTER TABLE documents RENAME TO documents_legacy")
        conn.execute(
            "CREATE TABLE documents ("
            "document_key TEXT PRIMARY KEY, "
            "user_id TEXT NOT NULL DEFAULT '', "
            "source_file TEXT NOT NULL, "
            "content TEXT NOT NULL, "
            "metadata TEXT NOT NULL DEFAULT '{}', "
            "ingested_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))"
            ")"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_user ON documents (user_id)")
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_user_source ON documents (user_id, source_file)"
        )
        conn.execute(
            "INSERT INTO documents (document_key, user_id, source_file, content, metadata, ingested_at) "
            "SELECT source_file, '', source_file, content, metadata, ingested_at FROM documents_legacy"
        )
        conn.execute("DROP TABLE documents_legacy")

    def _document_key(self, user_id: str, source_file: str) -> str:
        return f"{user_id}:{source_file}"

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
                (c.get("metadata") or {}).get("user_id", ""),
            )
            for c in chunks
        ]
        with self._lock:
            conn = self._connect()
            try:
                conn.executemany(
                    "INSERT OR REPLACE INTO bm25_corpus "
                    "(chunk_id, document, metadata, source_file, user_id) VALUES (?, ?, ?, ?, ?)",
                    rows,
                )
                conn.commit()
                logger.debug("BM25 corpus: upserted %d chunk(s).", len(rows))
            finally:
                conn.close()

    def get_all(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """Return every corpus chunk as a dict: ``{id, document, metadata}``."""
        with self._lock:
            conn = self._connect()
            try:
                if user_id is None:
                    cur = conn.execute("SELECT chunk_id, document, metadata FROM bm25_corpus")
                else:
                    cur = conn.execute(
                        "SELECT chunk_id, document, metadata FROM bm25_corpus WHERE user_id = ?",
                        (user_id,),
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
            result.append({"id": row["chunk_id"], "document": row["document"], "metadata": meta})
        return result

    def delete_by_source(self, source_file: str, user_id: str | None = None) -> int:
        """Remove all chunks belonging to *source_file*. Returns the count removed."""
        with self._lock:
            conn = self._connect()
            try:
                if user_id is None:
                    cur = conn.execute(
                        "DELETE FROM bm25_corpus WHERE source_file = ?", (source_file,)
                    )
                else:
                    cur = conn.execute(
                        "DELETE FROM bm25_corpus WHERE source_file = ? AND user_id = ?",
                        (source_file, user_id),
                    )
                conn.commit()
                removed = cur.rowcount
            finally:
                conn.close()
        logger.debug(
            "BM25 corpus: removed %d chunk(s) for user=%r source=%r.",
            removed,
            user_id,
            source_file,
        )
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
        self,
        user_id: str,
        source_file: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Insert or replace a full document record.

        ``content`` should be the complete text of the document (e.g. all
        chunk texts joined in reading order).  Re-ingesting the same
        ``user_id`` + ``source_file`` pair overwrites the previous record and refreshes
        ``ingested_at``.
        """
        meta_json = json.dumps(metadata or {})
        document_key = self._document_key(user_id, source_file)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO documents "
                    "(document_key, user_id, source_file, content, metadata, ingested_at) "
                    "VALUES (?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))",
                    (document_key, user_id, source_file, content, meta_json),
                )
                conn.commit()
                logger.debug(
                    "Documents store: upserted user=%r source=%r (%d chars).",
                    user_id,
                    source_file,
                    len(content),
                )
            finally:
                conn.close()

    def get_all_documents(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """Return every stored document as a dict.

        Keys: ``user_id``, ``source_file``, ``content``, ``metadata``, ``ingested_at``.
        Suitable for building a document-level BM25 index when chunk-level
        granularity is not required.
        """
        with self._lock:
            conn = self._connect()
            try:
                if user_id is None:
                    cur = conn.execute(
                        "SELECT user_id, source_file, content, metadata, ingested_at FROM documents"
                    )
                else:
                    cur = conn.execute(
                        "SELECT user_id, source_file, content, metadata, ingested_at "
                        "FROM documents WHERE user_id = ?",
                        (user_id,),
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
                    "user_id": row["user_id"],
                    "source_file": row["source_file"],
                    "content": row["content"],
                    "metadata": meta,
                    "ingested_at": row["ingested_at"],
                }
            )
        return result

    def delete_document(self, user_id: str, source_file: str) -> int:
        """Remove a document record and all its chunks.

        Returns the total number of rows removed across both tables.
        """
        chunk_count = self.delete_by_source(source_file, user_id=user_id)
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "DELETE FROM documents WHERE user_id = ? AND source_file = ?",
                    (user_id, source_file),
                )
                conn.commit()
                doc_count = cur.rowcount
            finally:
                conn.close()
        logger.debug(
            "Documents store: removed user=%r source=%r (%d doc row, %d chunk rows).",
            user_id,
            source_file,
            doc_count,
            chunk_count,
        )
        return doc_count + chunk_count
