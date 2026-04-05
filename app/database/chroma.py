from importlib import import_module
from typing import Any

from chromadb import get_settings

from app.core import SETTING
from app.core.logging import get_logger
from app.database.base import BaseVectorDB

logger = get_logger(__name__)


class ChromaVectorStore(BaseVectorDB):
    """Persistent vector store backed by ChromaDB (PersistentClient)."""

    def __init__(self, collection_name: str = "qa-assistant", path: str | None = None) -> None:
        self.collection_name = collection_name
        self.path = path or SETTING.CHROMA_PATH
        self._client = None
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            try:
                chromadb = import_module("chromadb")
            except Exception as exc:
                raise RuntimeError(
                    "chromadb is not installed. Add `chromadb` to dependencies."
                ) from exc

            self._client = chromadb.PersistentClient(path=self.path)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        raw = self.collection.query(**kwargs)

        results: list[dict[str, Any]] = []
        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for doc_id, doc, meta, dist in zip(ids, docs, metas, distances):
            results.append(
                {
                    "id": doc_id,
                    "document": doc,
                    "metadata": meta or {},
                    "distance": dist,
                }
            )
        return results

    def delete(self, ids: list[str]) -> None:
        self.collection.delete(ids=ids)

    def health_check(self) -> tuple[bool, str]:
        try:
            count = self.collection.count()
            return True, f"ok(docs={count})"
        except Exception as exc:
            logger.warning("ChromaVectorStore health check failed: %s", exc)
            return False, str(exc)
