from abc import ABC, abstractmethod
from typing import Any


class BaseVectorDB(ABC):
    """Abstract interface for vector database providers."""

    @abstractmethod
    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Insert or update documents with their embeddings."""

    @abstractmethod
    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the top-n nearest documents to the query embedding."""

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Remove documents by their IDs."""

    @abstractmethod
    def health_check(self) -> tuple[bool, str]:
        """Return provider health status and an optional message."""
