from abc import ABC, abstractmethod
from collections.abc import Sequence


class BaseLLM(ABC):
    """Abstract interface for chat/completion LLM providers."""

    @abstractmethod
    def stream_text(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 1,
    ) -> Sequence[str]:
        """Return streamed chunks for a prompt."""

    @abstractmethod
    def health_check(self) -> tuple[bool, str]:
        """Return provider health status and an optional message."""


class BaseEmbeddings(ABC):
    """Abstract interface for embedding model providers."""

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Create embeddings for multiple document strings."""

    @abstractmethod
    def health_check(self) -> tuple[bool, str]:
        """Return provider health status and an optional message."""
