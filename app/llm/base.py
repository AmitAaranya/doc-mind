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

    def describe_image(self, image_b64: str, mime_type: str = "image/png") -> str:
        """Generate a text description of a base64-encoded image.

        Default implementation raises NotImplementedError.  Subclasses that
        support multimodal input should override this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support image description. "
            "Override describe_image() in your LLM subclass."
        )

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
