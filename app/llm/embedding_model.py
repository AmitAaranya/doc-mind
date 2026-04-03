from collections.abc import Sequence
from importlib import import_module

from app.core.logging import get_logger
from app.llm.base import BaseEmbeddings

logger = get_logger(__name__)


class FastEmbedEmbeddings(BaseEmbeddings):
	"""Lightweight open-source embeddings using FastEmbed (ONNX runtime)."""

	def __init__(self, model_name: str | None = None) -> None:
		self.model_name = model_name or "BAAI/bge-small-en-v1.5"
		self._engine = None
		self._init_error: str | None = None

	@property
	def engine(self):
		if self._engine is None:
			try:
				fastembed_module = import_module("fastembed")
				text_embedding_cls = fastembed_module.TextEmbedding
			except Exception as exc:  # pragma: no cover - depends on environment
				self._init_error = str(exc)
				raise RuntimeError(
					"fastembed is not installed. Add `fastembed` to dependencies."
				) from exc

			self._engine = text_embedding_cls(model_name=self.model_name)
		return self._engine

	def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
		if not texts:
			return []

		vectors = self.engine.embed(list(texts))
		return [vector.tolist() for vector in vectors]

	def health_check(self) -> tuple[bool, str]:
		try:
			vectors = self.engine.embed(["health-check"])
			first_vector = next(iter(vectors), None)
			if first_vector is None:
				return False, "Empty response from embeddings model"

			return True, f"ok(dim={len(first_vector)})"
		except Exception as exc:
			logger.warning("Embeddings health check failed: %s", exc)
			return False, str(exc)
