"""Document chunker — converts an OrderedPDFDocument into Chunk objects.

Chunking strategies per block type
------------------------------------
- TextBlock  → recursive character split with overlap (no extra deps)
- TableBlock → convert to Markdown → one atomic Chunk (never split mid-row)
- ImageBlock → call LLM for a text description → split that description

Public API
----------
chunker = DocumentChunker()                        # uses SETTING defaults
chunks  = await chunker.chunk_document(doc, llm, source="report.pdf")

Individual helpers are also public:
  chunker.chunk_text_block(block, source)          -> list[Chunk]
  chunker.chunk_table_block(block, source)         -> list[Chunk]
  await chunker.chunk_image_block(block, llm, src) -> list[Chunk]
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum
from uuid import uuid4

from app.core import SETTING
from app.llm.base import BaseLLM
from app.utils.pdf_processor import (
    BlockType,
    ImageBlock,
    OrderedPDFDocument,
    TableBlock,
    TextBlock,
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

_EXT_TO_MIME: dict[str, str] = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "bmp": "image/bmp",
    "tiff": "image/tiff",
}

_IMAGE_SYSTEM_PROMPT = (
    "You are a document analysis assistant. "
    "Describe PDF images for use in a retrieval-augmented generation (RAG) system."
)

_IMAGE_USER_PROMPT = (
    "Describe this image extracted from a PDF document. "
    "Include: visual type (chart, photo, diagram, screenshot, etc.), "
    "key data points, all visible text and labels, axes or legend values if present, "
    "and any contextual information that would help answer questions about this document. "
    "Be factual and specific."
)


class ChunkType(StrEnum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


@dataclass
class Chunk:
    """A single embeddable unit of content produced by the chunker."""

    id: str  # uuid4 string — stable ID for upsert into vector DB
    content: str  # text to embed and store
    chunk_type: ChunkType
    metadata: dict  # carries provenance for retrieval filtering


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


class DocumentChunker:
    """Convert an OrderedPDFDocument into a flat list of Chunk objects.

    Parameters
    ----------
    chunk_size:
        Maximum number of characters per text chunk (~512 tokens at 4 chars/token
        for BAAI/bge-small-en-v1.5).  Defaults to SETTING.CHUNK_SIZE.
    chunk_overlap:
        Characters from the end of the previous chunk prepended to the next,
        preserving sentence context across boundaries.  Defaults to SETTING.CHUNK_OVERLAP.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or SETTING.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or SETTING.CHUNK_OVERLAP

    # ------------------------------------------------------------------
    # Public: per-block chunkers
    # ------------------------------------------------------------------

    def chunk_text_block(self, block: TextBlock, source: str) -> list[Chunk]:
        """Split a TextBlock into overlapping character chunks."""
        splits = self._split_text(block.content)
        return [
            Chunk(
                id=str(uuid4()),
                content=text,
                chunk_type=ChunkType.TEXT,
                metadata={
                    "source_file": source,
                    "page_number": block.page_number,
                    "block_order": block.order,
                    "bbox": block.bbox,
                    "block_type": BlockType.TEXT.value,
                    "char_start": char_start,
                },
            )
            for text, char_start in splits
            if text.strip()
        ]

    def chunk_table_block(self, block: TableBlock, source: str) -> list[Chunk]:
        """Convert a TableBlock to Markdown and return it as one atomic Chunk."""
        md = block.to_markdown()
        if not md.strip():
            return []
        return [
            Chunk(
                id=str(uuid4()),
                content=md,
                chunk_type=ChunkType.TABLE,
                metadata={
                    "source_file": source,
                    "page_number": block.page_number,
                    "block_order": block.order,
                    "bbox": block.bbox,
                    "block_type": BlockType.TABLE.value,
                    "headers": block.table.headers,
                    "row_count": len(block.table.rows),
                },
            )
        ]

    async def chunk_image_block(
        self,
        block: ImageBlock,
        llm: BaseLLM,
        source: str,
    ) -> list[Chunk]:
        """Ask the LLM to describe the image then split the description into chunks."""
        mime = _EXT_TO_MIME.get(block.image.extension.lower(), "image/png")
        description: str = await asyncio.to_thread(
            llm.describe_image,
            block.image.data_b64,
            mime,
        )
        if not description.strip():
            return []

        splits = self._split_text(description)
        return [
            Chunk(
                id=str(uuid4()),
                content=text,
                chunk_type=ChunkType.IMAGE,
                metadata={
                    "source_file": source,
                    "page_number": block.page_number,
                    "block_order": block.order,
                    "bbox": block.bbox,
                    "block_type": BlockType.IMAGE.value,
                    "width": block.image.width,
                    "height": block.image.height,
                    "image_extension": block.image.extension,
                    "image_index": block.image.index,
                    "char_start": char_start,
                },
            )
            for text, char_start in splits
            if text.strip()
        ]

    # ------------------------------------------------------------------
    # Public: full-document chunker
    # ------------------------------------------------------------------

    async def chunk_document(
        self,
        doc: OrderedPDFDocument,
        llm: BaseLLM,
        source: str,
    ) -> list[Chunk]:
        """Chunk every block in the document, maintaining page + reading order.

        Text and table blocks are processed synchronously (fast).
        Image blocks call the LLM; they are gathered concurrently per page to
        reduce latency while staying within a single document scope.

        Parameters
        ----------
        doc:
            The result of ``PDFProcessor.extract_ordered()``.
        llm:
            A ``BaseLLM`` instance that implements ``describe_image()``.
        source:
            A human-readable source identifier stored in chunk metadata
            (e.g. original filename or document ID).
        """
        all_chunks: list[Chunk] = []

        for page in doc.pages:
            # Collect image coroutines so we can gather them concurrently
            image_tasks: list[tuple[int, asyncio.Task[list[Chunk]]]] = []

            # Process non-image blocks immediately; schedule image tasks
            page_text_table_chunks: list[tuple[int, list[Chunk]]] = []
            for block in page.blocks:
                if isinstance(block, TextBlock):
                    page_text_table_chunks.append(
                        (block.order, self.chunk_text_block(block, source))
                    )
                elif isinstance(block, TableBlock):
                    page_text_table_chunks.append(
                        (block.order, self.chunk_table_block(block, source))
                    )
                elif isinstance(block, ImageBlock):
                    task = asyncio.create_task(self.chunk_image_block(block, llm, source))
                    image_tasks.append((block.order, task))

            # Await all image tasks for this page concurrently
            image_results: list[tuple[int, list[Chunk]]] = []
            if image_tasks:
                orders = [o for o, _ in image_tasks]
                results = await asyncio.gather(*[t for _, t in image_tasks])
                image_results = list(zip(orders, results))

            # Merge text/table + image chunks, restore reading order
            combined = page_text_table_chunks + image_results
            combined.sort(key=lambda x: x[0])
            for _, chunks in combined:
                all_chunks.extend(chunks)

        return all_chunks

    # ------------------------------------------------------------------
    # Private: text splitting
    # ------------------------------------------------------------------

    def _split_text(self, text: str) -> list[tuple[str, int]]:
        """Split *text* into ``(chunk_text, char_start)`` pairs.

        Uses a greedy backwards-search for clean split points
        (``\\n\\n`` > ``\\n`` > ``. `` > `` ``).  Overlap is applied by
        rewinding the start pointer by ``chunk_overlap`` characters so the
        next chunk begins with the tail of the previous one.
        """
        if len(text) <= self.chunk_size:
            return [(text, 0)]

        chunks: list[tuple[str, int]] = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            if end >= len(text):
                chunks.append((text[start:], start))
                break

            # Walk backwards from `end` to find a clean separator
            split_at = end
            for sep in ("\n\n", "\n", ". ", " "):
                pos = text.rfind(sep, start, end)
                if pos > start:
                    split_at = pos + len(sep)
                    break

            chunks.append((text[start:split_at], start))
            # Overlap: next chunk starts `chunk_overlap` chars before split_at
            # max(start + 1, ...) guarantees forward progress
            start = max(start + 1, split_at - self.chunk_overlap)

        return chunks
