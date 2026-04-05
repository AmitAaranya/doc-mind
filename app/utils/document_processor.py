"""Document processor — extracts text and tables from multiple file formats.

Supported formats
-----------------
- .pdf   → delegates to PDFProcessor (text + images + tables)
- .docx  → delegates to DocxProcessor (text + tables)
- .doc   → delegates to DocxProcessor (text + tables)
- .md    → raw markdown text (text only)
- .txt   → plain text (text only)

All formats return an ``OrderedPDFDocument`` so the chunker pipeline is
unchanged.  Non-PDF formats produce only ``TextBlock`` and ``TableBlock``
objects (no ``ImageBlock``).

Public API
----------
processor = DocumentProcessor("report.docx")
doc = processor.extract_ordered()   # -> OrderedPDFDocument
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from app.utils.docx_processor import DocxProcessor
from app.utils.pdf_processor import (
    BlockType,
    OrderedPageData,
    OrderedPDFDocument,
    PDFProcessor,
    TextBlock,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx", ".doc", ".md", ".txt"})

# Maximum characters per "virtual page" for flat-text formats (md / txt).
# Keeps chunk sizes manageable without introducing real page boundaries.
_TEXT_PAGE_SIZE = 4_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_into_pages(text: str, page_size: int = _TEXT_PAGE_SIZE) -> list[str]:
    """Split *text* into chunks of at most *page_size* characters, preferring
    paragraph boundaries (double-newline) when possible."""
    if not text.strip():
        return []

    paragraphs = text.split("\n\n")
    pages: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2  # +2 for the "\n\n" separator
        if current_len + para_len > page_size and current:
            pages.append("\n\n".join(current).strip())
            current = []
            current_len = 0
        # If a single paragraph exceeds page_size, hard-wrap it
        if para_len > page_size:
            for chunk in textwrap.wrap(para, page_size, break_long_words=True):
                pages.append(chunk)
        else:
            current.append(para)
            current_len += para_len

    if current:
        pages.append("\n\n".join(current).strip())

    return [p for p in pages if p]


def _pages_to_ordered_doc(
    file_path: str,
    pages: list[str],
    metadata: dict,
) -> OrderedPDFDocument:
    """Wrap a list of page-text strings into an ``OrderedPDFDocument``."""
    ordered_pages: list[OrderedPageData] = []
    for i, page_text in enumerate(pages):
        pn = i + 1
        block = TextBlock(
            block_type=BlockType.TEXT,
            page_number=pn,
            order=0,
            bbox=(0.0, 0.0, 0.0, 0.0),
            content=page_text,
        )
        ordered_pages.append(OrderedPageData(page_number=pn, blocks=[block]))

    return OrderedPDFDocument(
        file_path=file_path,
        total_pages=len(ordered_pages),
        metadata=metadata,
        pages=ordered_pages,
    )


# ---------------------------------------------------------------------------
# Per-format extraction
# ---------------------------------------------------------------------------


def _extract_text_file(path: Path) -> OrderedPDFDocument:
    """Extract plain text from .txt or .md files."""
    text = path.read_text(encoding="utf-8", errors="replace")
    pages = _split_into_pages(text)
    metadata = {"format": path.suffix.lstrip(".")}
    return _pages_to_ordered_doc(str(path), pages, metadata)


# ---------------------------------------------------------------------------
# Public processor
# ---------------------------------------------------------------------------


class DocumentProcessor:
    """Unified document processor for PDF, DOCX, Markdown, and plain-text files.

    Parameters
    ----------
    file_path:
        Path to the document to process.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is not supported.
    """

    def __init__(self, file_path: str | Path) -> None:
        self._path = Path(file_path)
        if not self._path.exists():
            raise FileNotFoundError(f"File not found: {self._path}")
        ext = self._path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        self._ext = ext

    def extract_ordered(self) -> OrderedPDFDocument:
        """Extract content as an ``OrderedPDFDocument`` regardless of source format."""
        if self._ext == ".pdf":
            return PDFProcessor(self._path).extract_ordered()
        if self._ext in {".docx", ".doc"}:
            return DocxProcessor(self._path).extract_ordered()
        if self._ext in {".md", ".txt"}:
            return _extract_text_file(self._path)
        # Should never reach here due to __init__ validation
        raise ValueError(f"Unsupported extension: {self._ext}")
