"""PDF processor — extracts text, images, and tables from PDF files.

Public API (original — unordered)
----------------------------------
processor = PDFProcessor("report.pdf")

processor.extract_text()    -> PDFDocument (text only)
processor.extract_images()  -> PDFDocument (images only)
processor.extract_tables()  -> PDFDocument (tables only)
processor.extract_all()     -> PDFDocument (text + images + tables)

Public API (ordered — for RAG chunking)
----------------------------------------
processor.extract_ordered() -> OrderedPDFDocument

Every page in OrderedPDFDocument has an ordered list of Block objects
(TextBlock | ImageBlock | TableBlock) sorted by reading order (top→bottom,
left→right).  This lets downstream chunkers know *exactly* where on the page
each piece of content appeared.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import fitz  # pymupdf
import pdfplumber

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ExtractMode(StrEnum):
    TEXT = "text"
    IMAGES = "images"
    TABLES = "tables"
    ALL = "all"


class BlockType(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

# (x0, y0, x1, y1) in PDF-point units, origin top-left, y increases downward
BBox = tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Data models — original (unordered, flat per-page)
# ---------------------------------------------------------------------------


@dataclass
class ImageData:
    """A single image extracted from a PDF page."""

    index: int
    width: int
    height: int
    extension: str  # e.g. "png", "jpeg"
    data_b64: str  # base64-encoded raw image bytes


@dataclass
class TableData:
    """A single table extracted from a PDF page."""

    index: int
    headers: list[str | None] | None  # first row treated as header (may be None)
    rows: list[list[str | None]]  # remaining data rows


@dataclass
class PageData:
    """All extracted content for one PDF page."""

    page_number: int  # 1-based
    text: str = ""
    images: list[ImageData] = field(default_factory=list)
    tables: list[TableData] = field(default_factory=list)


@dataclass
class PDFDocument:
    """Extraction result for an entire PDF file."""

    file_path: str
    total_pages: int
    metadata: dict
    pages: list[PageData] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience aggregators
    # ------------------------------------------------------------------

    @property
    def all_text(self) -> str:
        """Concatenated text from every page, with page markers."""
        return "\n\n".join(
            f"[Page {p.page_number}]\n{p.text}" for p in self.pages if p.text.strip()
        )

    @property
    def all_images(self) -> list[tuple[int, ImageData]]:
        """Flat list of (page_number, ImageData) for every image in the document."""
        return [(p.page_number, img) for p in self.pages for img in p.images]

    @property
    def all_tables(self) -> list[tuple[int, TableData]]:
        """Flat list of (page_number, TableData) for every table in the document."""
        return [(p.page_number, tbl) for p in self.pages for tbl in p.tables]


# ---------------------------------------------------------------------------
# Data models — ordered blocks (for RAG chunking)
# ---------------------------------------------------------------------------


@dataclass
class Block:
    """Base class for a positioned content block on a PDF page."""

    block_type: BlockType
    page_number: int  # 1-based
    order: int  # 0-based reading order within the page
    bbox: BBox  # (x0, y0, x1, y1) in PDF points


@dataclass
class TextBlock(Block):
    """A run of text extracted from a specific position on the page."""

    content: str = ""


@dataclass
class ImageBlock(Block):
    """An image extracted from a specific position on the page."""

    image: ImageData = field(default_factory=lambda: ImageData(0, 0, 0, "png", ""))


@dataclass
class TableBlock(Block):
    """A table extracted from a specific position on the page."""

    table: TableData = field(default_factory=lambda: TableData(0, None, []))

    def to_markdown(self) -> str:
        """Convert this table to a GitHub-Flavored Markdown string."""
        headers = [str(h or "") for h in (self.table.headers or [])]
        if not headers:
            return ""
        sep = ["---"] * len(headers)
        rows = [[str(c or "") for c in row] for row in self.table.rows]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(sep) + " |",
            *("| " + " | ".join(row) + " |" for row in rows),
        ]
        return "\n".join(lines)


@dataclass
class OrderedPageData:
    """All content blocks for one PDF page, sorted in reading order."""

    page_number: int  # 1-based
    blocks: list[Block] = field(default_factory=list)

    @property
    def text_blocks(self) -> list[TextBlock]:
        return [b for b in self.blocks if isinstance(b, TextBlock)]

    @property
    def image_blocks(self) -> list[ImageBlock]:
        return [b for b in self.blocks if isinstance(b, ImageBlock)]

    @property
    def table_blocks(self) -> list[TableBlock]:
        return [b for b in self.blocks if isinstance(b, TableBlock)]


@dataclass
class OrderedPDFDocument:
    """Ordered extraction result for an entire PDF file."""

    file_path: str
    total_pages: int
    metadata: dict
    pages: list[OrderedPageData] = field(default_factory=list)

    @property
    def all_blocks(self) -> list[Block]:
        """Flat list of every block across all pages, in page + reading order."""
        return [block for page in self.pages for block in page.blocks]


class PDFProcessor:
    """Extract text, images, and/or tables from a PDF file.

    Parameters
    ----------
    file_path:
        Path to the PDF file to process.
    """

    def __init__(self, file_path: str | Path) -> None:
        self._path = Path(file_path)
        if not self._path.exists():
            raise FileNotFoundError(f"PDF not found: {self._path}")
        if self._path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {self._path.suffix}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_text(self) -> PDFDocument:
        """Return a PDFDocument populated with text for each page."""
        doc = self._open_fitz()
        try:
            pages = [
                PageData(page_number=i + 1, text=self._page_text(doc.load_page(i)))
                for i in range(doc.page_count)
            ]
            return self._build_document(doc, pages)
        finally:
            doc.close()

    def extract_images(self) -> PDFDocument:
        """Return a PDFDocument populated with images for each page."""
        doc = self._open_fitz()
        try:
            pages = [
                PageData(
                    page_number=i + 1,
                    images=self._page_images(doc, doc.load_page(i)),
                )
                for i in range(doc.page_count)
            ]
            return self._build_document(doc, pages)
        finally:
            doc.close()

    def extract_tables(self) -> PDFDocument:
        """Return a PDFDocument populated with tables for each page."""
        pages = self._pages_via_pdfplumber()
        doc = self._open_fitz()
        try:
            return self._build_document(doc, pages)
        finally:
            doc.close()

    def extract_all(self) -> PDFDocument:
        """Return a PDFDocument with text, images, and tables for each page."""
        # pdfplumber handles tables; fitz handles everything else
        table_map: dict[int, PageData] = {p.page_number: p for p in self._pages_via_pdfplumber()}
        doc = self._open_fitz()
        try:
            pages: list[PageData] = []
            for i in range(doc.page_count):
                pn = i + 1
                fitz_page = doc.load_page(i)
                page_data = table_map.get(pn, PageData(page_number=pn))
                page_data.text = self._page_text(fitz_page)
                page_data.images = self._page_images(doc, fitz_page)
                pages.append(page_data)
            return self._build_document(doc, pages)
        finally:
            doc.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _open_fitz(self) -> fitz.Document:
        return fitz.open(str(self._path))

    def _build_document(self, fitz_doc: fitz.Document, pages: list[PageData]) -> PDFDocument:
        return PDFDocument(
            file_path=str(self._path),
            total_pages=fitz_doc.page_count,
            metadata=fitz_doc.metadata or {},
            pages=pages,
        )

    @staticmethod
    def _page_text(page: fitz.Page) -> str:
        return page.get_text("text").strip()  # type: ignore[union-attr]

    @staticmethod
    def _page_images(doc: fitz.Document, page: fitz.Page) -> list[ImageData]:
        images: list[ImageData] = []
        for idx, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
            except Exception:
                continue
            images.append(
                ImageData(
                    index=idx,
                    width=base_image.get("width", 0),
                    height=base_image.get("height", 0),
                    extension=base_image.get("ext", "png"),
                    data_b64=base64.b64encode(base_image["image"]).decode(),
                )
            )
        return images

    def _pages_via_pdfplumber(self) -> list[PageData]:
        """Use pdfplumber to extract tables from every page."""
        pages: list[PageData] = []
        with pdfplumber.open(str(self._path)) as pdf:
            for i, pdf_page in enumerate(pdf.pages):
                raw_tables = pdf_page.extract_tables() or []
                tables = [
                    TableData(
                        index=t_idx,
                        headers=table[0] if table else None,
                        rows=table[1:] if len(table) > 1 else [],
                    )
                    for t_idx, table in enumerate(raw_tables)
                ]
                pages.append(PageData(page_number=i + 1, tables=tables))
        return pages

    # ------------------------------------------------------------------
    # Ordered extraction (for RAG chunking pipeline)
    # ------------------------------------------------------------------

    def extract_ordered(self) -> OrderedPDFDocument:
        """Return an OrderedPDFDocument where every page contains Block objects
        (TextBlock | ImageBlock | TableBlock) sorted by reading order.

        Algorithm per page
        ------------------
        1. pdfplumber locates table regions and extracts their data + bbox.
        2. fitz provides every text and image block with its bbox.
        3. Text blocks whose centre falls inside a table region are suppressed
           (pdfplumber owns that content — avoids double-extraction).
        4. All blocks are merged and sorted by (y0, x0) → reading order.
        5. `block.order` is set to the 0-based index after sorting.
        """
        fitz_doc = self._open_fitz()
        try:
            with pdfplumber.open(str(self._path)) as pdf:
                pages = [
                    self._extract_ordered_page(fitz_doc[i], pdf.pages[i])
                    for i in range(fitz_doc.page_count)
                ]
            return OrderedPDFDocument(
                file_path=str(self._path),
                total_pages=fitz_doc.page_count,
                metadata=fitz_doc.metadata or {},
                pages=pages,
            )
        finally:
            fitz_doc.close()

    @staticmethod
    def _extract_ordered_page(
        fitz_page: fitz.Page,
        plumber_page,  # pdfplumber.page.PageImage
    ) -> OrderedPageData:
        pn = fitz_page.number + 1  # type: ignore[operator]  # fitz uses 0-based page index

        # ── 1. Tables via pdfplumber ──────────────────────────────────────
        table_blocks: list[Block] = []
        table_bboxes: list[BBox] = []

        for t_idx, tbl in enumerate(plumber_page.find_tables()):
            tbl_bbox: BBox = tbl.bbox  # (x0, top, x1, bottom)
            raw = tbl.extract()
            if not raw:
                continue
            table_data = TableData(
                index=t_idx,
                headers=raw[0],
                rows=raw[1:] if len(raw) > 1 else [],
            )
            table_blocks.append(
                TableBlock(
                    block_type=BlockType.TABLE,
                    page_number=pn,
                    order=0,
                    bbox=tbl_bbox,
                    table=table_data,
                )
            )
            table_bboxes.append(tbl_bbox)

        # ── 2. Text & image blocks via fitz ──────────────────────────────
        content_blocks: list[Block] = []
        img_counter = 0

        fitz_dict = fitz_page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        for blk in fitz_dict.get("blocks", []):  # type: ignore[union-attr]
            blk_bbox: BBox = tuple(blk["bbox"])  # type: ignore[assignment]

            if blk["type"] == 0:  # text block
                # Suppress if block centre falls inside a table region
                cx = (blk_bbox[0] + blk_bbox[2]) / 2
                cy = (blk_bbox[1] + blk_bbox[3]) / 2
                if any(tb[0] <= cx <= tb[2] and tb[1] <= cy <= tb[3] for tb in table_bboxes):
                    continue

                # Concatenate all spans on all lines
                content_parts: list[str] = []
                for line in blk.get("lines", []):
                    for span in line.get("spans", []):
                        content_parts.append(span.get("text", ""))
                    content_parts.append("\n")
                content = "".join(content_parts).strip()
                if not content:
                    continue

                content_blocks.append(
                    TextBlock(
                        block_type=BlockType.TEXT,
                        page_number=pn,
                        order=0,
                        bbox=blk_bbox,
                        content=content,
                    )
                )

            elif blk["type"] == 1:  # image block
                img_bytes: bytes | None = blk.get("image")
                if not img_bytes:
                    continue
                img_data = ImageData(
                    index=img_counter,
                    width=blk.get("width", 0),
                    height=blk.get("height", 0),
                    extension=blk.get("ext", "png"),
                    data_b64=base64.b64encode(img_bytes).decode(),
                )
                img_counter += 1
                content_blocks.append(
                    ImageBlock(
                        block_type=BlockType.IMAGE,
                        page_number=pn,
                        order=0,
                        bbox=blk_bbox,
                        image=img_data,
                    )
                )

        # ── 3. Merge + sort by (y0, x0) → reading order ──────────────────
        all_blocks = sorted(
            table_blocks + content_blocks,
            key=lambda b: (b.bbox[1], b.bbox[0]),
        )
        for idx, blk in enumerate(all_blocks):
            blk.order = idx

        return OrderedPageData(page_number=pn, blocks=all_blocks)
