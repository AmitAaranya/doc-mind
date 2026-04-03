"""Ingest route — upload, process, chunk, embed, and store PDF documents."""

from __future__ import annotations

import asyncio
import base64
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from app.core import SETTING
from app.core.logging import get_logger
from app.database.chroma import ChromaVectorStore
from app.llm import embeddings, llm_chat
from app.utils.chunker import DocumentChunker
from app.utils.pdf_processor import ImageBlock, OrderedPDFDocument, PDFProcessor

ingest_route = APIRouter(prefix="/ingest", tags=["ingest"])
logger = get_logger(__name__)

_chunker = DocumentChunker()
_vector_store = ChromaVectorStore()


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class FileIngestResult(BaseModel):
    filename: str
    chunks_stored: int
    images_saved: int


class IngestResponse(BaseModel):
    results: list[FileIngestResult]
    total_chunks: int
    total_images_saved: int
    errors: dict[str, str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_images(doc: OrderedPDFDocument, stem: str) -> int:
    """Decode base64 image data from ImageBlocks and write files to disk.

    Files are written to ``IMAGES_OUTPUT_DIR/<stem>/page_N_img_I.<ext>``.
    Returns the number of images successfully saved.
    """
    images_dir = Path(SETTING.IMAGES_OUTPUT_DIR) / stem
    images_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for page in doc.pages:
        for block in page.blocks:
            if isinstance(block, ImageBlock):
                img = block.image
                dest = images_dir / f"page_{page.page_number}_img_{img.index}.{img.extension}"
                try:
                    dest.write_bytes(base64.b64decode(img.data_b64))
                    saved += 1
                except Exception as exc:
                    logger.warning("Could not save image %s: %s", dest, exc)
    return saved


def _sanitize_metadata(meta: dict) -> dict:
    """Convert metadata values to types accepted by ChromaDB (str/int/float/bool)."""
    sanitized: dict = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            sanitized[k] = v
        else:
            sanitized[k] = str(v)
    return sanitized


async def _process_one(
    upload: UploadFile,
) -> tuple[FileIngestResult | None, str | None]:
    """Process a single uploaded PDF file end-to-end.

    Returns ``(FileIngestResult, None)`` on success or ``(None, error_message)``
    on failure.
    """
    filename = upload.filename or "upload.pdf"

    try:
        content = await upload.read()
    except Exception as exc:
        return None, f"Could not read upload: {exc}"

    # ── 1. Write to a temp file so PDFProcessor can open it by path ───────
    tmp_path: Path | None = None
    doc: OrderedPDFDocument
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        try:
            processor = PDFProcessor(tmp_path)
            doc = await asyncio.to_thread(processor.extract_ordered)
        except Exception as exc:
            logger.error("PDF extraction failed for %s: %s", filename, exc)
            return None, f"Extraction failed: {exc}"
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()

    # ── 2. Save images to disk ─────────────────────────────────────────────
    stem = Path(filename).stem
    images_saved = _save_images(doc, stem)

    # ── 3. Chunk ───────────────────────────────────────────────────────────
    try:
        chunks = await _chunker.chunk_document(doc, llm_chat, source=filename)
    except Exception as exc:
        logger.error("Chunking failed for %s: %s", filename, exc)
        return None, f"Chunking failed: {exc}"

    if not chunks:
        return (
            FileIngestResult(filename=filename, chunks_stored=0, images_saved=images_saved),
            None,
        )

    # ── 3b. Inject image_path into image chunk metadata ───────────────────
    images_dir = Path(SETTING.IMAGES_OUTPUT_DIR) / stem
    for chunk in chunks:
        if chunk.metadata.get("block_type") == "image":
            pn = chunk.metadata["page_number"]
            idx = chunk.metadata["image_index"]
            ext = chunk.metadata["image_extension"]
            chunk.metadata["image_path"] = str(
                images_dir / f"page_{pn}_img_{idx}.{ext}"
            )

    # ── 4. Embed ───────────────────────────────────────────────────────────
    try:
        texts = [c.content for c in chunks]
        vectors = await asyncio.to_thread(embeddings.embed_documents, texts)
    except Exception as exc:
        logger.error("Embedding failed for %s: %s", filename, exc)
        return None, f"Embedding failed: {exc}"

    # ── 5. Upsert into vector store ────────────────────────────────────────
    try:
        _vector_store.add(
            ids=[c.id for c in chunks],
            embeddings=vectors,
            documents=texts,
            metadatas=[_sanitize_metadata(c.metadata) for c in chunks],
        )
    except Exception as exc:
        logger.error("Vector store write failed for %s: %s", filename, exc)
        return None, f"Vector store write failed: {exc}"

    return (
        FileIngestResult(filename=filename, chunks_stored=len(chunks), images_saved=images_saved),
        None,
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@ingest_route.post("", response_model=IngestResponse, status_code=200)
async def ingest_pdfs(files: list[UploadFile]) -> IngestResponse:
    """Upload a list of PDF files to be processed, chunked, embedded, and stored.

    For each PDF the endpoint will:

    1. Extract ordered text, table, and image blocks via ``PDFProcessor``.
    2. Persist extracted images to ``IMAGES_OUTPUT_DIR/<pdf-stem>/``.
    3. Chunk every content block (text / table / image — images get an LLM description).
    4. Embed each chunk and upsert into the ChromaDB vector store.

    Returns a summary of chunks stored and images saved per file, plus any
    per-file errors that did not prevent other files from being processed.
    """
    if not files:
        raise HTTPException(status_code=422, detail="No files provided.")

    non_pdfs = [f.filename for f in files if not (f.filename or "").lower().endswith(".pdf")]
    if non_pdfs:
        raise HTTPException(status_code=422, detail=f"Non-PDF files rejected: {non_pdfs}")

    results: list[FileIngestResult] = []
    errors: dict[str, str] = {}

    for upload in files:
        result, error = await _process_one(upload)
        if error:
            errors[upload.filename or "unknown"] = error
        elif result:
            results.append(result)

    return IngestResponse(
        results=results,
        total_chunks=sum(r.chunks_stored for r in results),
        total_images_saved=sum(r.images_saved for r in results),
        errors=errors,
    )
