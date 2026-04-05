"""Ingest route — upload, process, chunk, embed, and store documents.

Supported formats: .pdf, .docx, .doc, .md, .txt
"""

import asyncio
import base64
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.core import SETTING
from app.core.logging import get_logger
from app.database.bm25_store import BM25CorpusStore
from app.database.chroma import ChromaVectorStore
from app.llm import embeddings, llm_chat
from app.utils.chunker import DocumentChunker
from app.utils.document_processor import SUPPORTED_EXTENSIONS, DocumentProcessor
from app.utils.pdf_processor import ImageBlock, OrderedPDFDocument

ingest_route = APIRouter(prefix="/ingest", tags=["ingest"])
logger = get_logger(__name__)

_chunker = DocumentChunker()
_vector_store = ChromaVectorStore()
_bm25_store = BM25CorpusStore()


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
    The directory is only created when at least one image block exists.
    Returns the number of images successfully saved.
    """
    # Collect image blocks first — skip directory creation when there are none.
    image_pairs: list[tuple] = [
        (page, block)
        for page in doc.pages
        for block in page.blocks
        if isinstance(block, ImageBlock)
    ]
    if not image_pairs:
        return 0

    images_dir = Path(SETTING.IMAGES_OUTPUT_DIR) / stem
    images_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for page, block in image_pairs:
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
    suffix = Path(filename).suffix.lower()
    tmp_path: Path | None = None
    doc: OrderedPDFDocument
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        try:
            processor = DocumentProcessor(tmp_path)
            doc = await asyncio.to_thread(processor.extract_ordered)
        except Exception as exc:
            logger.error("Document extraction failed for %s: %s", filename, exc)
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
            chunk.metadata["image_path"] = str(images_dir / f"page_{pn}_img_{idx}.{ext}")

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

    # ── 6. Upsert into BM25 corpus (SQLite) ───────────────────────────────
    try:
        _bm25_store.upsert(
            [
                {"id": c.id, "document": c.content, "metadata": _sanitize_metadata(c.metadata)}
                for c in chunks
            ]
        )
        # Assemble full document text from all text/table chunks in order, then
        # store as a single row so the whole document is queryable without
        # having to re-join individual chunks later.
        full_content = "\n\n".join(
            c.content
            for c in chunks
            if c.metadata.get("block_type") not in ("image",)
        )
        _bm25_store.upsert_document(
            source_file=filename,
            content=full_content,
            metadata={"filename": filename, "total_chunks": len(chunks)},
        )
        logger.info("BM25 corpus updated: %d chunk(s) from %s.", len(chunks), filename)
    except Exception as exc:
        # Non-fatal — vector store already has the data
        logger.warning("BM25 corpus write failed for %s: %s", filename, exc)

    return (
        FileIngestResult(filename=filename, chunks_stored=len(chunks), images_saved=images_saved),
        None,
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@ingest_route.post("", response_model=IngestResponse, status_code=200)
async def ingest_documents(files: Annotated[list[UploadFile], File(...)]) -> IngestResponse:
    """Upload documents to be processed, chunked, embedded, and stored.

    Supported formats: ``.pdf``, ``.docx``, ``.doc``, ``.md``, ``.txt``

    For each file the endpoint will:

    1. Extract ordered text, table, and image blocks via ``DocumentProcessor``.
    2. Persist extracted images (PDF only) to ``IMAGES_OUTPUT_DIR/<stem>/``.
    3. Chunk every content block (text / table / image — images get an LLM description).
    4. Embed each chunk and upsert into the ChromaDB vector store.

    Returns a summary of chunks stored and images saved per file, plus any
    per-file errors that did not prevent other files from being processed.
    """
    if not files:
        raise HTTPException(status_code=422, detail="No files provided.")

    unsupported = [
        f.filename
        for f in files
        if Path(f.filename or "").suffix.lower() not in SUPPORTED_EXTENSIONS
    ]
    if unsupported:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported file type(s): {unsupported}. "
                f"Accepted: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )

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
