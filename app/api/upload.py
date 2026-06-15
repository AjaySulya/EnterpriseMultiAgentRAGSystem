"""
Upload API — PDF upload, ingestion trigger, and document listing.
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.db import get_db
from app.database.schemas import DocumentRead, DocumentListResponse, IngestRequest, IngestResponse
from app.services.pdf_service import PDFService
from app.rag.ingestion import ingest_pdf
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


async def _run_ingestion(
    document_id: str,
    file_path: str,
    original_filename: str,
    db_session_factory,
) -> None:
    """
    Background task: run full ingestion pipeline and update DB status.
    Opens its own DB session so it is independent of the request session.
    """
    from app.database.db import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        svc = PDFService(db)
        try:
            await svc.update_status(document_id, "processing")
            await db.commit()

            chunk_count = ingest_pdf(
                file_path=file_path,
                document_id=document_id,
                source_name=original_filename,
            )

            await svc.update_status(document_id, "indexed", chunk_count=chunk_count)
            await db.commit()
            logger.info("Background ingestion complete", document_id=document_id, chunks=chunk_count)

        except Exception as exc:
            logger.error("Background ingestion failed", document_id=document_id, error=str(exc))
            await svc.update_status(document_id, "error", error_message=str(exc))
            await db.commit()


@router.post(
    "/upload-pdf",
    response_model=DocumentRead,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a PDF document",
)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to upload"),
    db: AsyncSession = Depends(get_db),
) -> DocumentRead:
    """
    Upload a PDF file.  
    The file is saved to disk, a DB record is created, and ingestion
    (text extraction → chunking → embedding → ChromaDB) is queued as
    a background task so the response is immediate.
    """
    svc = PDFService(db)

    try:
        doc = await svc.save_upload(file)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    # Queue ingestion without blocking the response
    background_tasks.add_task(
        _run_ingestion,
        document_id=doc.id,
        file_path=doc.file_path,
        original_filename=doc.original_filename,
        db_session_factory=None,  # factory resolved inside the task
    )

    logger.info("PDF upload accepted", document_id=doc.id, filename=doc.original_filename)
    return DocumentRead.model_validate(doc)


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Re-trigger ingestion for an existing document",
)
async def ingest_document(
    payload: IngestRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> IngestResponse:
    """
    Manually trigger (or re-trigger) the ingestion pipeline for a
    previously uploaded document.  Useful for retrying failed ingestions.
    """
    svc = PDFService(db)
    doc = await svc.get_document(payload.document_id)

    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{payload.document_id}' not found.",
        )

    background_tasks.add_task(
        _run_ingestion,
        document_id=doc.id,
        file_path=doc.file_path,
        original_filename=doc.original_filename,
        db_session_factory=None,
    )

    return IngestResponse(
        document_id=doc.id,
        status="processing",
        chunk_count=0,
        message="Ingestion queued. Poll GET /api/v1/documents to track status.",
    )


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all uploaded documents",
)
async def list_documents(db: AsyncSession = Depends(get_db)) -> DocumentListResponse:
    """Return metadata for every uploaded document, newest first."""
    svc = PDFService(db)
    docs = await svc.list_documents()
    return DocumentListResponse(
        total=len(docs),
        documents=[DocumentRead.model_validate(d) for d in docs],
    )


@router.get(
    "/documents/{document_id}",
    response_model=DocumentRead,
    summary="Get a single document by ID",
)
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
) -> DocumentRead:
    """Retrieve metadata for a specific document."""
    svc = PDFService(db)
    doc = await svc.get_document(document_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )
    return DocumentRead.model_validate(doc)