import shutil
import uuid
from pathlib import Path

import aiofiles
from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database.models import Document
from app.utils.logger import get_logger

logger = get_logger(__name__)

UPLOAD_DIR = Path(settings.UPLOAD_DIR)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_BYTES = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024


class PDFService:
    """Manages PDF file storage and Document DB records."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def save_upload(self, file: UploadFile) -> Document:
        """
        Persist an uploaded PDF and create a DB record.

        Args:
            file: FastAPI UploadFile object.

        Returns:
            Newly created Document ORM object.

        Raises:
            ValueError: If the file exceeds the size limit or is not a PDF.
        """
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise ValueError("Only PDF files are supported.")

        doc_id = str(uuid.uuid4())
        safe_name = f"{doc_id}.pdf"
        dest_path = UPLOAD_DIR / safe_name

        # Stream to disk while checking size
        total = 0
        async with aiofiles.open(dest_path, "wb") as out:
            while chunk := await file.read(65536):
                total += len(chunk)
                if total > MAX_BYTES:
                    dest_path.unlink(missing_ok=True)
                    raise ValueError(
                        f"File exceeds maximum size of {settings.MAX_UPLOAD_SIZE_MB} MB."
                    )
                await out.write(chunk)

        doc = Document(
            id=doc_id,
            filename=safe_name,
            original_filename=file.filename,
            file_path=str(dest_path),
            file_size_bytes=total,
            status="pending",
        )
        self._db.add(doc)
        await self._db.flush()
        logger.info("PDF saved", doc_id=doc_id, filename=file.filename, bytes=total)
        return doc

    async def get_document(self, document_id: str) -> Document | None:
        """Fetch a Document record by ID."""
        result = await self._db.execute(
            select(Document).where(Document.id == document_id)
        )
        return result.scalar_one_or_none()

    async def list_documents(self) -> list[Document]:
        """Return all Document records ordered by upload date descending."""
        result = await self._db.execute(
            select(Document).order_by(Document.upload_date.desc())
        )
        return list(result.scalars().all())

    async def update_status(
        self,
        document_id: str,
        status: str,
        chunk_count: int = 0,
        error_message: str | None = None,
    ) -> None:
        """Update ingestion status on a Document record."""
        doc = await self.get_document(document_id)
        if doc:
            doc.status = status
            doc.chunk_count = chunk_count
            doc.error_message = error_message
            await self._db.flush()
