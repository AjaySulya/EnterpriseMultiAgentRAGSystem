"""
Health check endpoint — verifies all downstream services.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.config import settings
from app.database.db import get_db
from app.database.schemas import HealthResponse
from app.rag.vector_store import get_chroma_client
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/health", response_model=HealthResponse, summary="System health check")
async def health_check(db: AsyncSession = Depends(get_db)) -> HealthResponse:
    """
    Return the health status of the API and all connected services.
    """
    service_status: dict[str, str] = {}

    # PostgreSQL
    try:
        await db.execute(text("SELECT 1"))
        service_status["postgresql"] = "healthy"
    except Exception as exc:
        logger.error("PostgreSQL health check failed", error=str(exc))
        service_status["postgresql"] = f"unhealthy: {exc}"

    # ChromaDB
    try:
        client = get_chroma_client()
        client.heartbeat()
        service_status["chromadb"] = "healthy"
    except Exception as exc:
        logger.error("ChromaDB health check failed", error=str(exc))
        service_status["chromadb"] = f"unhealthy: {exc}"

    overall = (
        "healthy"
        if all(v == "healthy" for v in service_status.values())
        else "degraded"
    )

    return HealthResponse(
        status=overall,
        version=settings.APP_VERSION,
        services=service_status,
    )