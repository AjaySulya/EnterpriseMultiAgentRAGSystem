"""
Enterprise Multi-Agent RAG System — Application Entry Point
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, upload, health
from app.config import settings
from app.database.db import create_tables
from app.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown events."""
    logger.info("Starting Enterprise RAG System", version=settings.APP_VERSION)
    await create_tables()
    logger.info("Database tables ensured")
    yield
    logger.info("Shutting down Enterprise RAG System")


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Enterprise Multi-Agent RAG System with PDF, SQL, and Web agents.",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, tags=["Health"])
    app.include_router(upload.router, prefix="/api/v1", tags=["Documents"])
    app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])

    return app


app = create_app()