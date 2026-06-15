"""
Pydantic v2 schemas for API request / response validation.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr, Field


# ─── User ────────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    email: EmailStr


class UserRead(BaseModel):
    id: str
    username: str
    email: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ─── Document ────────────────────────────────────────────────────────────────

class DocumentRead(BaseModel):
    id: str
    filename: str
    original_filename: str
    file_size_bytes: int
    upload_date: datetime
    status: Literal["pending", "processing", "indexed", "error"]
    chunk_count: int
    error_message: str | None = None

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    total: int
    documents: list[DocumentRead]


# ─── Chat ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096, description="User question")
    session_id: str = Field(
        default="default-session",
        description="Conversation session identifier",
    )
    user_id: str | None = Field(None, description="Optional user identifier")


class ChatResponse(BaseModel):
    session_id: str
    query: str
    answer: str
    agent_used: str
    sources: list[dict] = Field(default_factory=list)


# ─── Ingestion ───────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    document_id: str


class IngestResponse(BaseModel):
    document_id: str
    status: str
    chunk_count: int
    message: str


# ─── Health ──────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    services: dict[str, str]