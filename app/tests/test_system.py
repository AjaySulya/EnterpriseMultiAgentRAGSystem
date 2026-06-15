"""
Test suite for the Enterprise RAG System.

Run with:  pytest tests/ -v
"""

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_db():
    """Return a minimal async DB session mock."""
    db = AsyncMock()
    db.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=lambda: None, scalars=lambda: MagicMock(all=lambda: [])))
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    db.close = AsyncMock()
    return db


@pytest.fixture
def app(mock_db):
    """Create a test FastAPI app with DB dependency overridden."""
    from app.main import app as fastapi_app
    from app.database.db import get_db

    async def override_get_db():
        yield mock_db

    fastapi_app.dependency_overrides[get_db] = override_get_db
    yield fastapi_app
    fastapi_app.dependency_overrides.clear()


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


# ─── Router Agent unit tests ─────────────────────────────────────────────────

class TestRouterClassifier:
    def test_pdf_route(self):
        from app.agents.router_agent import _classify
        assert _classify("What does the uploaded document say about risk?") == "pdf"

    def test_sql_route(self):
        from app.agents.router_agent import _classify
        assert _classify("How many users registered this month?") == "sql"
        assert _classify("Show me all database records") == "sql"

    def test_web_route(self):
        from app.agents.router_agent import _classify
        assert _classify("Summarise https://example.com/article") == "web"
        assert _classify("What does http://news.site say about AI?") == "web"

    def test_default_to_pdf(self):
        from app.agents.router_agent import _classify
        # Ambiguous → defaults to pdf
        assert _classify("Tell me something interesting") == "pdf"


# ─── PDF Agent unit tests ────────────────────────────────────────────────────

class TestPDFAgent:
    @pytest.mark.asyncio
    async def test_no_hits_returns_fallback(self):
        from app.agents.pdf_agent import PDFAgent

        agent = PDFAgent()
        with patch.object(agent._retriever, "retrieve", return_value=("", [])):
            result = await agent.answer("What is in the document?")

        assert result["agent"] == "pdf_agent"
        assert "couldn't find" in result["answer"].lower()
        assert result["sources"] == []

    @pytest.mark.asyncio
    async def test_llm_error_graceful_degradation(self):
        from app.agents.pdf_agent import PDFAgent

        agent = PDFAgent()
        fake_hits = [{"text": "Some chunk", "metadata": {"source": "test.pdf", "page": 1}, "distance": 0.1}]
        with patch.object(agent._retriever, "retrieve", return_value=("Some chunk", fake_hits)):
            with patch("app.agents.pdf_agent.httpx.AsyncClient") as mock_client:
                mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                    side_effect=Exception("Connection refused")
                )
                result = await agent.answer("What is the main topic?")

        assert result["agent"] == "pdf_agent"
        assert isinstance(result["answer"], str)


# ─── SQL Agent unit tests ────────────────────────────────────────────────────

class TestSQLAgent:
    def test_safe_select_allowed(self):
        from app.agents.sql_agent import SQLAgent
        agent = SQLAgent(db=AsyncMock())
        assert agent._is_safe("SELECT count(*) FROM users;") is True

    def test_drop_blocked(self):
        from app.agents.sql_agent import SQLAgent
        agent = SQLAgent(db=AsyncMock())
        assert agent._is_safe("DROP TABLE users;") is False

    def test_delete_blocked(self):
        from app.agents.sql_agent import SQLAgent
        agent = SQLAgent(db=AsyncMock())
        assert agent._is_safe("DELETE FROM documents WHERE id='x';") is False

    def test_insert_blocked(self):
        from app.agents.sql_agent import SQLAgent
        agent = SQLAgent(db=AsyncMock())
        assert agent._is_safe("INSERT INTO users VALUES ('a','b','c');") is False

    def test_format_empty_results(self):
        from app.agents.sql_agent import SQLAgent
        result = SQLAgent._format_results("q", "SELECT 1;", [])
        assert "no results" in result.lower()

    def test_format_with_rows(self):
        from app.agents.sql_agent import SQLAgent
        rows = [{"username": "alice", "email": "a@b.com"}]
        result = SQLAgent._format_results("List users", "SELECT * FROM users;", rows)
        assert "alice" in result


# ─── Web Agent unit tests ────────────────────────────────────────────────────

class TestWebAgent:
    def test_extract_url(self):
        from app.agents.web_agent import _extract_url
        assert _extract_url("Check https://example.com/page") == "https://example.com/page"
        assert _extract_url("No URL here") is None

    def test_domain_slug(self):
        from app.agents.web_agent import _domain_slug
        slug = _domain_slug("https://www.example.com/path")
        assert "." not in slug
        assert len(slug) <= 40

    @pytest.mark.asyncio
    async def test_no_url_in_query(self):
        from app.agents.web_agent import WEBAgent
        agent = WEBAgent()
        result = await agent.answer("Tell me about the news")
        assert result["agent"] == "web_agent"
        assert "url" in result["answer"].lower() or "URL" in result["answer"]


# ─── Ingestion unit tests ────────────────────────────────────────────────────

class TestIngestion:
    def test_chunk_pages_produces_chunks(self):
        from app.rag.ingestion import chunk_pages
        pages = [{"text": "Hello world. " * 100, "page": 1}]
        texts, metas, ids = chunk_pages(pages, "test.pdf", "doc-123")
        assert len(texts) > 0
        assert all(m["document_id"] == "doc-123" for m in metas)
        assert len(texts) == len(ids) == len(metas)

    def test_chunk_metadata_fields(self):
        from app.rag.ingestion import chunk_pages
        pages = [{"text": "A" * 600, "page": 5}]
        _, metas, _ = chunk_pages(pages, "report.pdf", "xyz")
        for m in metas:
            assert "source" in m
            assert "page" in m
            assert "document_id" in m


# ─── API route integration tests ─────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        with patch("app.api.health.get_chroma_client") as mock_chroma:
            mock_chroma.return_value.heartbeat.return_value = True
            resp = client.get("/health")
        # May be degraded if postgres mock is partial, but must return 200
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "version" in data
        assert "services" in data


class TestDocumentEndpoints:
    def test_list_documents_empty(self, client, mock_db):
        from sqlalchemy.ext.asyncio import AsyncSession
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute = AsyncMock(return_value=mock_result)

        resp = client.get("/api/v1/documents")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_get_nonexistent_document(self, client, mock_db):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        resp = client.get(f"/api/v1/documents/{uuid.uuid4()}")
        assert resp.status_code == 404


class TestChatEndpoint:
    def test_empty_query_rejected(self, client):
        resp = client.post("/api/v1/chat", json={"query": "  ", "session_id": "test"})
        assert resp.status_code in (400, 422)

    @pytest.mark.asyncio
    async def test_chat_returns_answer(self, app):
        fake_state = {
            "answer": "The document discusses AI systems.",
            "agent_used": "pdf_agent",
            "sources": [],
        }
        with patch("app.api.chat.run_rag_pipeline", AsyncMock(return_value=fake_state)):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                resp = await ac.post(
                    "/api/v1/chat",
                    json={"query": "What is this document about?", "session_id": "s1"},
                )
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "The document discusses AI systems."
        assert data["agent_used"] == "pdf_agent"