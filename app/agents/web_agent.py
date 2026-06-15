"""
Web Agent — fetches a URL, extracts text, embeds it on-the-fly,
retrieves relevant passages, and generates an answer.
"""

import re
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from app.config import settings
from app.rag.retriever import RAGRetriever
from app.rag.vector_store import VectorStore
from app.utils.logger import get_logger

logger = get_logger(__name__)

HF_API_URL = (
    f"https://api-inference.huggingface.co/models/{settings.HF_LLM_MODEL}"
)


def _extract_url(text: str) -> str | None:
    """Pull a URL from a free-text query."""
    pattern = re.compile(
        r"https?://[^\s\"'<>]+"
    )
    match = pattern.search(text)
    return match.group(0) if match else None


class WEBAgent:
    """
    Handles queries that involve website content.

    Strategy:
    1. Extract URL from the query (or use a pre-scraped page).
    2. Fetch & clean the page.
    3. Chunk and embed the content into a temporary ChromaDB collection.
    4. Run similarity search + LLM generation.
    """

    def __init__(self) -> None:
        self._retriever = RAGRetriever()

    async def answer(self, query: str, url: str | None = None) -> dict[str, Any]:
        """
        Answer a question about web content.

        Args:
            query: The user's question.
            url:   Explicit URL to scrape (or extracted from query).

        Returns:
            Dict with 'answer', 'sources', 'url', and 'agent'.
        """
        logger.info("WebAgent processing query", query=query[:80])

        target_url = url or _extract_url(query)
        if not target_url:
            return {
                "answer": (
                    "Please include a URL in your question so I can fetch the page content. "
                    "Example: 'Summarize https://example.com/article'"
                ),
                "sources": [],
                "url": None,
                "agent": "web_agent",
            }

        # Fetch page
        page_text, error = await self._fetch_page(target_url)
        if error:
            return {
                "answer": f"Could not fetch the page: {error}",
                "sources": [],
                "url": target_url,
                "agent": "web_agent",
            }

        # Store in a per-domain temporary collection and search
        collection_name = f"web_{_domain_slug(target_url)}"
        store = VectorStore(collection_name=collection_name)

        import uuid
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        chunks = splitter.split_text(page_text)

        if chunks:
            ids = [str(uuid.uuid4()) for _ in chunks]
            metadatas = [{"source": target_url, "page": "web"} for _ in chunks]
            store.add_documents(chunks, metadatas, ids)

        hits = store.similarity_search(query, k=settings.TOP_K_RESULTS)
        if not hits:
            return {
                "answer": "No relevant content found on the page.",
                "sources": [],
                "url": target_url,
                "agent": "web_agent",
            }

        context = "\n\n---\n\n".join(h["text"] for h in hits)
        prompt = self._retriever.format_prompt(query, context)
        answer_text = await self._call_llm(prompt)

        return {
            "answer": answer_text,
            "sources": [{"url": target_url, "chunks_used": len(hits)}],
            "url": target_url,
            "agent": "web_agent",
        }

    async def _fetch_page(self, url: str) -> tuple[str, str | None]:
        """Download and parse a web page. Returns (text, error)."""
        try:
            async with httpx.AsyncClient(
                timeout=settings.WEB_TIMEOUT_SECONDS,
                follow_redirects=True,
            ) as client:
                resp = await client.get(url, headers={"User-Agent": "EnterpriseRAG/1.0"})
                resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "lxml")
            # Remove boilerplate
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            text = soup.get_text(separator=" ", strip=True)
            text = re.sub(r"\s{3,}", "\n\n", text)
            logger.info("Page fetched", url=url, chars=len(text))
            return text, None

        except httpx.HTTPStatusError as exc:
            return "", f"HTTP {exc.response.status_code}"
        except Exception as exc:
            logger.error("Page fetch failed", url=url, error=str(exc))
            return "", str(exc)

    async def _call_llm(self, prompt: str) -> str:
        """Call the HuggingFace Inference API."""
        headers = {"Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": settings.MAX_TOKENS,
                "temperature": settings.TEMPERATURE,
                "return_full_text": False,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(HF_API_URL, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()

            if isinstance(data, list) and data:
                return data[0].get("generated_text", "").strip()
            return str(data)

        except Exception as exc:
            logger.error("LLM call failed", error=str(exc))
            return "LLM unavailable — please try again later."


def _domain_slug(url: str) -> str:
    """Convert a URL to a safe collection-name slug."""
    domain = urlparse(url).netloc.replace(".", "_").replace("-", "_")
    return domain[:40]