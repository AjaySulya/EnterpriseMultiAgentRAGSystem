"""
PDF Agent — answers questions from ingested PDF documents via RAG.
"""

from typing import Any

import httpx

from app.config import settings
from app.rag.retriever import RAGRetriever
from app.utils.logger import get_logger

logger = get_logger(__name__)

HF_API_URL = (
    f"https://api-inference.huggingface.co/models/{settings.HF_LLM_MODEL}"
)


class PDFAgent:
    """
    Retrieves relevant PDF chunks and generates an answer via
    the HuggingFace Inference API.
    """

    def __init__(self) -> None:
        self._retriever = RAGRetriever()

    async def answer(self, query: str) -> dict[str, Any]:
        """
        Run the PDF RAG pipeline for the given query.

        Args:
            query: Natural-language question.

        Returns:
            Dict with 'answer', 'sources', and 'agent'.
        """
        logger.info("PDFAgent processing query", query=query[:80])

        context, hits = self._retriever.retrieve(query)

        if not hits:
            return {
                "answer": "I couldn't find relevant information in the uploaded documents.",
                "sources": [],
                "agent": "pdf_agent",
            }

        prompt = self._retriever.format_prompt(query, context)
        answer_text = await self._call_llm(prompt)

        sources = [
            {
                "source": h["metadata"].get("source", "unknown"),
                "page": h["metadata"].get("page"),
                "score": round(1 - h["distance"], 4),
            }
            for h in hits
        ]

        return {"answer": answer_text, "sources": sources, "agent": "pdf_agent"}

    async def _call_llm(self, prompt: str) -> str:
        """
        Call the HuggingFace Inference API.

        Args:
            prompt: Formatted prompt string.

        Returns:
            Generated text from the LLM.
        """
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
                response = await client.post(
                    HF_API_URL, json=payload, headers=headers
                )
                response.raise_for_status()
                data = response.json()

            if isinstance(data, list) and data:
                return data[0].get("generated_text", "").strip()
            return str(data)

        except httpx.HTTPStatusError as exc:
            logger.error(
                "HuggingFace API error",
                status=exc.response.status_code,
                detail=exc.response.text[:200],
            )
            # Graceful degradation — return context summary
            return (
                f"(LLM unavailable — raw context follows)\n\n"
                f"{prompt.split('Context:')[-1].split('Question:')[0].strip()[:800]}"
            )
        except Exception as exc:
            logger.exception("Unexpected LLM error", error=str(exc))
            return "An error occurred while generating the answer."