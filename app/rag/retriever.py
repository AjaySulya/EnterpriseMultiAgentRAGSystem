"""
RAG retriever — wraps VectorStore and formats retrieved context.
"""
 
from typing import Any
 
from app.config import settings
from app.rag.vector_store import VectorStore
from app.utils.logger import get_logger
 
logger = get_logger(__name__)
 
 
class RAGRetriever:
    """
    Retrieves relevant document chunks and assembles context for the LLM.
    """
 
    def __init__(self) -> None:
        self._store = VectorStore()
 
    def retrieve(
        self,
        query: str,
        k: int | None = None,
        document_id: str | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Retrieve the top-k relevant chunks and format them as a context string.
 
        Args:
            query:       The user's question.
            k:           Number of chunks to retrieve.
            document_id: Restrict search to a specific document.
 
        Returns:
            A tuple of (formatted_context, raw_hits).
        """
        where_filter = {"document_id": document_id} if document_id else None
 
        hits = self._store.similarity_search(
            query=query,
            k=k or settings.TOP_K_RESULTS,
            where=where_filter,
        )
 
        if not hits:
            logger.warning("No relevant chunks found", query=query)
            return "No relevant documents found.", []
 
        context_parts: list[str] = []
        for i, hit in enumerate(hits, 1):
            source = hit["metadata"].get("source", "unknown")
            page = hit["metadata"].get("page", "?")
            context_parts.append(
                f"[Source {i}: {source}, page {page}]\n{hit['text']}"
            )
 
        formatted = "\n\n---\n\n".join(context_parts)
        logger.debug("Context assembled", chunks=len(hits))
        return formatted, hits
 
    def format_prompt(self, query: str, context: str) -> str:
        """
        Build the final LLM prompt from query and context.
 
        Args:
            query:   The user question.
            context: Retrieved context string.
 
        Returns:
            Full prompt string.
        """
        return (
            "You are an expert assistant. "
            "Answer the question using ONLY the context below. "
            "If the answer is not in the context, say 'I don't know'.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
 