"""
ChromaBD vector store wrapper.
Provide the documents ingestiona and similarity search.
"""

from functools import lru_cache
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.rag.embeddings import get_embedding_model
from app.utils.logger import get_logger


logger = get_logger(__name__)



@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.HttpClient:
    """
    return a cached ChromaDB HTTP client.
    
    """
    logger.info(
                "Connecting to ChromaDB",
                host=settings.CHROMA_HOST,
                port=settings.CHROMA_PORT,
            )

    client = chromadb.HttpClient(
        host=settings.CHROMA_HOST,
        port=settings.CHROMA_PORT,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    
    return client


def get_or_create_collection(collection_name: str | None = None) -> chromadb.Collection:
    """
    Get or create a ChromaDB collection using cosine similarity.
 
    Args:
        collection_name: Name of the collection (defaults to settings value).
 
    Returns:
        ChromaDB collection object.
    """
    name = collection_name or settings.CHROMA_COLLECTION_NAME
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.debug("Collection ready", collection=name)
    return collection
 
 
class VectorStore:
    """
    High-level ChromaDB interface for document storage and retrieval.
    """
 
    def __init__(self, collection_name: str | None = None) -> None:
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self._embeddings = get_embedding_model()
 
    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict],
        ids: list[str],
    ) -> int:
        """
        Embed and add documents to the vector store.
 
        Args:
            texts:     Text chunks to embed.
            metadatas: Metadata dicts aligned with texts.
            ids:       Unique IDs aligned with texts.
 
        Returns:
            Number of documents added.
        """
        if not texts:
            return 0
 
        collection = get_or_create_collection(self.collection_name)
        embeddings = self._embeddings.embed_documents(texts)
 
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info("Documents upserted to ChromaDB", count=len(texts))
        return len(texts)
 
    def similarity_search(
        self,
        query: str,
        k: int | None = None,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        """
        Find the top-k most similar documents for a query.
 
        Args:
            query: Query string.
            k:     Number of results (defaults to TOP_K_RESULTS).
            where: Optional ChromaDB metadata filter.
 
        Returns:
            List of dicts with 'text', 'metadata', 'id', and 'distance'.
        """
        n = k or settings.TOP_K_RESULTS
        collection = get_or_create_collection(self.collection_name)
        query_embedding = self._embeddings.embed_query(query)
 
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n, collection.count() or 1),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
 
        hits: list[dict[str, Any]] = []
        for doc, meta, dist, doc_id in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0],
        ):
            hits.append(
                {
                    "text": doc,
                    "metadata": meta,
                    "id": doc_id,
                    "distance": dist,
                }
            )
        return hits
 
    def delete_by_document_id(self, document_id: str) -> None:
        """Remove all chunks belonging to a document."""
        collection = get_or_create_collection(self.collection_name)
        collection.delete(where={"document_id": document_id})
        logger.info("Deleted vectors for document", document_id=document_id)
 
    def count(self) -> int:
        """Return total number of stored vectors."""
        collection = get_or_create_collection(self.collection_name)
        return collection.count()
 