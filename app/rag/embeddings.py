"""
Huggingface embedding model wrapper.
uses sentence-transformers locally (no API keys required for embeddigns.) 
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer
from app.config import settings
from app.utils.logger import get_logger


logger = get_logger(__name__)

@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Return a cached sentencetransormer instance.
    The model is downloaded onces and reused all the requests."""
    
    logger.info("Loading the embedding model",model = settings.HF_EMBEDDING_MODEL)
    
    model = SentenceTransformer(
        settings.HF_EMBEDDING_MODEL,
        device="cpu"
    )

        
    logger.info("Embedding model loaded successfully.")
    
    
    return model


def embed_texts(texts:list[str]) -> list[list[float]]:
    """
    embed a list of texts and returns the vector representes.
    
    args:
        list of texts strings to embed
        
    return :
             return list of embedding vectors
    """
    
    model = get_embedding_model()
    return model.encode(
        texts,
        normalize_embeddings=True
    ).tolist()


def embed_query(query:str) -> list[float]:
    """
    Embed a single query of string.
    Args:
        the query texts
        
    returns :
            Embeddings vectors.
    """
    
    model = get_embedding_model()
    
    return model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

