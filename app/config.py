"""
Centralized the configuration management via pydantic-settings.
All values loaded from the environment variable / .env file.

"""

from functools import lru_cache
from typing import Literal

from pydantic import Field,SecretStr
from pydantic_settings import BaseSettings,SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive= True,
    )
    
    
    
    APP_NAME: str = Field(default="EnterpriseRAG", description="Application name", )    
    
    APP_VERSION: str = Field( default="1.0.0", description="Application version", )
    
    DEBUG: bool = False
    
    LOG_LEVEL: Literal[ "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", ] = "INFO"
    
    # PostgreSQL
    
    POSTGRES_HOST: str = Field( default="postgres", min_length=1, )
    
    POSTGRES_PORT: int = Field( default=5432, ge=1, le=65535, )
    POSTGRES_DB: str = Field( default="enterprise_rag", min_length=1, )
    
    POSTGRES_USER: str = Field( default="raguser", min_length=1, )
    
    POSTGRES_PASSWORD: SecretStr = Field( default=SecretStr("ragpassword"), )
    
    DATABASE_URL: str = Field( default="postgresql+asyncpg://raguser:ragpassword@postgres:5432/enterprise_rag" )
    
    # ChromaDB
    CHROMA_HOST: str = Field( default="chromadb", min_length=1, )
    CHROMA_PORT: int = Field( default=8000, ge=1, le=65535, )
    
    CHROMA_COLLECTION_NAME: str = Field( default="documents", min_length=1, )
    
    
    # HuggingFace
    HUGGINGFACE_API_KEY: SecretStr | None = None
    HF_EMBEDDING_MODEL: str = Field( default="sentence-transformers/all-MiniLM-L6-v2" )
    HF_LLM_MODEL: str = Field( default="mistralai/Mistral-7B-Instruct-v0.2" )
    
    # RAG Configuration
    CHUNK_SIZE: int = Field( default=512, gt=0, le=4096, )
    CHUNK_OVERLAP: int = Field( default=64, ge=0, le=1024, )
    TOP_K_RESULTS: int = Field( default=5, ge=1, le=20, )
    
    MAX_TOKENS: int = Field( default=1024, gt=0, le=8192, )
    TEMPERATURE: float = Field( default=0.2, ge=0.0, le=2.0, )
    
    # File Upload
    UPLOAD_DIR: str = Field( default="uploads", )
    MAX_UPLOAD_SIZE_MB: int = Field( default=50, ge=1, le=500, )
    
    
    # Web Scraping
    WEB_TIMEOUT_SECONDS: int = Field( default=15, ge=1, le=120, )

@lru_cache 
def get_settings() -> Settings: 
    return Settings()

settings: Settings = get_settings()