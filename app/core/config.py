import os
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Core Application Settings
    PROJECT_NAME: str = "ERA System"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    API_V1_STR: str = "/api/v1"

    # LLM & Embedding Defaults
    USE_LOCAL_MODELS: bool = True
    LLM_PROVIDER: Literal["ollama", "openai"] = "ollama"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"

    OPENAI_API_KEY: str = "sk-proj-mock-key-for-development"
    OPENAI_MODEL: str = "gpt-4o-mini"

    DENSE_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Storage Databases
    VECTOR_STORE_TYPE: Literal["chroma", "qdrant"] = "chroma"
    CHROMA_PERSIST_DIRECTORY: str = "./data/chroma"
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333

    # Caching
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    SEMANTIC_CACHE_THRESHOLD: float = 0.88

    # Reranking Engine
    FLASHRANK_MODEL: str = "ms-marco-MiniLM-L-12-v2"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


settings = Settings()