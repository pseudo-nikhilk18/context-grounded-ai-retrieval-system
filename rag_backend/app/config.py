"""
Configuration for RAG Backend
(Simple + Stable Gemini Setup)
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ===== Gemini =====
    GEMINI_API_KEY: str
    GEMINI_LLM_MODEL: str = "gemini-1.5-flash"
    GEMINI_EMBEDDING_MODEL: str = "models/embedding-001"

    # ===== Chroma =====
    CHROMA_PERSIST_DIR: str = "chroma_db"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()