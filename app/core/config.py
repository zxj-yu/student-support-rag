"""Application configuration, loaded from environment variables."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Qdrant — two ways to connect:
    # 1. Local / docker-compose: host + port (the default below)
    # 2. Qdrant Cloud: set QDRANT_URL (https://xxx.cloud.qdrant.io:6333)
    #    and QDRANT_API_KEY; when qdrant_url is set it takes precedence.
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    collection_name: str = "student_kb"

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384  # all-MiniLM-L6-v2 output size

    # Retrieval
    top_k: int = 4
    max_question_chars: int = 1000  # reject pasted-novel inputs early

    # LLM
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-6"
    max_tokens: int = 1024


settings = Settings()
