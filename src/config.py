from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # LLM
    openai_api_key: str = Field(default="")
    anthropic_api_key: str = Field(default="")
    grader_model: str = Field(default="gpt-4o-mini")
    generator_model: str = Field(default="claude-sonnet-4-6")
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dimensions: int = Field(default=1536)

    # LangSmith
    langchain_tracing_v2: bool = Field(default=False)
    langchain_api_key: str = Field(default="")
    langchain_project: str = Field(default="agentic-rag")

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: str = Field(default="")
    qdrant_collection: str = Field(default="documents")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379")
    redis_ttl_seconds: int = Field(default=3600)

    # Tavily
    tavily_api_key: str = Field(default="")

    # RAG behaviour
    retrieval_top_k: int = Field(default=5)
    max_rewrite_attempts: int = Field(default=2)

    # App
    app_env: str = Field(default="development")
    log_level: str = Field(default="INFO")
    metrics_port: int = Field(default=9090)


settings = Settings()
