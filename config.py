from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    # LLM
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    llm_provider: Literal["openai", "anthropic"] = "openai"
    llm_model: str = "gpt-4o"

    # Embeddings
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Vector Store
    vector_store: Literal["faiss", "pgvector"] = "faiss"
    faiss_index_path: str = "./data/faiss_index"
    postgres_url: str = "postgresql://postgres:password@localhost:5432/ragdb"

    # RAG Config
    retrieval_top_k: int = 10
    rerank_top_k: int = 4
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Evaluation
    eval_sample_size: int = 20
    ragas_llm_model: str = "gpt-4o"

    # Observability
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    enable_tracing: bool = False

    # App
    app_title: str = "Production RAG System"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

