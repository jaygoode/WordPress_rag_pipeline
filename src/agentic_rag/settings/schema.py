from __future__ import annotations

from pathlib import Path
from typing import Optional
import os

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import logging
logger = logging.getLogger(__name__)

class VectorStoreConfig(BaseModel):
    implementation: Optional[str] = Field(default="pgvector")
    collection: str = Field(default="wordpress")
    embedding_model: Optional[str] = Field(default="all-MiniLM-L6-v2")
    cross_encoder_model: Optional[str] = Field(default=None)
    top_k: int = Field(default=5)
    retrieval_k: int = Field(
        default=100,
        description="Number of candidates to retrieve before reranking"
    )
    reranker_top_k: int = Field(
        default=10,
        description="Number of results to return after reranking"
    )


class EvaluationConfig(BaseModel):
    recall_at_k: list[int] = Field(default_factory=lambda: [5, 10])
    mrr: bool = Field(default=True)
    ndcg: bool = Field(default=False)
    
    @field_validator('recall_at_k', mode='before')
    @classmethod
    def parse_recall_at_k(cls, v):
        if isinstance(v, str):
            return [int(x.strip()) for x in v.split(',')]
        return v


class ChunkingConfig(BaseModel):
    max_tokens: int = Field(default=150)
    overlap: int = Field(default=20)
    batch_size: int = Field(default=32)


class TelemetryConfig(BaseModel):
    enabled: bool = Field(default=True)
    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=False)


class DatasetConfig(BaseModel):
    name: str = Field(default="mteb/cqadupstack-wordpress")
    corpus_filename: str = Field(default="corpus.jsonl")
    queries_filename: str = Field(default="queries.jsonl")
    qrels_filename: str = Field(default="qrels.jsonl")


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENTIC_RAG_",
        env_file=str(Path(__file__).resolve().parents[2] / ".env"),
        env_file_encoding='utf-8',
        env_nested_delimiter='__',
        extra="ignore"
    )

    project_root: Path = Path(__file__).resolve().parents[2]
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    artifacts_dir: Path = Path("artifacts")
    
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    
    ingestion_class: Optional[str] = None
    agent_controller_class: Optional[str] = None
    evaluator_class: Optional[str] = None
    retriever_class: Optional[str] = None
    reranker_class: Optional[str] = None


_settings: Optional[AppSettings] = None


def get_settings() -> AppSettings:
    global _settings
    if _settings is None:
        _settings = AppSettings()
        
        logger.info(f"\n=== Loaded Settings ===")
        logger.info(f"retriever_class: {_settings.retriever_class}")
        logger.info(f"vector_store.top_k: {_settings.vector_store.top_k}")
        logger.info(f"vector_store.embedding_model: {_settings.vector_store.embedding_model}")
        logger.info("==================\n")
    
    return _settings
