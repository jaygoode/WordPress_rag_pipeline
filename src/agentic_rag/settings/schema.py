from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VectorStoreConfig(BaseModel):
    implementation: Optional[str] = Field(
        default="pgvector",
        description="Name of the vector store backend (e.g., pgvector, chroma).",
    )
    collection: str = Field(default="wordpress", description="Vector collection name.")
    embedding_model: Optional[str] = Field(default="all-MiniLM-L6-v2")
    cross_encoder_model: Optional[str] = Field(default=None)
    top_k: int = Field(default=5, description="Number of top documents to retrieve.")
    
class EvaluationConfig(BaseModel):
    recall_at_k: int = Field(default=5, description="Number of top documents to consider for evaluation.")
    mrr: bool = Field(default=True, description="Whether to compute MRR metric.")
    ndcg: bool = Field(default=False, description="Whether to compute NDCG metric.")

class ChunkingConfig(BaseModel):
    max_tokens: int = Field(default=150, description="Maximum tokens per chunk.")
    overlap: int = Field(default=20, description="Number of overlapping tokens between chunks.")
    batch_size: int = Field(default=32, description="Batch size for embedding computation.")
    
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
    model_config = SettingsConfigDict(env_prefix="AGENTIC_RAG_", env_file=".env", extra="ignore")

    project_root: Path = Path(__file__).resolve().parents[2]
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    artifacts_dir: Path = Path("artifacts")
    
    dataset: DatasetConfig = DatasetConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    telemetry: TelemetryConfig = TelemetryConfig()
    
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
    return _settings
