from pathlib import Path
from dotenv import load_dotenv
import os

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# === Embedding Config ===
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# === LLM Config ===
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local")
LLM_MODEL = os.getenv("LLM_MODEL", "=LLaMA-2-13B-chat")


# === Chunking / Retrieval ===
TOP_K = os.getenv("TOP_K", 5)
CHUNK_MAX_TOKENS = os.getenv("CHUNK_MAX_TOKENS", 150)
CHUNK_OVERLAP = os.getenv("CHUNK_OVERLAP", 20)
BATCH_SIZE = os.getenv("BATCH_SIZE", 32)

# === PostgreSQL / pgvector ===
PGHOST = os.getenv("PGHOST", "localhost")
PGPORT = os.getenv("PGPORT", 5432)
PGDATABASE = os.getenv("PGDATABASE", "rag")
PGUSER = os.getenv("PGUSER", "rag")
PGPASSWORD = os.getenv("PGPASSWORD", "rag")
RESET_DB = os.getenv("RESET_DB", "false").lower() == "true"

# local paths for debugging / artifacts
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output"))