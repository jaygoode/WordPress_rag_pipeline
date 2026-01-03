CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  chunk_id TEXT PRIMARY KEY,
  record_id TEXT NOT NULL,
  content TEXT NOT NULL,
  embedding VECTOR(384),
  metadata JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS documents_embedding_idx
ON documents
USING hnsw (embedding vector_cosine_ops);
