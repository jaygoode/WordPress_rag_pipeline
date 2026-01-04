# Setup
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .

# Infrastructure
make compose-up

# Run pipeline
make data              # Download dataset
make ingest            # Process & embed data
make evaluate          # Run evaluation
```

## Architecture

### Design Decisions

**Embedding Model:** `BAAI/bge-large-en-v1.5` (1024-dim)
- State-of-the-art open-source model
- Strong performance on technical Q&A (Recall@10: ~0.62)
- Better than all-mpnet-base-v2 (+20% improvement)

**Chunking Strategy:** 400 tokens, 50 overlap
- Balances context size with retrieval precision
- Overlap prevents information loss at boundaries
- Word-based splitting (simple, fast)

**Vector Store:** Postgres + pgvector
- HNSW indexing for fast cosine similarity search
- Dynamic schema handles different embedding dimensions
- Handles 48K+ documents efficiently

**Reranker:** Cross-encoder (optional, disabled by default)
- Tested ms-marco models but hurt performance (-11% on Recall@10)
- Domain mismatch: models trained on web search, not technical Q&A
- Embedding-only retrieval performs better for this dataset

### Data Flow
```
Raw Data → Clean → Chunk → Embed → pgvector → Retrieve → [Rerank] → Results

1. Ingestion: Loads JSONL, cleans HTML/markdown, chunks text
2. Embedding: Batch processing (64 docs/batch) on GPU
3. Storage: Upserts to Postgres with pgvector extension
4. Retrieval: Cosine similarity search, top-k results
5. Evaluation: Recall@5/10/20, MRR against qrels

### Testing
bash# Run all tests
pytest

# With coverage
pytest --cov=agentic_rag --cov-report=html

# Specific test suites
pytest tests/test_metrics.py -v
pytest tests/test_retriever.py -v
pytest tests/test_chunk_text.py -v
```

**Test Coverage:**
- ✅ Metrics (Recall@K, MRR)
- ✅ Chunking logic
- ✅ Retriever (mocked pgvector)
- ✅ Database utilities
- ✅ Reranker functionality

### Known Limitations

    1. Word-based chunking - Doesn't align with actual token counts (GPT tokenizer aware chunking would be better)
    2. No hybrid search - Pure embedding retrieval; adding BM25 could improve +10-15%
    3. Cross-encoder hurt performance - Domain mismatch with MS MARCO training data
    4. Single query embedding - No query expansion or multi-vector strategies
    5. Static chunking - No sentence-aware boundaries (may split mid-context)

### Dependencies
    Python 3.11+
    PostgreSQL 15+ with pgvector extension
    sentence-transformers (embeddings)
    psycopg 3.x (database)
    pytest (testing)
