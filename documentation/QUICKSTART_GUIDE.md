# Setup
the pipeline core startup functionality has NOT been modified, and works as the template intends. see original README.md and MakeFile.

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .

# Infrastructure
make compose-up

# Run pipeline
make data              # Download dataset
make ingest            # Process & embed data
make evaluate          # Run retrieval + evaluation
```

## Architecture

### Design Decisions

    ## Project Restrictions

        As the project has not specified design choices, I will set these myself based on my personal restrictions:
            1. all design choices must be free to use. which means local hosting of models.
            2. must be able to run on CPU: AMD Ryzen 7 9800X3D
                - IMPROVEMENT - run with CUDA and RTX3090 GPU for massive processing increase.
                    - currently technical issues preventing CUDA activation; currently investigating.
            3. storage limitation on locally hosted models, about 150GB of storage available.

    Below we have the best results I could get from my limited testing of the retrieval. We can see that Interestingly enough the best results I got were from using no reranker. when testing with the reranker using multiple different free to use huggingface models, we always lose accuracy by at least a few percentage points.


        2026-01-04 22:01:30,273 | agentic_rag.evaluation.runner | INFO | ==================================================
        2026-01-04 22:01:30,273 | agentic_rag.evaluation.runner | INFO | === Evaluation Results ===
        2026-01-04 22:01:30,273 | agentic_rag.evaluation.runner | INFO | ==================================================
        2026-01-04 22:01:30,273 | agentic_rag.evaluation.runner | INFO | Embedding model: all-mpnet-base-v2
        2026-01-04 22:01:30,273 | agentic_rag.evaluation.runner | INFO | No reranker used
        2026-01-04 22:01:30,273 | agentic_rag.evaluation.runner | INFO | Chunking - overlap: 50, max_tokens: 400
        2026-01-04 22:01:30,273 | agentic_rag.evaluation.runner | INFO | --------------------------------------------------
        2026-01-04 22:01:30,273 | agentic_rag.evaluation.runner | INFO | recall@5: 0.4036
        2026-01-04 22:01:30,273 | agentic_rag.evaluation.runner | INFO | recall@10: 0.5189
        2026-01-04 22:01:30,273 | agentic_rag.evaluation.runner | INFO | recall@20: 0.5833
        2026-01-04 22:01:30,274 | agentic_rag.evaluation.runner | INFO | mrr: 0.3226
        2026-01-04 22:01:30,274 | agentic_rag.evaluation.runner | INFO | ==================================================

    **Embedding Model:** `all-mpnet-base-v2` (768-dim)
        - REASON: 
            - Balanced model, not too big for running on my hardware. 
            - Solid performance on technical Q&A (Recall@10: ~0.52)
        - IMPROVEMENT: 
            - a bigger model here could improve results significantly, approx 20% - `BAAI/bge-large-en-v1.5` (1024-dim) -> needs same dim embedding model - adds to improvement
            - testing multiple models specialized in QA and text comprehension recommended
        - if lower tier CPU in use: 
            -for maximum speed, switch to all-MiniLM-L6-v2 embedding model and ms-marco-MiniLM-L-6-v2 reranking encoder in .env

    **Chunking Strategy:** 400 tokens, 50 overlap
        - Balances context size with retrieval precision
        - Overlap prevents information loss at boundaries
        - Word-based splitting (simple, fast)
        - IMPROVEMENT:
            - values should be tweaked further to maximize results.

    **Vector Store:** Postgres + pgvector
        - HNSW indexing for fast cosine similarity search
        - Dynamic schema handles different embedding dimensions
        - Handles 48K+ documents efficiently

    **Reranker:** Cross-encoder (optional, disabled by default)
        - Tested multiple models (all listed in .env) but hurt performance, in METRICS.md are saved metrics comparing results of different rerankers (around -10% on Recall@10)
        CAUSE OF ISSUES:
            - Domain mismatch: models trained on web search, not technical Q&A
            - Embedding-only retrieval performs better for this dataset
        

### Data Flow
```
Raw Data → Clean → Chunk → Embed → pgvector → Retrieve → [Rerank] → Results

1. Ingestion: Loads JSONL, cleans HTML/markdown, chunks text
2. Embedding: Batch processing (64 docs/batch) on GPU/CPU
3. Storage: Upserts to Postgres with pgvector extension
4. Retrieval: Cosine similarity search, top-k results
5. Evaluation: Recall@5/10/20, MRR against qrels

### Testing
-not 100% test coverage at the moment, will implement in future.
 
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
