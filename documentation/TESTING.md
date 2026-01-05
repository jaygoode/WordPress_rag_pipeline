Developer notes for development and planning of testing, not exhaustive.

PLANNED TESTS:
    storage:
        db connection
        db storage?
    cli tests?

    data/ingestion
        -chunk text
        -cleaning OK
        -rag_pipeline
        
    embeddings - model
        embedding test?

    eval
        -metrics 
            -recall@k 
            -mmr 
            -evaluate func
        -runner
            -QreelsEvaluator
                -load queries
                -load qrels
                -iter queries
                -evaluate

    retrieval
    rerank
    retriever - search

# Run all tests
pytest tests/test_chunk_text.py -v

# Run with coverage
pytest tests/test_chunk_text.py --cov=agentic_rag.data.chunk_text --cov-report=html

# Run specific test class
pytest tests/test_chunk_text.py::TestChunkText -v

# Run edge cases only
pytest tests/test_chunk_text.py::TestChunkTextEdgeCases -v