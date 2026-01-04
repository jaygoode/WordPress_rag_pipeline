MD file containing some of the results of evaluations, to compare while tweaking and changing solutions to improve results. this data is always getting logged in runs.

2026-01-04 16:09:37,417 | sentence_transformers.SentenceTransformer | INFO | Load pretrained SentenceTransformer: all-MiniLM-L6-v2
reranker in use = FALSE
recall@5: 0.3579
recall@10: 0.4344
recall@20: 0.5220
mrr: 0.2967

______________________________________________________________
reranker in use = FALSE
AGENTIC_RAG_CHUNKING__MAX_TOKENS=150
AGENTIC_RAG_CHUNKING__OVERLAP=20
AGENTIC_RAG_CHUNKING__BATCH_SIZE=32
2026-01-04 18:15:16,599 | sentence_transformers.SentenceTransformer | INFO | Load pretrained SentenceTransformer: all-mpnet-base-v2
recall@5: 0.3845
recall@10: 0.4979
recall@20: 0.5650
mrr: 0.3152

2026-01-04 18:39:20,415 | agentic_rag.evaluation.runner | INFO | === Evaluation Results ===
2026-01-04 18:39:20,416 | agentic_rag.evaluation.runner | INFO | embedding model: all-mpnet-base-v2
2026-01-04 18:39:20,416 | agentic_rag.evaluation.runner | INFO | chunking.overlap: 50
2026-01-04 18:39:20,416 | agentic_rag.evaluation.runner | INFO | chunking.max_tokens: 400
2026-01-04 18:39:20,416 | agentic_rag.evaluation.runner | INFO | recall@5: 0.4036
2026-01-04 18:39:20,416 | agentic_rag.evaluation.runner | INFO | recall@10: 0.5189
2026-01-04 18:39:20,416 | agentic_rag.evaluation.runner | INFO | recall@20: 0.5833
2026-01-04 18:39:20,416 | agentic_rag.evaluation.runner | INFO | mrr: 0.3248

