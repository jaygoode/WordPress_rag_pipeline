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


NO RERANKER BEST RESULTS
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

cross-encoder/ms-marco-MiniLM-L-6-v2
2026-01-04 21:13:00,368 | agentic_rag.evaluation.runner | INFO | === Evaluation Results ===
2026-01-04 21:13:00,368 | agentic_rag.evaluation.runner | INFO | ==================================================
2026-01-04 21:13:00,368 | agentic_rag.evaluation.runner | INFO | Embedding model: all-mpnet-base-v2
2026-01-04 21:13:00,368 | agentic_rag.evaluation.runner | INFO | Cross-encoder model: cross-encoder/ms-marco-MiniLM-L-6-v2
2026-01-04 21:13:00,368 | agentic_rag.evaluation.runner | INFO | Retrieval strategy: retrieve 100 → rerank to 10
2026-01-04 21:13:00,368 | agentic_rag.evaluation.runner | INFO | Chunking - overlap: 50, max_tokens: 400
2026-01-04 21:13:00,368 | agentic_rag.evaluation.runner | INFO | --------------------------------------------------
2026-01-04 21:13:00,369 | agentic_rag.evaluation.runner | INFO | recall@5: 0.3840
2026-01-04 21:13:00,369 | agentic_rag.evaluation.runner | INFO | recall@10: 0.4612
2026-01-04 21:13:00,369 | agentic_rag.evaluation.runner | INFO | recall@20: 0.4612  SAME as k=10 because not enough chunks retrieved
2026-01-04 21:13:00,369 | agentic_rag.evaluation.runner | INFO | mrr: 0.3216
2026-01-04 21:13:00,369 | agentic_rag.evaluation.runner | INFO | ==================================================

cross-encoder/ms-marco-electra-base
2026-01-04 21:36:33,426 | agentic_rag.evaluation.runner | INFO | ==================================================
2026-01-04 21:36:33,426 | agentic_rag.evaluation.runner | INFO | === Evaluation Results ===
2026-01-04 21:36:33,426 | agentic_rag.evaluation.runner | INFO | ==================================================
2026-01-04 21:36:33,426 | agentic_rag.evaluation.runner | INFO | Embedding model: all-mpnet-base-v2
2026-01-04 21:36:33,426 | agentic_rag.evaluation.runner | INFO | Cross-encoder model: cross-encoder/ms-marco-electra-base
2026-01-04 21:36:33,426 | agentic_rag.evaluation.runner | INFO | Retrieval strategy: retrieve 100 → rerank to 20
2026-01-04 21:36:33,426 | agentic_rag.evaluation.runner | INFO | Chunking - overlap: 50, max_tokens: 400
2026-01-04 21:36:33,426 | agentic_rag.evaluation.runner | INFO | --------------------------------------------------
2026-01-04 21:36:33,426 | agentic_rag.evaluation.runner | INFO | recall@5: 0.3840
2026-01-04 21:36:33,426 | agentic_rag.evaluation.runner | INFO | recall@10: 0.4612
2026-01-04 21:36:33,427 | agentic_rag.evaluation.runner | INFO | recall@20: 0.5198
2026-01-04 21:36:33,427 | agentic_rag.evaluation.runner | INFO | mrr: 0.3257
2026-01-04 21:36:33,427 | agentic_rag.evaluation.runner | INFO | ==================================================

 cross-encoder/ms-marco-MiniLM-L-12-v2
2026-01-04 21:45:17,749 | agentic_rag.evaluation.runner | INFO | ==================================================
2026-01-04 21:45:17,749 | agentic_rag.evaluation.runner | INFO | === Evaluation Results ===
2026-01-04 21:45:17,749 | agentic_rag.evaluation.runner | INFO | ==================================================
2026-01-04 21:45:17,749 | agentic_rag.evaluation.runner | INFO | Embedding model: all-mpnet-base-v2
2026-01-04 21:45:17,749 | agentic_rag.evaluation.runner | INFO | Cross-encoder model: cross-encoder/ms-marco-MiniLM-L-12-v2
2026-01-04 21:45:17,749 | agentic_rag.evaluation.runner | INFO | Retrieval strategy: retrieve 100 → rerank to 20
2026-01-04 21:45:17,749 | agentic_rag.evaluation.runner | INFO | Chunking - overlap: 50, max_tokens: 400
2026-01-04 21:45:17,749 | agentic_rag.evaluation.runner | INFO | --------------------------------------------------
2026-01-04 21:45:17,749 | agentic_rag.evaluation.runner | INFO | recall@5: 0.3774
2026-01-04 21:45:17,749 | agentic_rag.evaluation.runner | INFO | recall@10: 0.4554
2026-01-04 21:45:17,749 | agentic_rag.evaluation.runner | INFO | recall@20: 0.5241
2026-01-04 21:45:17,749 | agentic_rag.evaluation.runner | INFO | mrr: 0.3178
2026-01-04 21:45:17,750 | agentic_rag.evaluation.runner | INFO | ==================================================


REALLY BAD - cross-encoder/qnli-distilroberta-base
2026-01-04 21:55:27,849 | agentic_rag.evaluation.runner | INFO | ==================================================
2026-01-04 21:55:27,849 | agentic_rag.evaluation.runner | INFO | === Evaluation Results ===
2026-01-04 21:55:27,849 | agentic_rag.evaluation.runner | INFO | ==================================================
2026-01-04 21:55:27,850 | agentic_rag.evaluation.runner | INFO | Embedding model: all-mpnet-base-v2
2026-01-04 21:55:27,850 | agentic_rag.evaluation.runner | INFO | Cross-encoder model: cross-encoder/qnli-distilroberta-base
2026-01-04 21:55:27,850 | agentic_rag.evaluation.runner | INFO | Retrieval strategy: retrieve 100 → rerank to 20
2026-01-04 21:55:27,850 | agentic_rag.evaluation.runner | INFO | Chunking - overlap: 50, max_tokens: 400
2026-01-04 21:55:27,850 | agentic_rag.evaluation.runner | INFO | --------------------------------------------------
2026-01-04 21:55:27,850 | agentic_rag.evaluation.runner | INFO | recall@5: 0.2126
2026-01-04 21:55:27,850 | agentic_rag.evaluation.runner | INFO | recall@10: 0.2889
2026-01-04 21:55:27,850 | agentic_rag.evaluation.runner | INFO | recall@20: 0.3658
2026-01-04 21:55:27,850 | agentic_rag.evaluation.runner | INFO | mrr: 0.1737
2026-01-04 21:55:27,851 | agentic_rag.evaluation.runner | INFO | ==================================================
2026-01-04 21:55:27,851 | agentic_rag.cli | INFO | Evaluation completed successfully