from __future__ import annotations

import abc
from typing import Iterable, Optional
from agentic_rag.retrieval import Query, RetrievedChunk
from collections import defaultdict
from pathlib import Path

from ..retrieval.base import BaseReranker, BaseRetriever
from .metrics import MetricSuite
from ..utils.io import read_jsonl
from ..settings import get_settings
import logging

logger = logging.getLogger(__name__)
settings = get_settings()


class BaseEvaluator(abc.ABC):
    @abc.abstractmethod
    def iter_queries(self) -> Iterable[Query]:
        """Yield evaluation queries."""

    @abc.abstractmethod
    def evaluate(self) -> None:
        """Run the evaluation suite."""


class QrelsEvaluator(BaseEvaluator):
    def __init__(
        self, 
        *, 
        retriever: BaseRetriever, 
        reranker: Optional[BaseReranker] = None, 
        metrics: MetricSuite, 
        data_dir: Path
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.metrics = metrics
        self.data_dir = data_dir
        self.queries = self._load_queries()
        self.qrels = self._load_qrels()
        
        if self.reranker:
            logger.info("Evaluator initialized WITH reranker")
        else:
            logger.info("Evaluator initialized WITHOUT reranker")

    def _load_queries(self):
        logger.info("Loading queries")
        queries = {}
        for obj in read_jsonl(self.data_dir / "queries.jsonl"):
            queries[obj["_id"]] = Query(
                text=obj["text"],
                metadata={"query_id": obj["_id"]}
            )
        
        logger.info(
            f"Loaded {len(queries)} queries",
            extra={"query_count": len(queries)}
        )
        return queries
        
    def _load_qrels(self):
        logger.info("Loading qrels (relevance judgments)")
        qrels = defaultdict(set)
        
        for obj in read_jsonl(self.data_dir / "qrels.jsonl"):
            qrels[obj["query-id"]].add(obj["corpus-id"])
            
        logger.info(
            f"Loaded qrels for {len(qrels)} queries",
            extra={"qrels_count": len(qrels)}
        )
        
        return qrels
    
    def iter_queries(self) -> Iterable[Query]:
        return self.queries.values()

    def evaluate(self) -> None:
        """Run evaluation with optional reranking."""
        all_scores = defaultdict(list)
        total_queries = len(self.queries)
        
        logger.info(f"Starting evaluation on {total_queries} queries")

        for idx, query in enumerate(self.iter_queries(), 1):
            query_id = query.metadata.get("query_id") if query.metadata else None
            
            # Progress logging every 100 queries
            if idx % 100 == 0:
                logger.info(
                    f"Progress: {idx}/{total_queries} queries evaluated",
                    extra={"progress": idx, "total": total_queries}
                )
            
            # Step 1: Initial retrieval
            # Retrieve more if using reranker, otherwise retrieve final amount
            retrieval_k = settings.vector_store.retrieval_k if self.reranker else settings.vector_store.top_k
            retrieved_chunks: list[RetrievedChunk] = self.retriever.search(
                query, 
                k=retrieval_k
            )
            
            logger.debug(
                f"Query {query_id}: Retrieved {len(retrieved_chunks)} candidates",
                extra={"query_id": query_id, "retrieved_count": len(retrieved_chunks)}
            )
            
            # Step 2: Rerank if reranker is available
            if self.reranker:
                reranked_chunks = self.reranker.rerank(
                    query, 
                    retrieved_chunks, 
                    k=settings.vector_store.reranker_top_k
                )
                
                logger.debug(
                    f"Query {query_id}: Reranked to {len(reranked_chunks)} results",
                    extra={
                        "query_id": query_id, 
                        "reranked_count": len(reranked_chunks),
                        "score_change": (
                            reranked_chunks[0].score - retrieved_chunks[0].score
                            if reranked_chunks and retrieved_chunks else 0
                        )
                    }
                )
                
                final_chunks = reranked_chunks
            else:
                final_chunks = retrieved_chunks[:settings.vector_store.top_k]
            
            # Step 3: Evaluate against qrels
            relevant_qrels = self.qrels.get(query_id, set())

            scores = self.metrics.evaluate(
                query=query,
                retrieved_chunks=final_chunks,
                relevant_qrels=relevant_qrels,
            )

            # Accumulate scores
            for name, value in scores.items():
                all_scores[name].append(value)
        
        # Log final results
        logger.info("=" * 50)
        logger.info("=== Evaluation Results ===")
        logger.info("=" * 50)
        logger.info(f"Embedding model: {settings.vector_store.embedding_model}")
        
        if self.reranker:
            logger.info(f"Cross-encoder model: {settings.vector_store.cross_encoder_model}")
            logger.info(f"Retrieval strategy: retrieve {settings.vector_store.retrieval_k} → rerank to {settings.vector_store.reranker_top_k}")
        else:
            logger.info("No reranker used")
        
        logger.info(f"Chunking - overlap: {settings.chunking.overlap}, max_tokens: {settings.chunking.max_tokens}")
        logger.info("-" * 50)
        
        for name, values in all_scores.items():
            mean = sum(values) / len(values) if values else 0.0
            logger.info(f"{name}: {mean:.4f}")
        
        logger.info("=" * 50)