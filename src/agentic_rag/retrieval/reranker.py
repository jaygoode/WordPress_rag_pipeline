from typing import Iterable, Sequence
from sentence_transformers import CrossEncoder
from .base import BaseReranker
from .schemas import Query, RetrievedChunk
import logging

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    """Reranks candidates using a cross-encoder model for precise relevance scoring."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the cross-encoder reranker.
        
        Args:
            model_name: Hugging Face model name for cross-encoder
        """
        logger.info(f"Loading cross-encoder model: {model_name}")
        self.model = CrossEncoder(model_name)
        logger.info("Cross-encoder model loaded successfully")
    
    def rerank(
        self, 
        query: Query, 
        candidates: Iterable[RetrievedChunk], 
        *, 
        k: int = 5
    ) -> Sequence[RetrievedChunk]:
        """
        Rerank candidates using cross-encoder.
        
        Args:
            query: The search query
            candidates: Initial retrieved candidates
            k: Number of top results to return
            
        Returns:
            Top-k reranked results
        """
        candidates_list = list(candidates)
        
        if not candidates_list:
            logger.warning("No candidates to rerank")
            return []
        
        logger.debug(f"Reranking {len(candidates_list)} candidates")
        
        # Create query-document pairs for cross-encoder
        pairs = [[query.text, c.text] for c in candidates_list]
        
        # Get cross-encoder scores
        scores = self.model.predict(pairs)
        
        # Create new RetrievedChunk objects with updated scores
        reranked = [
            RetrievedChunk(
                chunk_id=c.chunk_id,
                text=c.text,
                score=float(score),  # Cross-encoder score
                metadata=c.metadata
            )
            for c, score in zip(candidates_list, scores)
        ]
        
        # Sort by new scores and return top-k
        reranked_sorted = sorted(reranked, key=lambda x: x.score, reverse=True)[:k]
        
        logger.debug(
            f"Reranking complete. Top score: {reranked_sorted[0].score:.4f}"
            if reranked_sorted else "No results after reranking"
        )
        
        return reranked_sorted