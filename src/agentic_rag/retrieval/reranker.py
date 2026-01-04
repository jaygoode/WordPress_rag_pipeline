from typing import Iterable, Sequence, Optional
from sentence_transformers import CrossEncoder
from .base import BaseReranker
from .schemas import Query, RetrievedChunk
import logging

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    """Reranks candidates using a cross-encoder model for precise relevance scoring."""
    
    def __init__(self, model_name: Optional[str] = None):  # ← Changed: Optional[str] = None
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: Optional model name. If None, uses settings configuration.
        """
        from ..settings import get_settings
        settings = get_settings()
        
        # Use model from settings if not provided
        if model_name is None:
            model_name = settings.vector_store.cross_encoder_model
            if not model_name or model_name == "null":
                model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
                logger.warning(
                    f"No cross-encoder model in settings, using default: {model_name}"
                )
        
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
        
        logger.debug(
            f"Reranking {len(candidates_list)} candidates, returning top {k}",
            extra={"candidate_count": len(candidates_list), "k": k}
        )
        
        # Create query-document pairs for cross-encoder
        pairs = [[query.text, c.text] for c in candidates_list]
        
        # Get cross-encoder scores
        scores = self.model.predict(pairs)
        
        logger.debug(
            f"Cross-encoder scores - min: {scores.min():.4f}, max: {scores.max():.4f}, mean: {scores.mean():.4f}",
            extra={
                "min_score": float(scores.min()), 
                "max_score": float(scores.max()),
                "mean_score": float(scores.mean())
            }
        )
        
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
        
        if reranked_sorted:
            logger.debug(
                f"Reranking complete. Top score: {reranked_sorted[0].score:.4f}, "
                f"Bottom score: {reranked_sorted[-1].score:.4f}",
                extra={
                    "top_score": reranked_sorted[0].score,
                    "bottom_score": reranked_sorted[-1].score,
                    "returned_count": len(reranked_sorted)
                }
            )
        
        return reranked_sorted

