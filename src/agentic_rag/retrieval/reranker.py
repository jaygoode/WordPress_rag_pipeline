from typing import Iterable, Sequence
from .base import BaseReranker
from .schemas import Query, RetrievedChunk
from agentic_rag.embeddings.model import embed_batch
import numpy as np

class EmbeddingReranker(BaseReranker):
    """Reranks candidates using cosine similarity with the query vector."""

    def rerank(
        self, query: Query, candidates: Iterable[RetrievedChunk], *, k: int = 5
    ) -> Sequence[RetrievedChunk]:

        # Embed the query
        query_vec = embed_batch([query.text])[0]

        # Compute similarity for each candidate
        for c in candidates:
            candidate_vec = np.array(c.metadata.get("embedding"))
            c.score = float(np.dot(query_vec, candidate_vec) / 
                            (np.linalg.norm(query_vec) * np.linalg.norm(candidate_vec)))

        # Return top-k
        return sorted(candidates, key=lambda x: x.score, reverse=True)[:k]
