
from .base import BaseRetriever, BaseReranker
from .schemas import Query, RetrievedChunk
from ..storage.db import get_connection
from typing import Sequence
from agentic_rag.embeddings.model import embed_batch
import json

class PgVectorRetriever(BaseRetriever):
    """Retrieves chunks from Postgres using pgvector cosine similarity."""

    def search(self, query: Query, *, k: int = 5) -> Sequence[RetrievedChunk]: #TODO: make k configurable
        # Step 1: Embed the query
        query_vector = embed_batch([query.text])[0]  # returns 1 vector

        # Step 2: Query the database
        sql = """
        SELECT chunk_id, content, metadata,
               embedding <-> %s AS score
        FROM documents
        ORDER BY score
        LIMIT %s;
        """

        with get_connection() as conn, conn.cursor() as cur:
            cur.execute(sql, (query_vector, k))
            rows = cur.fetchall()

        # Step 3: Return as RetrievedChunk
        results = [
            RetrievedChunk(
                chunk_id=row[0],
                text=row[1],
                score=row[3],
                metadata=json.loads(row[2]) if row[2] else None,
            )
            for row in rows
        ]
        return results
