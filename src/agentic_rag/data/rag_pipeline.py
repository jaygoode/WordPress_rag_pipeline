from __future__ import annotations
from pathlib import Path
from typing import Iterable, List
from datetime import datetime
import json
from .pipeline import BaseIngestionPipeline
from .types import Chunk, RawRecord
from agentic_rag.utils.io import read_jsonl
from agentic_rag.embeddings.model import embed_batch
from agentic_rag.storage.db import get_connection, ensure_schema
from .cleaning import clean_text
from .chunk_text import chunk_text
from ..settings import get_settings
import logging

logger = logging.getLogger(__name__)
settings = get_settings()

class WordPressIngestionPipeline(BaseIngestionPipeline):
    """Ingestion pipeline for WordPress export XML files."""
    
    def load_raw(self, raw_dir: Path) -> Iterable[RawRecord]:
        path = raw_dir / "corpus.jsonl"
        logger.info(f"Loading raw data from {path}")
        rows = []
        for obj in read_jsonl(path): 
            rows.append(
                RawRecord(
                    identifier=obj["_id"],
                    title=obj.get("title", ""),
                    body=obj.get("text", ""),
                    metadata={"source": "cqadupstack-wordpress"},
                )
            )
        logger.info(
                f"Loaded {len(rows)} raw records",
                extra={"record_count": len(rows), "source_file": str(path)}
            )
        return sorted(rows, key=lambda r: r.identifier)

    def transform(self, records: Iterable[RawRecord]) -> Iterable[Chunk]:
        logger.info("Transforming raw records into chunks")
        for record in records:
            text = clean_text(record.title + "\n\n" + record.body)
            for i, chunk_text_str in enumerate(chunk_text(text)):
                yield Chunk(
                    chunk_id=f"{record.identifier}_{i}",
                    record_id=record.identifier,
                    text=chunk_text_str,
                    metadata={"original_id": record.identifier, "chunk_index": i}, #make sure this original id addition doesnt break anything
                )
        logger.info("Transformation complete")

    def persist(self, chunks: List[Chunk], output_dir: Path) -> None:
        """Persist a batch of chunks to Postgres + pgvector and write JSONL for inspection."""
        if not chunks:
            logger.warning("No chunks to persist")
            return

        logger.debug(f"Persisting batch of {len(chunks)} chunks")
        
        texts = [c.text for c in chunks]
        embeddings = embed_batch(texts)  # returns list of vectors
        
        logger.debug(f"Generated {len(embeddings)} embeddings")
        assert len(embeddings) == len(chunks)

        with get_connection() as conn:
            ensure_schema(conn)
            sql_query = """
                INSERT INTO documents (
                    chunk_id,
                    record_id,
                    content,
                    embedding,
                    metadata,
                    created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata;
            """

            now = datetime.utcnow()
            rows = [
                (
                    c.chunk_id,
                    c.record_id,
                    c.text,
                    embedding,                
                    json.dumps(c.metadata),
                    c.created_at or now,
                )
                for c, embedding in zip(chunks, embeddings)
            ]

            with conn.cursor() as cur:
                cur.executemany(sql_query, rows)
            conn.commit()
        
        logger.debug(
            f"Inserted {len(chunks)} chunks to database",
            extra={"chunk_count": len(chunks)}
        )
        
        # write batch to JSONL for inspection
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "chunks.jsonl"
        with out_path.open("a", encoding="utf-8") as f: 
            for c in chunks:
                f.write(json.dumps({
                    "chunk_id": c.chunk_id,
                    "record_id": c.record_id,
                    "text": c.text,
                    "metadata": c.metadata,
                }) + "\n")

    def run(self, raw_dir: Path, output_dir: Path) -> None:
        """Run the ingestion pipeline with batching for embeddings and DB inserts."""
        
        logger.info("Starting WordPress ingestion pipeline")
        
        records = self.load_raw(raw_dir)
        batch: List[Chunk] = []
        total_chunks = 0
        
        #TODO is raw record a list or dict? do i need to clean all values? checkthis
        for chunk in self.transform(records):
            batch.append(chunk)
            if len(batch) >= settings.chunking.batch_size:
                self.persist(batch, output_dir)
                total_chunks += len(batch)
                logger.info(
                    f"Progress: {total_chunks} chunks processed",
                    extra={"total_chunks": total_chunks}
                )
                batch.clear()

        if batch:
            self.persist(batch, output_dir)
            total_chunks += len(batch)
            
        logger.info(
            f"Ingestion pipeline completed",
            extra={"total_chunks": total_chunks, "output_dir": str(output_dir)}
        )