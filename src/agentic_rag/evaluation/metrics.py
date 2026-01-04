from __future__ import annotations

import abc
from typing import Iterable, Sequence

from ..retrieval import Query, RetrievedChunk


class Metric(abc.ABC):
    name: str

    @abc.abstractmethod
    def compute(self, *, query: Query, retrieved: Sequence[RetrievedChunk], relevant: Iterable[str]) -> float:
        """Return the metric value for a single query."""


class MetricSuite:
    def __init__(self, metrics: Sequence[Metric]):
        self._metrics = metrics

    def evaluate(
        self,
        *,
        query: Query,
        retrieved_chunks: Sequence[RetrievedChunk],
        relevant_qrels: Iterable[str],
    ) -> dict[str, float]:
        return {metric.name: metric.compute(query=query, retrieved_chunks=retrieved_chunks, relevant_qrels=relevant_qrels) for metric in self._metrics}

class RecallAtK(Metric):
    def __init__(self, k: int):
        self.k = k
        self.name = f"recall@{k}"

    def compute(self, *, query, retrieved_chunks, relevant_qrels) -> float:
        if not relevant_qrels:
            print("No relevant qrels for query:", query)
            return 0.0
        retrieved_ids = {c.metadata["original_id"] for c in retrieved_chunks[: self.k]}
        relevant_ids = set(relevant_qrels)

        return len(retrieved_ids & relevant_ids) / len(relevant_ids)

class MRR(Metric):
    name = "mrr"

    def compute(self, *, query, retrieved_chunks, relevant_qrels) -> float:
        relevant_ids = set(relevant_qrels)
        for rank, chunk in enumerate(retrieved_chunks, start=1):
            if chunk.metadata["original_id"] in relevant_ids:
                return 1.0 / rank
        return 0.0
    
    