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
        retrieved: Sequence[RetrievedChunk],
        relevant: Iterable[str],
    ) -> dict[str, float]:
        return {metric.name: metric.compute(query=query, retrieved=retrieved, relevant=relevant) for metric in self._metrics}

class RecallAtK(Metric):
    def __init__(self, k: int):
        self.k = k
        self.name = f"recall@{k}"

    def compute(self, *, query, retrieved, relevant) -> float:
        if not relevant:
            return 0.0

        retrieved_ids = {c.metadata["original_id"] for c in retrieved[: self.k]}
        relevant_ids = set(relevant)

        return len(retrieved_ids & relevant_ids) / len(relevant_ids)
    
class MRR(Metric):
    name = "mrr"

    def compute(self, *, query, retrieved, relevant) -> float:
        relevant_ids = set(relevant)
        for rank, chunk in enumerate(retrieved, start=1):
            if chunk.metadata["original_id"] in relevant_ids:
                return 1.0 / rank
        return 0.0
    
    