from __future__ import annotations

import abc
from typing import Iterable
from agentic_rag.retrieval import Query, RetrievedChunk
from collections import defaultdict
from pathlib import Path
from .metrics import MetricSuite
from ..utils.io import read_jsonl


class BaseEvaluator(abc.ABC):
    @abc.abstractmethod
    def iter_queries(self) -> Iterable[Query]:
        """Yield evaluation queries."""

    @abc.abstractmethod
    def evaluate(self) -> None:
        """Run the evaluation suite."""

class QrelsEvaluator(BaseEvaluator):
    def __init__(self, *, retriever, metrics: MetricSuite, data_dir: Path):
        self.retriever = retriever
        self.metrics = metrics
        self.data_dir = data_dir
        self.queries = self._load_queries()
        self.qrels = self._load_qrels()

    def _load_queries(self):
        queries = {}
        for obj in read_jsonl(self.data_dir / "queries.jsonl"):
            queries[obj["_id"]] = Query(
            text=obj["text"],
            metadata={"query_id": obj["_id"]}
        )
            
        return queries
        
    def _load_qrels(self):
        qrels = defaultdict(set)
        for obj in read_jsonl(self.data_dir / "qrels.jsonl"):
            qrels[obj["query-id"]].add(obj["corpus-id"])
            
        return qrels
    
    def iter_queries(self) -> Iterable[Query]:
        return self.queries.values()

    def evaluate(self) -> None:
        all_scores = defaultdict(list)

        for query in self.iter_queries(): 
            retrieved: list[RetrievedChunk] = self.retriever.search(query, k=5) #TODO make k configurable

            query_id = query.metadata.get("query_id") if query.metadata else None
            relevant = self.qrels.get(query_id, set())
            #TODO
            scores = self.metrics.evaluate(
                query=query,
                retrieved=retrieved,
                relevant=relevant,
            )

            for name, value in scores.items():
                all_scores[name].append(value)

        for name, values in all_scores.items():
            mean = sum(values) / len(values) if values else 0.0
            print(f"{name}: {mean:.4f}")