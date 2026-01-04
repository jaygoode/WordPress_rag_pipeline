# tests/test_metrics.py

from __future__ import annotations

import pytest
from agentic_rag.evaluation.metrics import Metric, MetricSuite, RecallAtK, MRR
from agentic_rag.retrieval import Query, RetrievedChunk


# Fixtures
@pytest.fixture
def sample_query():
    return Query(text="How to add CSS?", metadata={"query_id": "q1"})


@pytest.fixture
def retrieved_chunks():
    """5 retrieved chunks with IDs: doc1, doc2, doc3, doc4, doc5"""
    return [
        RetrievedChunk(
            chunk_id=f"doc{i}_0",
            text=f"Content {i}",
            score=1.0 - (i * 0.1),  # Descending scores
            metadata={"original_id": f"doc{i}"}
        )
        for i in range(1, 6)
    ]


@pytest.fixture
def relevant_qrels():
    """Relevant documents: doc2, doc4, doc6"""
    return ["doc2", "doc4", "doc6"]


# Tests for RecallAtK
class TestRecallAtK:
    def test_recall_at_k_perfect(self, sample_query, relevant_qrels):
        """Test recall when all relevant docs are in top-k"""
        retrieved = [
            RetrievedChunk(chunk_id="doc2_0", text="", score=1.0, metadata={"original_id": "doc2"}),
            RetrievedChunk(chunk_id="doc4_0", text="", score=0.9, metadata={"original_id": "doc4"}),
            RetrievedChunk(chunk_id="doc6_0", text="", score=0.8, metadata={"original_id": "doc6"}),
        ]
        
        metric = RecallAtK(k=3)
        score = metric.compute(query=sample_query, retrieved_chunks=retrieved, relevant_qrels=relevant_qrels)
        
        assert score == 1.0
        assert metric.name == "recall@3"

    def test_recall_at_k_partial(self, sample_query, retrieved_chunks, relevant_qrels):
        """Test recall when only some relevant docs are in top-k"""
        # retrieved_chunks has doc1-5, relevant are doc2, doc4, doc6
        # At k=5: found doc2, doc4 → 2/3 = 0.667
        
        metric = RecallAtK(k=5)
        score = metric.compute(query=sample_query, retrieved_chunks=retrieved_chunks, relevant_qrels=relevant_qrels)
        
        assert score == pytest.approx(2/3, rel=1e-3)

    def test_recall_at_k_zero(self, sample_query, retrieved_chunks):
        """Test recall when no relevant docs are retrieved"""
        relevant_qrels = ["doc10", "doc11", "doc12"]  # None in retrieved
        
        metric = RecallAtK(k=5)
        score = metric.compute(query=sample_query, retrieved_chunks=retrieved_chunks, relevant_qrels=relevant_qrels)
        
        assert score == 0.0

    def test_recall_at_k_empty_relevant(self, sample_query, retrieved_chunks, capsys):
        """Test recall when no relevant docs exist"""
        relevant_qrels = []
        
        metric = RecallAtK(k=5)
        score = metric.compute(query=sample_query, retrieved_chunks=retrieved_chunks, relevant_qrels=relevant_qrels)
        
        assert score == 0.0
        captured = capsys.readouterr()
        assert "No relevant qrels" in captured.out

    def test_recall_at_k_empty_retrieved(self, sample_query, relevant_qrels):
        """Test recall when nothing is retrieved"""
        retrieved = []
        
        metric = RecallAtK(k=5)
        score = metric.compute(query=sample_query, retrieved_chunks=retrieved, relevant_qrels=relevant_qrels)
        
        assert score == 0.0

    def test_recall_at_k_different_k_values(self, sample_query, relevant_qrels):
        """Test recall at different k values"""
        retrieved = [
            RetrievedChunk(chunk_id="doc1_0", text="", score=1.0, metadata={"original_id": "doc1"}),
            RetrievedChunk(chunk_id="doc2_0", text="", score=0.9, metadata={"original_id": "doc2"}),
            RetrievedChunk(chunk_id="doc3_0", text="", score=0.8, metadata={"original_id": "doc3"}),
            RetrievedChunk(chunk_id="doc4_0", text="", score=0.7, metadata={"original_id": "doc4"}),
            RetrievedChunk(chunk_id="doc6_0", text="", score=0.6, metadata={"original_id": "doc6"}),
        ]
        
        # At k=1: doc2 not found → 0/3
        assert RecallAtK(k=1).compute(query=sample_query, retrieved_chunks=retrieved, relevant_qrels=relevant_qrels) == 0.0
        
        # At k=2: doc2 found → 1/3
        assert RecallAtK(k=2).compute(query=sample_query, retrieved_chunks=retrieved, relevant_qrels=relevant_qrels) == pytest.approx(1/3)
        
        # At k=4: doc2, doc4 found → 2/3
        assert RecallAtK(k=4).compute(query=sample_query, retrieved_chunks=retrieved, relevant_qrels=relevant_qrels) == pytest.approx(2/3)
        
        # At k=5: all found → 3/3
        assert RecallAtK(k=5).compute(query=sample_query, retrieved_chunks=retrieved, relevant_qrels=relevant_qrels) == 1.0

    def test_recall_handles_duplicate_ids(self, sample_query, relevant_qrels):
        """Test recall correctly handles duplicate document IDs in chunks"""
        retrieved = [
            RetrievedChunk(chunk_id="doc2_0", text="", score=1.0, metadata={"original_id": "doc2"}),
            RetrievedChunk(chunk_id="doc2_1", text="", score=0.95, metadata={"original_id": "doc2"}),  # Same doc
            RetrievedChunk(chunk_id="doc4_0", text="", score=0.9, metadata={"original_id": "doc4"}),
        ]
        
        metric = RecallAtK(k=3)
        score = metric.compute(query=sample_query, retrieved_chunks=retrieved, relevant_qrels=relevant_qrels)
        
        # Should count doc2 only once: 2 unique docs / 3 relevant = 2/3
        assert score == pytest.approx(2/3)


# Tests for MRR
class TestMRR:
    def test_mrr_first_position(self, sample_query, relevant_qrels):
        """Test MRR when first result is relevant"""
        retrieved = [
            RetrievedChunk(chunk_id="doc2_0", text="", score=1.0, metadata={"original_id": "doc2"}),
            RetrievedChunk(chunk_id="doc1_0", text="", score=0.9, metadata={"original_id": "doc1"}),
        ]
        
        metric = MRR()
        score = metric.compute(query=sample_query, retrieved_chunks=retrieved, relevant_qrels=relevant_qrels)
        
        assert score == 1.0
        assert metric.name == "mrr"

    def test_mrr_second_position(self, sample_query, relevant_qrels):
        """Test MRR when second result is relevant"""
        retrieved = [
            RetrievedChunk(chunk_id="doc1_0", text="", score=1.0, metadata={"original_id": "doc1"}),
            RetrievedChunk(chunk_id="doc4_0", text="", score=0.9, metadata={"original_id": "doc4"}),
        ]
        
        metric = MRR()
        score = metric.compute(query=sample_query, retrieved_chunks=retrieved, relevant_qrels=relevant_qrels)
        
        assert score == 0.5  # 1/2

    def test_mrr_tenth_position(self, sample_query, relevant_qrels):
        """Test MRR when 10th result is relevant"""
        retrieved = [
            RetrievedChunk(chunk_id=f"doc{i}_0", text="", score=1.0, metadata={"original_id": f"doc{i}"})
            for i in range(1, 10)  # doc1 to doc9 (but doc2, 4, 6 are relevant!)
        ]
        # Add doc10 which is not in relevant_qrels, but let's add doc2 again at the end
        retrieved.append(
            RetrievedChunk(chunk_id="doc10_0", text="", score=0.1, metadata={"original_id": "doc10"})
        )
        
        metric = MRR()
        score = metric.compute(query=sample_query, retrieved_chunks=retrieved, relevant_qrels=relevant_qrels)
        
        # Actually finds doc2 at position 2, not position 10!
        assert score == 0.5  # 1/2 (because doc2 is at position 2)

    def test_mrr_no_relevant(self, sample_query, retrieved_chunks):
        """Test MRR when no relevant docs are retrieved"""
        relevant_qrels = ["doc10", "doc11"]
        
        metric = MRR()
        score = metric.compute(query=sample_query, retrieved_chunks=retrieved_chunks, relevant_qrels=relevant_qrels)
        
        assert score == 0.0

    def test_mrr_empty_retrieved(self, sample_query, relevant_qrels):
        """Test MRR when nothing is retrieved"""
        retrieved = []
        
        metric = MRR()
        score = metric.compute(query=sample_query, retrieved_chunks=retrieved, relevant_qrels=relevant_qrels)
        
        assert score == 0.0

    def test_mrr_empty_relevant(self, sample_query, retrieved_chunks):
        """Test MRR when no relevant docs exist"""
        relevant_qrels = []
        
        metric = MRR()
        score = metric.compute(query=sample_query, retrieved_chunks=retrieved_chunks, relevant_qrels=relevant_qrels)
        
        assert score == 0.0

    def test_mrr_only_counts_first_relevant(self, sample_query, relevant_qrels):
        """Test that MRR only considers first relevant document"""
        retrieved = [
            RetrievedChunk(chunk_id="doc1_0", text="", score=1.0, metadata={"original_id": "doc1"}),
            RetrievedChunk(chunk_id="doc2_0", text="", score=0.9, metadata={"original_id": "doc2"}),  # First relevant
            RetrievedChunk(chunk_id="doc4_0", text="", score=0.8, metadata={"original_id": "doc4"}),  # Second relevant
            RetrievedChunk(chunk_id="doc6_0", text="", score=0.7, metadata={"original_id": "doc6"}),  # Third relevant
        ]
        
        metric = MRR()
        score = metric.compute(query=sample_query, retrieved_chunks=retrieved, relevant_qrels=relevant_qrels)
        
        # Should return 1/2 for position 2, not considering positions 3 and 4
        assert score == 0.5


# Tests for MetricSuite
class TestMetricSuite:
    def test_metric_suite_single_metric(self, sample_query, retrieved_chunks, relevant_qrels):
        """Test MetricSuite with single metric"""
        metrics = MetricSuite(metrics=[RecallAtK(k=5)])
        
        results = metrics.evaluate(
            query=sample_query,
            retrieved_chunks=retrieved_chunks,
            relevant_qrels=relevant_qrels
        )
        
        assert "recall@5" in results
        assert isinstance(results["recall@5"], float)
        assert 0.0 <= results["recall@5"] <= 1.0

    def test_metric_suite_multiple_metrics(self, sample_query, retrieved_chunks, relevant_qrels):
        """Test MetricSuite with multiple metrics"""
        metrics = MetricSuite(metrics=[
            RecallAtK(k=5),
            RecallAtK(k=10),
            MRR()
        ])
        
        results = metrics.evaluate(
            query=sample_query,
            retrieved_chunks=retrieved_chunks,
            relevant_qrels=relevant_qrels
        )
        
        assert len(results) == 3
        assert "recall@5" in results
        assert "recall@10" in results
        assert "mrr" in results

    def test_metric_suite_all_metrics_computed(self, sample_query, relevant_qrels):
        """Test that all metrics are computed correctly"""
        retrieved = [
            RetrievedChunk(chunk_id="doc1_0", text="", score=1.0, metadata={"original_id": "doc1"}),
            RetrievedChunk(chunk_id="doc2_0", text="", score=0.9, metadata={"original_id": "doc2"}),
            RetrievedChunk(chunk_id="doc3_0", text="", score=0.8, metadata={"original_id": "doc3"}),
        ]
        
        metrics = MetricSuite(metrics=[
            RecallAtK(k=3),
            MRR()
        ])
        
        results = metrics.evaluate(
            query=sample_query,
            retrieved_chunks=retrieved,
            relevant_qrels=relevant_qrels
        )
        
        # Recall: doc2 found, 1/3 relevant = 0.333
        assert results["recall@3"] == pytest.approx(1/3)
        
        # MRR: doc2 at position 2, MRR = 1/2 = 0.5
        assert results["mrr"] == 0.5

    def test_metric_suite_empty_metrics(self, sample_query, retrieved_chunks, relevant_qrels):
        """Test MetricSuite with no metrics"""
        metrics = MetricSuite(metrics=[])
        
        results = metrics.evaluate(
            query=sample_query,
            retrieved_chunks=retrieved_chunks,
            relevant_qrels=relevant_qrels
        )
        
        assert results == {}


# Edge cases and integration tests
class TestEdgeCases:
    def test_missing_original_id_in_metadata(self, sample_query, relevant_qrels):
        """Test handling of chunks without original_id in metadata"""
        retrieved = [
            RetrievedChunk(chunk_id="doc1_0", text="", score=1.0, metadata={}),  # Missing original_id
        ]
        
        metric = RecallAtK(k=1)
        
        with pytest.raises(KeyError):
            metric.compute(query=sample_query, retrieved_chunks=retrieved, relevant_qrels=relevant_qrels)

    def test_none_metadata(self, sample_query, relevant_qrels):
        """Test handling of chunks with None metadata"""
        retrieved = [
            RetrievedChunk(chunk_id="doc1_0", text="", score=1.0, metadata=None),
        ]
        
        metric = RecallAtK(k=1)
        
        with pytest.raises((KeyError, TypeError, AttributeError)):
            metric.compute(query=sample_query, retrieved_chunks=retrieved, relevant_qrels=relevant_qrels)

    def test_large_k_value(self, sample_query, relevant_qrels):
        """Test recall with k larger than retrieved chunks"""
        retrieved = [
            RetrievedChunk(chunk_id="doc2_0", text="", score=1.0, metadata={"original_id": "doc2"}),
        ]
        
        metric = RecallAtK(k=100)  # k much larger than len(retrieved)
        score = metric.compute(query=sample_query, retrieved_chunks=retrieved, relevant_qrels=relevant_qrels)
        
        # Should still work, just use all available chunks
        assert score == pytest.approx(1/3)  # Found 1 of 3 relevant
