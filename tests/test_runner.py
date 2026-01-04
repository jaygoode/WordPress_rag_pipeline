# tests/test_runner.py

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from agentic_rag.evaluation.runner import QrelsEvaluator
from agentic_rag.evaluation.metrics import MetricSuite, RecallAtK, MRR
from agentic_rag.retrieval import Query, RetrievedChunk


class TestQrelsEvaluator:
    @pytest.fixture
    def mock_data_dir(self, tmp_path):
        """Create temporary directory with mock data files"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create queries.jsonl
        queries_file = data_dir / "queries.jsonl"
        queries_file.write_text(
            '{"_id": "q1", "text": "How to add CSS?"}\n'
            '{"_id": "q2", "text": "WordPress plugins"}\n'
        )
        
        # Create qrels.jsonl
        qrels_file = data_dir / "qrels.jsonl"
        qrels_file.write_text(
            '{"query-id": "q1", "corpus-id": "doc1"}\n'
            '{"query-id": "q1", "corpus-id": "doc2"}\n'
            '{"query-id": "q2", "corpus-id": "doc3"}\n'
        )
        
        return data_dir

    @pytest.fixture
    def mock_retriever(self):
        """Mock retriever that returns fixed results"""
        retriever = Mock()
        retriever.search = Mock(return_value=[
            RetrievedChunk(chunk_id="doc1_0", text="", score=0.9, metadata={"original_id": "doc1"}),
            RetrievedChunk(chunk_id="doc2_0", text="", score=0.8, metadata={"original_id": "doc2"}),
        ])
        return retriever

    @pytest.fixture
    def metric_suite(self):
        """Create a simple metric suite"""
        return MetricSuite(metrics=[RecallAtK(k=5), MRR()])

    def test_load_queries(self, mock_data_dir, mock_retriever, metric_suite):
        """Test loading queries from JSONL file"""
        evaluator = QrelsEvaluator(
            retriever=mock_retriever,
            metrics=metric_suite,
            data_dir=mock_data_dir
        )
        
        assert len(evaluator.queries) == 2
        assert "q1" in evaluator.queries
        assert "q2" in evaluator.queries
        assert evaluator.queries["q1"].text == "How to add CSS?"
        assert evaluator.queries["q1"].metadata["query_id"] == "q1"

    def test_load_qrels(self, mock_data_dir, mock_retriever, metric_suite):
        """Test loading qrels from JSONL file"""
        evaluator = QrelsEvaluator(
            retriever=mock_retriever,
            metrics=metric_suite,
            data_dir=mock_data_dir
        )
        
        assert len(evaluator.qrels) == 2
        assert evaluator.qrels["q1"] == {"doc1", "doc2"}
        assert evaluator.qrels["q2"] == {"doc3"}

    def test_iter_queries(self, mock_data_dir, mock_retriever, metric_suite):
        """Test query iteration"""
        evaluator = QrelsEvaluator(
            retriever=mock_retriever,
            metrics=metric_suite,
            data_dir=mock_data_dir
        )
        
        queries = list(evaluator.iter_queries())
        assert len(queries) == 2
        assert all(isinstance(q, Query) for q in queries)

    @patch('agentic_rag.evaluation.runner.logger')
    def test_evaluate(self, mock_logger, mock_data_dir, mock_retriever, metric_suite):
        """Test full evaluation run"""
        evaluator = QrelsEvaluator(
            retriever=mock_retriever,
            metrics=metric_suite,
            data_dir=mock_data_dir
        )
        
        evaluator.evaluate()
        
        # Verify retriever was called for each query
        assert mock_retriever.search.call_count == 2
        
        # Verify logging occurred
        assert mock_logger.info.call_count >= 3  # Header + metrics

    @patch('agentic_rag.evaluation.runner.logger')
    def test_evaluate_logs_results(self, mock_logger, mock_data_dir, mock_retriever, metric_suite):
        """Test that evaluation logs results correctly"""
        evaluator = QrelsEvaluator(
            retriever=mock_retriever,
            metrics=metric_suite,
            data_dir=mock_data_dir
        )
        
        evaluator.evaluate()
        
        # Check that results were logged
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        log_output = " ".join(log_calls)
        
        assert "recall@5" in log_output
        assert "mrr" in log_output

    def test_evaluate_with_no_relevant_docs(self, mock_retriever, metric_suite, tmp_path):
        """Test evaluation when query has no relevant documents"""
        # Remove mock_data_dir from parameters ^^^
        
        # Create data with query that has no qrels
        data_dir = tmp_path / "data"
        data_dir.mkdir()  # Now it won't exist
        
        queries_file = data_dir / "queries.jsonl"
        queries_file.write_text('{"_id": "q_orphan", "text": "Orphan query"}\n')
        
        qrels_file = data_dir / "qrels.jsonl"
        qrels_file.write_text('')  # Empty qrels
        
        evaluator = QrelsEvaluator(
            retriever=mock_retriever,
            metrics=metric_suite,
            data_dir=data_dir
        )
        
        evaluator.evaluate()
        
        # Should complete without errors
        assert mock_retriever.search.call_count == 1