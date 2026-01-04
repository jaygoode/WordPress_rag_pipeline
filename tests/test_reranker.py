import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from agentic_rag.retrieval.reranker import CrossEncoderReranker
from agentic_rag.retrieval.schemas import Query, RetrievedChunk


@pytest.fixture
def sample_query():
    return Query(text="How to add CSS?", metadata={"query_id": "q1"})


@pytest.fixture
def sample_candidates():
    return [
        RetrievedChunk(
            chunk_id="doc1_0",
            text="WordPress CSS customization guide",
            score=0.85,
            metadata={"original_id": "doc1"}
        ),
        RetrievedChunk(
            chunk_id="doc2_0",
            text="JavaScript tutorial for beginners",
            score=0.75,
            metadata={"original_id": "doc2"}
        ),
        RetrievedChunk(
            chunk_id="doc3_0",
            text="How to add custom CSS to WordPress theme",
            score=0.70,
            metadata={"original_id": "doc3"}
        ),
    ]


class TestCrossEncoderReranker:
    @patch('agentic_rag.retrieval.reranker.CrossEncoder')
    def test_rerank_basic(self, mock_cross_encoder_class, sample_query, sample_candidates):
        """Test basic reranking functionality"""
        # Mock the cross-encoder model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.6, 0.3, 0.9])  # doc3 should rank first
        mock_cross_encoder_class.return_value = mock_model
        
        # Initialize reranker
        reranker = CrossEncoderReranker()
        
        # Rerank
        results = reranker.rerank(sample_query, sample_candidates, k=3)
        
        # Verify
        assert len(results) == 3
        assert results[0].chunk_id == "doc3_0"  # Highest score (0.9)
        assert results[1].chunk_id == "doc1_0"  # Second (0.6)
        assert results[2].chunk_id == "doc2_0"  # Third (0.3)
        
        # Verify scores were updated
        assert results[0].score == 0.9
        assert results[1].score == 0.6
        assert results[2].score == 0.3
    
    @patch('agentic_rag.retrieval.reranker.CrossEncoder')
    def test_rerank_top_k(self, mock_cross_encoder_class, sample_query, sample_candidates):
        """Test that only top-k results are returned"""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.6, 0.3, 0.9])
        mock_cross_encoder_class.return_value = mock_model
        
        reranker = CrossEncoderReranker()
        results = reranker.rerank(sample_query, sample_candidates, k=2)
        
        # Only top 2
        assert len(results) == 2
        assert results[0].chunk_id == "doc3_0"
        assert results[1].chunk_id == "doc1_0"
    
    @patch('agentic_rag.retrieval.reranker.CrossEncoder')
    def test_rerank_empty_candidates(self, mock_cross_encoder_class, sample_query):
        """Test reranking with no candidates"""
        mock_model = MagicMock()
        mock_cross_encoder_class.return_value = mock_model
        
        reranker = CrossEncoderReranker()
        results = reranker.rerank(sample_query, [], k=5)
        
        assert len(results) == 0
        mock_model.predict.assert_not_called()
    
    @patch('agentic_rag.retrieval.reranker.CrossEncoder')
    def test_rerank_creates_new_chunks(self, mock_cross_encoder_class, sample_query, sample_candidates):
        """Test that reranker creates new chunks, doesn't modify originals"""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.6, 0.3, 0.9])
        mock_cross_encoder_class.return_value = mock_model
        
        original_scores = [c.score for c in sample_candidates]
        
        reranker = CrossEncoderReranker()
        results = reranker.rerank(sample_query, sample_candidates, k=3)
        
        # Original candidates should be unchanged
        for i, c in enumerate(sample_candidates):
            assert c.score == original_scores[i]
        
        # Results should be different objects
        assert results[0] is not sample_candidates[0]
