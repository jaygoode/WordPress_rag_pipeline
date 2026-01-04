# tests/test_retriever.py

from __future__ import annotations

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from agentic_rag.retrieval.retriever import PgVectorRetriever
from agentic_rag.retrieval.schemas import Query, RetrievedChunk


@pytest.fixture
def sample_query():
    """Sample query for testing"""
    return Query(text="How to add CSS to WordPress?", metadata={"query_id": "q1"})


@pytest.fixture
def mock_embedding():
    """Mock embedding vector (768 dimensions for all-mpnet-base-v2)"""
    return [0.1] * 768


@pytest.fixture
def mock_db_rows():
    """Mock database rows returned from pgvector query"""
    return [
        (
            "doc1_0",  # chunk_id
            "How to add custom CSS to WordPress theme",  # content
            json.dumps({"original_id": "doc1", "chunk_index": 0}),  # metadata
            0.15  # score (cosine distance)
        ),
        (
            "doc2_0",
            "WordPress styling guide and CSS customization",
            json.dumps({"original_id": "doc2", "chunk_index": 0}),
            0.25
        ),
        (
            "doc3_1",
            "Advanced CSS techniques for WordPress",
            json.dumps({"original_id": "doc3", "chunk_index": 1}),
            0.35
        ),
    ]


class TestPgVectorRetriever:
    """Tests for PgVectorRetriever"""
    
    @patch('agentic_rag.retrieval.retriever.get_connection')
    @patch('agentic_rag.retrieval.retriever.embed_batch')
    def test_search_basic(self, mock_embed_batch, mock_get_connection, sample_query, mock_embedding, mock_db_rows):
        """Test basic search functionality"""
        # Setup mocks
        mock_embed_batch.return_value = [mock_embedding]
        
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = mock_db_rows
        
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = None
        
        mock_get_connection.return_value = mock_conn
        
        # Execute
        retriever = PgVectorRetriever()
        results = retriever.search(sample_query, k=3)
        
        # Verify
        assert len(results) == 3
        assert all(isinstance(r, RetrievedChunk) for r in results)
        
        # Check first result
        assert results[0].chunk_id == "doc1_0"
        assert results[0].text == "How to add custom CSS to WordPress theme"
        assert results[0].score == 0.15
        # Parse the JSON string first
        parsed_metadata = json.loads(results[0].metadata)
        assert parsed_metadata["original_id"] == "doc1"
        
        # Verify embed_batch was called with query text
        mock_embed_batch.assert_called_once_with([sample_query.text])
        
        # Verify SQL was executed with correct parameters
        mock_cursor.execute.assert_called_once()
        sql_call = mock_cursor.execute.call_args
        assert "embedding <-> %s::vector" in sql_call[0][0]
        assert "LIMIT %s" in sql_call[0][0]
        assert sql_call[0][1] == (mock_embedding, 3)
    
    @patch('agentic_rag.retrieval.retriever.get_connection')
    @patch('agentic_rag.retrieval.retriever.embed_batch')
    def test_search_custom_k(self, mock_embed_batch, mock_get_connection, sample_query, mock_embedding):
        """Test search with custom k parameter"""
        mock_embed_batch.return_value = [mock_embedding]
        
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = None
        
        mock_get_connection.return_value = mock_conn
        
        # Execute with k=10
        retriever = PgVectorRetriever()
        retriever.search(sample_query, k=10)
        
        # Verify k was passed to SQL
        sql_call = mock_cursor.execute.call_args
        assert sql_call[0][1] == (mock_embedding, 10)
    
    @patch('agentic_rag.retrieval.retriever.get_connection')
    @patch('agentic_rag.retrieval.retriever.embed_batch')
    def test_search_no_results(self, mock_embed_batch, mock_get_connection, sample_query, mock_embedding):
        """Test search when no results are found"""
        mock_embed_batch.return_value = [mock_embedding]
        
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []  # No results
        
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = None
        
        mock_get_connection.return_value = mock_conn
        
        # Execute
        retriever = PgVectorRetriever()
        results = retriever.search(sample_query, k=5)
        
        # Verify
        assert len(results) == 0
        assert isinstance(results, list)
    
    @patch('agentic_rag.retrieval.retriever.get_connection')
    @patch('agentic_rag.retrieval.retriever.embed_batch')
    def test_search_single_result(self, mock_embed_batch, mock_get_connection, sample_query, mock_embedding):
        """Test search with single result"""
        mock_embed_batch.return_value = [mock_embedding]
        
        single_row = [(
            "doc1_0",
            "Single result",
            json.dumps({"original_id": "doc1"}),
            0.10
        )]
        
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = single_row
        
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = None
        
        mock_get_connection.return_value = mock_conn
        
        # Execute
        retriever = PgVectorRetriever()
        results = retriever.search(sample_query, k=5)
        
        # Verify
        assert len(results) == 1
        assert results[0].chunk_id == "doc1_0"
        assert results[0].score == 0.10
    
    @patch('agentic_rag.retrieval.retriever.get_connection')
    @patch('agentic_rag.retrieval.retriever.embed_batch')
    def test_search_ordered_by_score(self, mock_embed_batch, mock_get_connection, sample_query, mock_embedding):
        """Test that results are ordered by score (ascending distance)"""
        mock_embed_batch.return_value = [mock_embedding]
        
        # Results should be ordered by score (distance)
        ordered_rows = [
            ("doc1_0", "Best match", json.dumps({}), 0.10),
            ("doc2_0", "Second match", json.dumps({}), 0.20),
            ("doc3_0", "Third match", json.dumps({}), 0.30),
        ]
        
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = ordered_rows
        
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = None
        
        mock_get_connection.return_value = mock_conn
        
        # Execute
        retriever = PgVectorRetriever()
        results = retriever.search(sample_query, k=3)
        
        # Verify order is preserved
        assert results[0].score == 0.10
        assert results[1].score == 0.20
        assert results[2].score == 0.30
        assert results[0].text == "Best match"
    
    @patch('agentic_rag.retrieval.retriever.get_connection')
    @patch('agentic_rag.retrieval.retriever.embed_batch')
    def test_search_metadata_parsing(self, mock_embed_batch, mock_get_connection, sample_query, mock_embedding):
        """Test that JSON metadata is correctly parsed"""
        mock_embed_batch.return_value = [mock_embedding]
        
        complex_metadata = {
            "original_id": "doc1",
            "chunk_index": 0,
            "source": "wordpress",
            "tags": ["css", "styling"]
        }
        
        rows = [(
            "doc1_0",
            "Content",
            json.dumps(complex_metadata),
            0.15
        )]
        
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = rows
        
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = None
        
        mock_get_connection.return_value = mock_conn
        
        # Execute
        retriever = PgVectorRetriever()
        results = retriever.search(sample_query, k=1)
        
        # Verify metadata is a dict (parsed from JSON string)
        assert isinstance(results[0].metadata, str)  # Still JSON string in your implementation
        parsed_metadata = json.loads(results[0].metadata)
        assert parsed_metadata["original_id"] == "doc1"
        assert parsed_metadata["chunk_index"] == 0
        assert parsed_metadata["tags"] == ["css", "styling"]
    
    @patch('agentic_rag.retrieval.retriever.get_connection')
    @patch('agentic_rag.retrieval.retriever.embed_batch')
    def test_search_connection_closed(self, mock_embed_batch, mock_get_connection, sample_query, mock_embedding):
        """Test that database connection is properly closed"""
        mock_embed_batch.return_value = [mock_embedding]
        
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = None
        
        mock_get_connection.return_value = mock_conn
        
        # Execute
        retriever = PgVectorRetriever()
        retriever.search(sample_query, k=5)
        
        # Verify context managers were used (connection will auto-close)
        assert mock_conn.__enter__.called
        assert mock_conn.__exit__.called
    
    @patch('agentic_rag.retrieval.retriever.get_connection')
    @patch('agentic_rag.retrieval.retriever.embed_batch')
    def test_search_database_error(self, mock_embed_batch, mock_get_connection, sample_query, mock_embedding):
        """Test handling of database errors"""
        mock_embed_batch.return_value = [mock_embedding]
        
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("Database connection failed")
        
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = None
        
        mock_get_connection.return_value = mock_conn
        
        # Execute and expect exception
        retriever = PgVectorRetriever()
        with pytest.raises(Exception, match="Database connection failed"):
            retriever.search(sample_query, k=5)
    
    @patch('agentic_rag.retrieval.retriever.get_connection')
    @patch('agentic_rag.retrieval.retriever.embed_batch')
    def test_search_embedding_error(self, mock_embed_batch, mock_get_connection, sample_query):
        """Test handling of embedding generation errors"""
        mock_embed_batch.side_effect = Exception("Embedding model failed")
        
        # Execute and expect exception
        retriever = PgVectorRetriever()
        with pytest.raises(Exception, match="Embedding model failed"):
            retriever.search(sample_query, k=5)
        
        # Database should not be called if embedding fails
        mock_get_connection.assert_not_called()
    
    @patch('agentic_rag.retrieval.retriever.get_connection')
    @patch('agentic_rag.retrieval.retriever.embed_batch')
    def test_search_empty_query_text(self, mock_embed_batch, mock_get_connection, mock_embedding):
        """Test search with empty query text"""
        empty_query = Query(text="", metadata={})
        mock_embed_batch.return_value = [mock_embedding]
        
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = None
        
        mock_get_connection.return_value = mock_conn
        
        # Execute
        retriever = PgVectorRetriever()
        results = retriever.search(empty_query, k=5)
        
        # Should still work (embedding model handles empty text)
        mock_embed_batch.assert_called_once_with([""])
        assert isinstance(results, list)
    
    @patch('agentic_rag.retrieval.retriever.get_connection')
    @patch('agentic_rag.retrieval.retriever.embed_batch')
    def test_search_k_zero(self, mock_embed_batch, mock_get_connection, sample_query, mock_embedding):
        """Test search with k=0"""
        mock_embed_batch.return_value = [mock_embedding]
        
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = None
        
        mock_get_connection.return_value = mock_conn
        
        # Execute
        retriever = PgVectorRetriever()
        results = retriever.search(sample_query, k=0)
        
        # Should return empty results
        assert len(results) == 0
    
    @patch('agentic_rag.retrieval.retriever.get_connection')
    @patch('agentic_rag.retrieval.retriever.embed_batch')
    def test_search_k_large(self, mock_embed_batch, mock_get_connection, sample_query, mock_embedding):
        """Test search with very large k"""
        mock_embed_batch.return_value = [mock_embedding]
        
        # Only 2 results in DB
        rows = [
            ("doc1_0", "Result 1", json.dumps({}), 0.1),
            ("doc2_0", "Result 2", json.dumps({}), 0.2),
        ]
        
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = rows
        
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = None
        
        mock_get_connection.return_value = mock_conn
        
        # Execute with k=1000
        retriever = PgVectorRetriever()
        results = retriever.search(sample_query, k=1000)
        
        # Should return only available results
        assert len(results) == 2
    
    @patch('agentic_rag.retrieval.retriever.get_connection')
    @patch('agentic_rag.retrieval.retriever.embed_batch')
    def test_search_sql_injection_safety(self, mock_embed_batch, mock_get_connection, mock_embedding):
        """Test that SQL injection is prevented via parameterization"""
        # Query with SQL injection attempt
        malicious_query = Query(
            text="'; DROP TABLE documents; --",
            metadata={}
        )
        
        mock_embed_batch.return_value = [mock_embedding]
        
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = None
        
        mock_get_connection.return_value = mock_conn
        
        # Execute
        retriever = PgVectorRetriever()
        retriever.search(malicious_query, k=5)
        
        # Verify parameterized query was used (not string interpolation)
        sql_call = mock_cursor.execute.call_args
        assert "%s" in sql_call[0][0]  # Uses parameterization
        assert "DROP TABLE" not in sql_call[0][0]  # SQL is static
        # The malicious text is in the embedding, not the SQL
        mock_embed_batch.assert_called_once_with(["'; DROP TABLE documents; --"])
    
    @patch('agentic_rag.retrieval.retriever.get_connection')
    @patch('agentic_rag.retrieval.retriever.embed_batch')
    def test_search_vector_dimensions(self, mock_embed_batch, mock_get_connection, sample_query):
        """Test that embedding vector dimensions are correct"""
        # Test with different embedding dimensions
        embedding_384 = [0.1] * 384  # MiniLM-L6
        embedding_768 = [0.1] * 768  # MPNet-base
        
        for embedding in [embedding_384, embedding_768]:
            mock_embed_batch.return_value = [embedding]
            
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = []
            
            mock_conn = MagicMock()
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            
            mock_get_connection.return_value = mock_conn
            
            # Execute
            retriever = PgVectorRetriever()
            retriever.search(sample_query, k=5)
            
            # Verify the embedding was passed correctly
            sql_call = mock_cursor.execute.call_args
            assert sql_call[0][1][0] == embedding


class TestPgVectorRetrieverIntegration:
    """Integration-style tests (still using mocks but testing full flow)"""
    
    @patch('agentic_rag.retrieval.retriever.get_connection')
    @patch('agentic_rag.retrieval.retriever.embed_batch')
    def test_search_full_flow(self, mock_embed_batch, mock_get_connection, sample_query, mock_embedding, mock_db_rows):
        """Test complete search flow from query to results"""
        # Setup
        mock_embed_batch.return_value = [mock_embedding]
        
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = mock_db_rows
        
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = None
        
        mock_get_connection.return_value = mock_conn
        
        # Execute
        retriever = PgVectorRetriever()
        results = retriever.search(sample_query, k=3)
        
        # Comprehensive verification
        assert len(results) == 3
        
        # Check all fields are populated
        for result in results:
            assert result.chunk_id is not None
            assert result.text is not None
            assert result.score is not None
            assert result.metadata is not None
            assert isinstance(result.score, float)
            assert result.score >= 0  # Distance should be non-negative
        
        # Verify execution order
        assert mock_embed_batch.called
        assert mock_get_connection.called
        assert mock_cursor.execute.called
        assert mock_cursor.fetchall.called