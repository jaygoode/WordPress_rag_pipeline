# tests/test_db.py

import pytest
from unittest.mock import Mock, MagicMock, patch, mock_open
from pathlib import Path
from agentic_rag.storage.db import get_connection, ensure_schema


class TestGetConnection:
    """Unit tests for get_connection"""
    
    @patch('agentic_rag.storage.db.psycopg.connect')
    def test_get_connection_success(self, mock_connect):
        """Test successful connection"""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        conn = get_connection()
        
        assert conn == mock_conn
        mock_connect.assert_called_once_with(
            "postgresql://rag:rag@localhost:5432/rag",
            autocommit=False
        )
    
    @patch('agentic_rag.storage.db.psycopg.connect')
    def test_get_connection_failure(self, mock_connect):
        """Test connection failure"""
        mock_connect.side_effect = Exception("Connection refused")
        
        with pytest.raises(Exception, match="Connection refused"):
            get_connection()
    
    @patch('agentic_rag.storage.db.psycopg.connect')
    def test_connection_string_format(self, mock_connect):
        """Test that connection string is correct"""
        get_connection()
        
        call_args = mock_connect.call_args
        conn_string = call_args[0][0]
        
        assert "postgresql://" in conn_string
        assert "rag:rag" in conn_string
        assert "localhost:5432" in conn_string
        assert "rag" in conn_string
    
    @patch('agentic_rag.storage.db.psycopg.connect')
    def test_autocommit_disabled(self, mock_connect):
        """Test that autocommit is disabled"""
        get_connection()
        
        call_args = mock_connect.call_args
        assert call_args[1]['autocommit'] == False


class TestEnsureSchema:
    """Unit tests for ensure_schema"""
    
    @patch('agentic_rag.storage.db.SCHEMA_PATH')
    def test_ensure_schema_reads_file(self, mock_schema_path):
        """Test that schema file is read"""
        mock_file_content = "CREATE TABLE test;"
        mock_file = mock_open(read_data=mock_file_content)
        mock_schema_path.open = mock_file
        
        # Use MagicMock for context manager support
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        ensure_schema(mock_conn)
        
        mock_file.assert_called_once_with(encoding="utf-8")
        mock_cursor.execute.assert_called_once_with(mock_file_content)
    
    @patch('agentic_rag.storage.db.SCHEMA_PATH')
    def test_ensure_schema_executes_ddl(self, mock_schema_path):
        """Test that DDL is executed"""
        ddl = "CREATE EXTENSION vector; CREATE TABLE documents;"
        mock_file = mock_open(read_data=ddl)
        mock_schema_path.open = mock_file
        
        # Use MagicMock for context manager support
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        ensure_schema(mock_conn)
        
        mock_cursor.execute.assert_called_once_with(ddl)
        mock_conn.commit.assert_called_once()
    
    @patch('agentic_rag.storage.db.SCHEMA_PATH')
    def test_ensure_schema_commits_transaction(self, mock_schema_path):
        """Test that transaction is committed"""
        mock_file = mock_open(read_data="CREATE TABLE test;")
        mock_schema_path.open = mock_file
        
        # Use MagicMock for context manager support
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        ensure_schema(mock_conn)
        
        assert mock_conn.commit.called
    
    @patch('agentic_rag.storage.db.SCHEMA_PATH')
    def test_ensure_schema_file_not_found(self, mock_schema_path):
        """Test handling when schema file doesn't exist"""
        mock_schema_path.open.side_effect = FileNotFoundError("schema.sql not found")
        
        mock_conn = MagicMock()
        
        with pytest.raises(FileNotFoundError):
            ensure_schema(mock_conn)
    
    @patch('agentic_rag.storage.db.SCHEMA_PATH')
    def test_ensure_schema_sql_error(self, mock_schema_path):
        """Test handling of SQL execution errors"""
        mock_file = mock_open(read_data="INVALID SQL;")
        mock_schema_path.open = mock_file
        
        # Use MagicMock for context manager support
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effe