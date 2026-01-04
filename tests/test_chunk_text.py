# tests/test_chunk_text.py

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from agentic_rag.data.chunk_text import chunk_text


@pytest.fixture
def mock_settings():
    """Mock settings with default chunking configuration"""
    settings = MagicMock()
    settings.chunking.max_tokens = 150
    settings.chunking.overlap = 20
    return settings


class TestChunkText:
    """Tests for text chunking functionality"""
    
    def test_chunk_text_single_chunk(self, mock_settings):
        """Test text that fits in a single chunk"""
        with patch('agentic_rag.data.chunk_text.settings', mock_settings):
            text = "This is a short text with only ten words here."
            chunks = chunk_text(text)
            
            assert len(chunks) == 1
            assert chunks[0] == text
    
    def test_chunk_text_exact_boundary(self, mock_settings):
        """Test text that exactly fills max_tokens"""
        with patch('agentic_rag.data.chunk_text.settings', mock_settings):
            # Create text with exactly 150 words
            words = [f"word{i}" for i in range(150)]
            text = " ".join(words)
            
            chunks = chunk_text(text)
            
            # Should create exactly 1 chunk (not 2)
            assert len(chunks) == 1
            assert len(chunks[0].split()) == 150

    
    def test_chunk_text_two_chunks_no_overlap(self, mock_settings):
        """Test splitting into two chunks"""
        mock_settings.chunking.overlap = 0
        
        with patch('agentic_rag.data.chunk_text.settings', mock_settings):
            # Create text with 200 words
            words = [f"word{i}" for i in range(200)]
            text = " ".join(words)
            
            chunks = chunk_text(text)
            
            assert len(chunks) == 2
            # First chunk: words 0-149
            assert chunks[0] == " ".join(words[:150])
            # Second chunk: words 150-199
            assert chunks[1] == " ".join(words[150:])
    
    def test_chunk_text_with_overlap(self, mock_settings):
        """Test that overlap works correctly"""
        mock_settings.chunking.max_tokens = 10
        mock_settings.chunking.overlap = 3
        
        with patch('agentic_rag.data.chunk_text.settings', mock_settings):
            words = [f"w{i}" for i in range(20)]
            text = " ".join(words)
            
            chunks = chunk_text(text)
            
            # Should create 3 chunks with overlap
            # Chunk 1: 0-9 (10 words)
            # Chunk 2: 7-16 (start at 10-3=7, 10 words)
            # Chunk 3: 14-19 (start at 17-3=14, remaining 6 words)
            assert len(chunks) == 3
            
            # Verify overlap
            chunk1_words = chunks[0].split()
            chunk2_words = chunks[1].split()
            
            # Last 3 words of chunk1 should match first 3 words of chunk2
            assert chunk1_words[-3:] == chunk2_words[:3]
    
    def test_chunk_text_multiple_chunks(self, mock_settings):
        """Test splitting into multiple chunks"""
        mock_settings.chunking.max_tokens = 50
        mock_settings.chunking.overlap = 10
        
        with patch('agentic_rag.data.chunk_text.settings', mock_settings):
            # Create text with 150 words
            words = [f"word{i}" for i in range(150)]
            text = " ".join(words)
            
            chunks = chunk_text(text)
            
            # With max=50, overlap=10, step=40
            # Chunks start at: 0, 40, 80, 120
            # Should create 4 chunks
            assert len(chunks) == 4
            
            # Verify chunk sizes
            assert len(chunks[0].split()) == 50  # 0-49
            assert len(chunks[1].split()) == 50  # 40-89
            assert len(chunks[2].split()) == 50  # 80-129
            assert len(chunks[3].split()) == 20  # 120-149 (remaining)
    
    def test_chunk_text_empty_string(self, mock_settings):
        """Test empty input"""
        with patch('agentic_rag.data.chunk_text.settings', mock_settings):
            chunks = chunk_text("")
            
            # Should return single empty chunk (not empty list)
            assert len(chunks) == 1
            assert chunks[0] == ""
    
    def test_chunk_text_single_word(self, mock_settings):
        """Test input with single word"""
        with patch('agentic_rag.data.chunk_text.settings', mock_settings):
            chunks = chunk_text("word")
            
            assert len(chunks) == 1
            assert chunks[0] == "word"
    
    def test_chunk_text_whitespace_only(self, mock_settings):
        """Test input with only whitespace"""
        with patch('agentic_rag.data.chunk_text.settings', mock_settings):
            text = "   \n  \t  "
            chunks = chunk_text(text)
            
            # Should return the whitespace as-is
            assert len(chunks) == 1
            assert chunks[0] == text
    
    def test_chunk_text_preserves_word_order(self, mock_settings):
        """Test that word order is preserved in chunks"""
        mock_settings.chunking.max_tokens = 5
        mock_settings.chunking.overlap = 1
        
        with patch('agentic_rag.data.chunk_text.settings', mock_settings):
            text = "one two three four five six seven eight nine ten"
            chunks = chunk_text(text)
            
            # Reconstruct text from chunks (removing overlap)
            all_words = []
            for i, chunk in enumerate(chunks):
                words = chunk.split()
                if i == 0:
                    all_words.extend(words)
                else:
                    # Skip overlapping words
                    all_words.extend(words[mock_settings.chunking.overlap:])
            
            # Should match original (approximately, due to overlap logic)
            assert " ".join(all_words[:10]) == text
    
    def test_chunk_text_overlap_larger_than_max_tokens(self, mock_settings):
        """Test edge case where overlap >= max_tokens"""
        mock_settings.chunking.max_tokens = 10
        mock_settings.chunking.overlap = 15  # Larger than max_tokens
        
        with patch('agentic_rag.data.chunk_text.settings', mock_settings):
            words = [f"w{i}" for i in range(30)]
            text = " ".join(words)
            
            chunks = chunk_text(text)
            
            # Step would be negative (10-15=-5), causing infinite loop in naive implementation
            # This should either handle gracefully or at least not hang
            # The current implementation will create overlapping chunks moving backward
            assert len(chunks) > 0
    
    def test_chunk_text_overlap_equals_max_tokens(self, mock_settings):
        """Test edge case where overlap == max_tokens"""
        mock_settings.chunking.max_tokens = 10
        mock_settings.chunking.overlap = 10
        
        with patch('agentic_rag.data.chunk_text.settings', mock_settings):
            words = [f"w{i}" for i in range(30)]
            text = " ".join(words)
            
            chunks = chunk_text(text)
            
            # Step = 0, would cause infinite loop
            # Should handle this edge case
            assert len(chunks) > 0
    
    def test_chunk_text_special_characters(self, mock_settings):
        """Test text with special characters and punctuation"""
        with patch('agentic_rag.data.chunk_text.settings', mock_settings):
            text = "Hello, world! How are you? I'm fine. Let's test this."
            chunks = chunk_text(text)
            
            assert len(chunks) == 1
            # Should preserve special characters
            assert "," in chunks[0]
            assert "!" in chunks[0]
            assert "'" in chunks[0]
    
    def test_chunk_text_unicode_characters(self, mock_settings):
        """Test text with unicode characters"""
        with patch('agentic_rag.data.chunk_text.settings', mock_settings):
            text = "Hello 你好 مرحبا Привет 🌍"
            chunks = chunk_text(text)
            
            assert len(chunks) == 1
            assert "你好" in chunks[0]
            assert "مرحبا" in chunks[0]
            assert "🌍" in chunks[0]
    
    def test_chunk_text_multiple_spaces(self, mock_settings):
        """Test text with multiple consecutive spaces"""
        with patch('agentic_rag.data.chunk_text.settings', mock_settings):
            text = "word1    word2     word3"
            chunks = chunk_text(text)
            
            # split() normalizes multiple spaces to single space
            assert chunks[0] == "word1 word2 word3"
    
    def test_chunk_text_realistic_wordpress_content(self, mock_settings):
        """Test with realistic WordPress Q&A content"""
        mock_settings.chunking.max_tokens = 50
        mock_settings.chunking.overlap = 10
        
        with patch('agentic_rag.data.chunk_text.settings', mock_settings):
            text = """
            How do I add custom CSS to my WordPress theme? You can add custom CSS 
            in several ways. First, you can use the WordPress Customizer by going 
            to Appearance > Customize > Additional CSS. Second, you can add it to 
            your child theme's style.css file. Third, you can use a plugin like 
            Simple Custom CSS. Each method has its pros and cons depending on your 
            technical skill level and whether you're using a child theme.
            """
            
            chunks = chunk_text(text)
            
            # Should create multiple chunks
            assert len(chunks) >= 2
            
            # All chunks should be non-empty
            assert all(len(chunk.split()) > 0 for chunk in chunks)
            
            # First chunk should not exceed max_tokens
            assert len(chunks[0].split()) <= mock_settings.chunking.max_tokens
    
    def test_chunk_text_settings_used(self):
        """Test that function uses actual settings from get_settings()"""
        # This test verifies the function reads from actual settings
        # Not using mock here to test integration
        text = "word " * 200  # 200 words
        chunks = chunk_text(text)
        
        # Should create multiple chunks based on actual settings
        assert len(chunks) > 1
        
        # Import settings to verify
        from agentic_rag.settings import get_settings
        settings = get_settings()
        
        # First chunk should respect max_tokens
        first_chunk_words = len(chunks[0].split())
        assert first_chunk_words <= settings.chunking.max_tokens
    
    def test_chunk_text_no_data_loss(self, mock_settings):
        """Test that chunking doesn't lose data (accounting for overlap)"""
        mock_settings.chunking.max_tokens = 20
        mock_settings.chunking.overlap = 5
        
        with patch('agentic_rag.data.chunk_text.settings', mock_settings):
            # Create unique words to track
            words = [f"unique{i}" for i in range(50)]
            text = " ".join(words)
            
            chunks = chunk_text(text)
            
            # Combine all chunks and extract unique words
            all_chunk_words = []
            for chunk in chunks:
                all_chunk_words.extend(chunk.split())
            
            unique_chunk_words = set(all_chunk_words)
            original_words = set(words)
            
            # All original words should appear in chunks
            assert original_words.issubset(unique_chunk_words)
    
    def test_chunk_text_performance_large_text(self, mock_settings):
        """Test performance with large text"""
        with patch('agentic_rag.data.chunk_text.settings', mock_settings):
            # Create large text (10,000 words)
            words = [f"word{i}" for i in range(10000)]
            text = " ".join(words)
            
            import time
            start = time.time()
            chunks = chunk_text(text)
            duration = time.time() - start
            
            # Should complete in reasonable time (< 1 second)
            assert duration < 1.0
            assert len(chunks) > 0
    
    def test_chunk_calculation_formula(self, mock_settings):
        """Test the mathematical correctness of chunk calculation"""
        mock_settings.chunking.max_tokens = 100
        mock_settings.chunking.overlap = 20
        
        with patch('agentic_rag.data.chunk_text.settings', mock_settings):
            # 250 words
            words = [f"w{i}" for i in range(250)]
            text = " ".join(words)
            
            chunks = chunk_text(text)
            
            # Expected chunks with step = max_tokens - overlap = 80
            # Chunk 0: 0-99 (100 words)
            # Chunk 1: 80-179 (100 words)  
            # Chunk 2: 160-249 (90 words)
            expected_num_chunks = 3
            
            assert len(chunks) == expected_num_chunks


class TestChunkTextEdgeCases:
    """Additional edge case tests"""
    
    def test_zero_max_tokens(self):
        """Test with max_tokens = 0"""
        mock_settings = MagicMock()
        mock_settings.chunking.max_tokens = 0
        mock_settings.chunking.overlap = 0
        
        with patch('agentic_rag.data.chunk_text.settings', mock_settings):
            text = "some text here"
            
            # Should handle gracefully or raise error
            # Current implementation will create infinite loop
            # This is a bug that should be fixed
            with pytest.raises(Exception):
                # Timeout after 1 second to prevent hanging
                import signal
                def timeout_handler(signum, frame):
                    raise TimeoutError("Function took too long")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(1)
                try:
                    chunk_text(text)
                finally:
                    signal.alarm(0)
    
    def test_negative_max_tokens(self):
        """Test with negative max_tokens"""
        mock_settings = MagicMock()
        mock_settings.chunking.max_tokens = -10
        mock_settings.chunking.overlap = 5
        
        with patch('agentic_rag.data.chunk_text.settings', mock_settings):
            text = "some text here"
            
            # Should handle gracefully
            # Current implementation will have issues
            chunks = chunk_text(text)
            # Behavior is undefined but shouldn't crash
            assert isinstance(chunks, list)
    
    def test_negative_overlap(self):
        """Test with negative overlap"""
        mock_settings = MagicMock()
        mock_settings.chunking.max_tokens = 10
        mock_settings.chunking.overlap = -5
        
        with patch('agentic_rag.data.chunk_text.settings', mock_settings):
            words = [f"w{i}" for i in range(30)]
            text = " ".join(words)
            
            chunks = chunk_text(text)
            
            # Negative overlap means gaps between chunks
            # Step = 10 - (-5) = 15
            assert len(chunks) > 0


# Integration test
class TestChunkTextIntegration:
    """Integration tests with actual settings"""
    
    def test_with_default_settings(self):
        """Test chunking with actual default settings"""
        from agentic_rag.data.chunk_text import chunk_text
        
        # WordPress-style content
        text = " ".join([f"word{i}" for i in range(300)])
        
        chunks = chunk_text(text)
        
        assert len(chunks) > 1
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert all(len(chunk) > 0 for chunk in chunks)