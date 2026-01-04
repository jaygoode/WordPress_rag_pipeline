from typing import List
from ..settings import get_settings

settings = get_settings()

def chunk_text(text: str) -> List[str]:
    """
    Split text into chunks suitable for embedding.
    
    Args:
        text: Input cleaned text.
        max_tokens: Approximate max number of words per chunk.
        overlap: Number of words to overlap between consecutive chunks.
    
    Returns:
        List of text chunks.
    """
    #TODO use langchain or tiktoken for tokenization? and chunking
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + settings.chunking.max_tokens, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += settings.chunking.max_tokens - settings.chunking.overlap
    return chunks
