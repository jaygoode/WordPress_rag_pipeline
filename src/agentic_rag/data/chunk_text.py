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


# def chunk_text(text: str) -> List[str]:
#     """
#     Split text into chunks suitable for embedding.
    
#     Args:
#         text: Input cleaned text.
    
#     Returns:
#         List of text chunks.
#     """
#     # Handle empty input
#     if not text or not text.strip():
#         return [text]  # Return original (even if empty)
    
#     words = text.split()
    
#     # Handle single word or very short text
#     if len(words) <= settings.chunking.max_tokens:
#         return [text]
    
#     # Validate settings
#     if settings.chunking.max_tokens <= 0:
#         raise ValueError(f"max_tokens must be positive, got {settings.chunking.max_tokens}")
    
#     if settings.chunking.overlap >= settings.chunking.max_tokens:
#         raise ValueError(
#             f"overlap ({settings.chunking.overlap}) must be less than "
#             f"max_tokens ({settings.chunking.max_tokens})"
#         )
    
#     chunks = []
#     start = 0
#     step = settings.chunking.max_tokens - settings.chunking.overlap
    
#     while start < len(words):
#         end = min(start + settings.chunking.max_tokens, len(words))
#         chunk = " ".join(words[start:end])
#         chunks.append(chunk)
        
#         # Move to next chunk
#         start += step
        
#         # Prevent infinite loop or tiny final chunks
#         if step <= 0:
#             break
    
#     return chunks