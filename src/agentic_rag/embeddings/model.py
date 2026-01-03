from typing import List
import torch
from sentence_transformers import SentenceTransformer
from agentic_rag import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE != "cuda":
    print("Warning: CUDA not available, using CPU for embeddings. This may be slow.")

model = SentenceTransformer(config.EMBEDDING_MODEL, device=DEVICE)

def embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts.

    Args:
        texts: List of strings

    Returns:
        List of embedding vectors (as Python lists)
    """
    embeddings = model.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.tolist()
