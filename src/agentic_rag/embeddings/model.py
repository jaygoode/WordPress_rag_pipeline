from typing import List
import torch
from sentence_transformers import SentenceTransformer
from ..settings import get_settings
import logging
settings = get_settings()
logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE != "cuda":
    logger.info("Warning: CUDA not available, using CPU for embeddings. This may be slow.")

model = SentenceTransformer(settings.vector_store.embedding_model, device=DEVICE)

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
