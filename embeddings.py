from config import settings
from loguru import logger
import numpy as np
from functools import lru_cache

# Lazy imports — models only downloaded when first called, not at import time
def _import_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except ImportError:
        raise ImportError("Run: pip install sentence-transformers")

def _import_cross_encoder():
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder
    except ImportError:
        raise ImportError("Run: pip install sentence-transformers")


@lru_cache(maxsize=1)
def get_embedding_model():
    SentenceTransformer = _import_sentence_transformers()
    logger.info(f"Loading embedding model: {settings.embedding_model}")
    return SentenceTransformer(settings.embedding_model)


@lru_cache(maxsize=1)
def get_reranker_model():
    CrossEncoder = _import_cross_encoder()
    logger.info(f"Loading reranker model: {settings.reranker_model}")
    return CrossEncoder(settings.reranker_model)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts using BGE-M3."""
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """Embed a single query with BGE-M3 query prefix."""
    model = get_embedding_model()
    embedding = model.encode(
        f"Represent this sentence for searching relevant passages: {query}",
        normalize_embeddings=True,
    )
    return embedding


def rerank(query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
    """Rerank documents using cross-encoder. Returns (original_index, score) sorted by score."""
    reranker = get_reranker_model()
    pairs = [[query, doc] for doc in documents]
    scores = reranker.predict(pairs)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
