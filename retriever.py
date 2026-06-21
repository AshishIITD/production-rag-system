import faiss
import numpy as np
import json
import os
from pathlib import Path
from loguru import logger
from embeddings import embed_texts, embed_query, rerank
from models import SourceDocument
from config import settings


class FAISSRetriever:
    """FAISS-based vector retriever with metadata store."""

    def __init__(self):
        self.index = None
        self.metadata: list[dict] = []
        self.index_path = Path(settings.faiss_index_path)
        self._load_or_init()

    def _load_or_init(self):
        idx_file = self.index_path / "index.faiss"
        meta_file = self.index_path / "metadata.json"
        if idx_file.exists() and meta_file.exists():
            logger.info("Loading existing FAISS index")
            self.index = faiss.read_index(str(idx_file))
            with open(meta_file) as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded {len(self.metadata)} chunks")
        else:
            logger.info("Initializing new FAISS index (dim=1024 for BGE-M3)")
            self.index = faiss.IndexFlatIP(1024)  # Inner product for cosine sim

    def add_documents(self, texts: list[str], sources: list[str], pages: list[int] = None):
        """Embed and add documents to the index."""
        logger.info(f"Embedding {len(texts)} chunks...")
        embeddings = embed_texts(texts).astype("float32")
        self.index.add(embeddings)
        for i, (text, source) in enumerate(zip(texts, sources)):
            self.metadata.append({
                "content": text,
                "source": source,
                "page": pages[i] if pages else None,
            })
        self._save()
        logger.info(f"Index now has {self.index.ntotal} vectors")

    def retrieve(self, query: str, top_k: int = None) -> list[SourceDocument]:
        """Retrieve top-k documents, then rerank."""
        if self.index.ntotal == 0:
            return []

        k = top_k or settings.retrieval_top_k
        rerank_k = settings.rerank_top_k

        query_vec = embed_query(query).astype("float32").reshape(1, -1)
        scores, indices = self.index.search(query_vec, min(k, self.index.ntotal))

        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            candidates.append(SourceDocument(
                content=meta["content"],
                source=meta["source"],
                page=meta.get("page"),
                score=float(score),
            ))

        if len(candidates) <= rerank_k:
            return candidates

        # Rerank with cross-encoder
        doc_texts = [d.content for d in candidates]
        reranked_indices = rerank(query, doc_texts, rerank_k)
        return [candidates[i] for i, _ in reranked_indices]

    def _save(self):
        self.index_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path / "index.faiss"))
        with open(self.index_path / "metadata.json", "w") as f:
            json.dump(self.metadata, f)

    @property
    def total_chunks(self) -> int:
        return self.index.ntotal if self.index else 0


# Singleton
_retriever = None

def get_retriever() -> FAISSRetriever:
    global _retriever
    if _retriever is None:
        _retriever = FAISSRetriever()
    return _retriever
