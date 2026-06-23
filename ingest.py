"""
Document ingestion pipeline.
Supports: PDF, plain text, URLs, raw strings.
Chunks with overlap and indexes into FAISS.
"""
import sys
from unittest.mock import MagicMock
sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()
sys.modules['langchain_community.embeddings.vertexai'] = MagicMock()

import re
import httpx

from pathlib import Path
from pypdf import PdfReader
from loguru import logger
from config import settings

def _get_retriever():
    from retriever import get_retriever
    return get_retriever()


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
    """Split text into overlapping chunks by token approximation."""
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) > 50]


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text.strip()


def ingest_pdf(path: str) -> int:
    """Ingest a PDF file. Returns number of chunks created."""
    reader = PdfReader(path)
    retriever = _get_retriever()
    texts, sources, pages = [], [], []
    for page_num, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        cleaned = clean_text(raw)
        if not cleaned:
            continue
        for chunk in chunk_text(cleaned):
            texts.append(chunk)
            sources.append(Path(path).name)
            pages.append(page_num + 1)
    if texts:
        retriever.add_documents(texts, sources, pages)
    logger.info(f"Ingested {len(texts)} chunks from {path}")
    return len(texts)


def ingest_text(text: str, source_name: str = "manual_input") -> int:
    cleaned = clean_text(text)
    chunks = chunk_text(cleaned)
    retriever = _get_retriever()
    retriever.add_documents(chunks, [source_name] * len(chunks))
    logger.info(f"Ingested {len(chunks)} chunks from {source_name}")
    return len(chunks)


async def ingest_url(url: str) -> int:
    async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
        response = await client.get(url)
        response.raise_for_status()
    text = re.sub(r'<[^>]+>', ' ', response.text)
    return ingest_text(text, source_name=url)


def ingest_sample_data():
    """Ingest sample AI/ML content for demo purposes."""
    sample_docs = [
        {
            "text": """Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval 
            with language model generation. Instead of relying solely on the model's parametric memory, 
            RAG first retrieves relevant documents from a knowledge base using semantic similarity search, 
            then passes those documents as context to the language model to generate a grounded answer.
            RAG reduces hallucinations because the model is constrained to use retrieved evidence.
            Key components include: a document store, an embedding model, a vector index, and a generator LLM.""",
            "source": "rag_overview.txt",
        },
        {
            "text": """RAGAS (Retrieval Augmented Generation Assessment) is an evaluation framework for RAG systems.
            It measures four key metrics: Faithfulness (is the answer supported by the retrieved context?), 
            Answer Relevancy (does the answer address the question?), Context Precision (are retrieved 
            chunks relevant to the question?), and Context Recall (were all relevant chunks retrieved?).
            RAGAS uses LLM-based evaluation to score these metrics without requiring human annotation.""",
            "source": "ragas_overview.txt",
        },
        {
            "text": """Large Language Models (LLMs) are neural networks trained on vast text corpora using 
            the Transformer architecture. Modern LLMs like GPT-4, Claude, and Gemini are trained using 
            a combination of supervised learning, reinforcement learning from human feedback (RLHF), 
            and constitutional AI methods. Fine-tuning techniques like LoRA and QLoRA allow adapting 
            these large models to specific tasks with limited GPU resources by training low-rank 
            adapter matrices instead of all model weights.""",
            "source": "llm_overview.txt",
        },
    ]
    retriever = _get_retriever()
    for doc in sample_docs:
        chunks = chunk_text(doc["text"])
        retriever.add_documents(chunks, [doc["source"]] * len(chunks))
    logger.info(f"Sample data ingested: {retriever.total_chunks} total chunks")


if __name__ == "__main__":
    logger.info("Starting sample data ingestion...")
    ingest_sample_data()
    logger.info("Done. Run: uvicorn app:app --reload")
