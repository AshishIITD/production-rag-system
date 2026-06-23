import sys
from unittest.mock import MagicMock
sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()
sys.modules['langchain_community.embeddings.vertexai'] = MagicMock()

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import tempfile
import os

import numpy as np
from loguru import logger

from models import QueryRequest, QueryResponse, IngestRequest, IngestResponse, MetricsSummary
from rag_pipeline import query_rag
from evaluator import evaluate_response, load_eval_history
from retriever import get_retriever
from ingest import ingest_text, ingest_pdf, ingest_url


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Production RAG System...")
    get_retriever()  # Warm up
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Production RAG System",
    description="RAG pipeline with RAGAS evaluation. Built by Ashish Singh.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    retriever = get_retriever()
    return {
        "status": "healthy",
        "indexed_chunks": retriever.total_chunks,
        "version": "1.0.0",
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Submit a question. Set run_evaluation=true to get RAGAS scores (adds ~3s latency)."""
    response = await query_rag(request)
    if request.run_evaluation and response.sources:
        response.ragas_scores = await evaluate_response(response)
    return response


@app.post("/ingest/text", response_model=IngestResponse)
async def ingest_text_endpoint(request: IngestRequest):
    if not request.text and not request.url:
        raise HTTPException(400, "Provide either 'text' or 'url'")
    if request.url:
        chunks = await ingest_url(request.url)
        return IngestResponse(status="success", chunks_created=chunks, source=request.url)
    chunks = ingest_text(request.text, "api_input")
    return IngestResponse(status="success", chunks_created=chunks, source="api_input")


@app.post("/ingest/pdf", response_model=IngestResponse)
async def ingest_pdf_endpoint(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    try:
        chunks = ingest_pdf(tmp_path)
        return IngestResponse(status="success", chunks_created=chunks, source=file.filename)
    finally:
        os.unlink(tmp_path)


@app.get("/metrics", response_model=MetricsSummary)
async def get_metrics():
    """Aggregate RAGAS metrics across all evaluated queries."""
    history = load_eval_history()
    if not history:
        return MetricsSummary(
            total_queries=0, avg_faithfulness=None,
            avg_answer_relevancy=None, avg_context_precision=None,
            avg_context_recall=None, avg_latency_ms=0, p95_latency_ms=0,
            queries_last_24h=0,
        )

    latencies = [r["latency_ms"] for r in history if r.get("latency_ms")]
    faith_scores = [r["faithfulness"] for r in history if r.get("faithfulness") is not None]
    rel_scores = [r["answer_relevancy"] for r in history if r.get("answer_relevancy") is not None]
    prec_scores = [r["context_precision"] for r in history if r.get("context_precision") is not None]
    rec_scores = [r["context_recall"] for r in history if r.get("context_recall") is not None]

    from datetime import datetime, timedelta
    cutoff = (datetime.utcnow() - timedelta(hours=24)).isoformat()
    recent = sum(1 for r in history if r.get("timestamp", "") > cutoff)

    return MetricsSummary(
        total_queries=len(history),
        avg_faithfulness=round(np.mean(faith_scores), 4) if faith_scores else None,
        avg_answer_relevancy=round(np.mean(rel_scores), 4) if rel_scores else None,
        avg_context_precision=round(np.mean(prec_scores), 4) if prec_scores else None,
        avg_context_recall=round(np.mean(rec_scores), 4) if rec_scores else None,
        avg_latency_ms=round(np.mean(latencies), 2) if latencies else 0,
        p95_latency_ms=round(np.percentile(latencies, 95), 2) if latencies else 0,
        queries_last_24h=recent,
    )
