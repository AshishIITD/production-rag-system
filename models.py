from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    run_evaluation: bool = False
    top_k: int = Field(default=4, ge=1, le=10)


class SourceDocument(BaseModel):
    content: str
    source: str
    page: Optional[int] = None
    score: float


class RAGASScores(BaseModel):
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceDocument]
    latency_ms: float
    model_used: str
    ragas_scores: Optional[RAGASScores] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    query_id: str


class IngestRequest(BaseModel):
    text: Optional[str] = None
    url: Optional[str] = None


class IngestResponse(BaseModel):
    status: str
    chunks_created: int
    source: str


class MetricsSummary(BaseModel):
    total_queries: int
    avg_faithfulness: Optional[float]
    avg_answer_relevancy: Optional[float]
    avg_context_precision: Optional[float]
    avg_context_recall: Optional[float]
    avg_latency_ms: float
    p95_latency_ms: float
    queries_last_24h: int
