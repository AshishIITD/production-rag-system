"""
RAGAS evaluation module.
Computes: faithfulness, answer_relevancy, context_precision, context_recall
"""
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset
from loguru import logger
from models import QueryResponse, RAGASScores
from config import settings
import json
from pathlib import Path
from datetime import datetime

EVAL_LOG_PATH = Path("./data/eval_log.jsonl")
EVAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


async def evaluate_response(response: QueryResponse) -> RAGASScores:
    """Run RAGAS evaluation on a single query-answer pair."""
    try:
        eval_data = {
            "question": [response.question],
            "answer": [response.answer],
            "contexts": [[doc.content for doc in response.sources]],
            "ground_truth": [response.question],  # Placeholder; replace with real GT if available
        }
        dataset = Dataset.from_dict(eval_data)
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        scores = RAGASScores(
            faithfulness=round(float(result["faithfulness"]), 4),
            answer_relevancy=round(float(result["answer_relevancy"]), 4),
            context_precision=round(float(result["context_precision"]), 4),
            context_recall=round(float(result["context_recall"]), 4),
        )
        _log_eval(response, scores)
        logger.info(f"RAGAS scores: {scores}")
        return scores
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        return RAGASScores()


def _log_eval(response: QueryResponse, scores: RAGASScores):
    """Persist evaluation results to JSONL for dashboard aggregation."""
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "query_id": response.query_id,
        "question": response.question,
        "latency_ms": response.latency_ms,
        "model": response.model_used,
        "faithfulness": scores.faithfulness,
        "answer_relevancy": scores.answer_relevancy,
        "context_precision": scores.context_precision,
        "context_recall": scores.context_recall,
    }
    with open(EVAL_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_eval_history() -> list[dict]:
    """Load all evaluation records for dashboard."""
    if not EVAL_LOG_PATH.exists():
        return []
    records = []
    with open(EVAL_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
