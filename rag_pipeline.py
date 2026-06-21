import time
import uuid
from loguru import logger
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from retriever import get_retriever
from models import QueryRequest, QueryResponse, SourceDocument
from config import settings

SYSTEM_PROMPT = """You are a precise, helpful assistant that answers questions strictly based on the provided context.

Rules:
- Answer ONLY from the provided context. Do not use outside knowledge.
- If the context does not contain enough information, say "I don't have enough context to answer this question."
- Be concise and direct. Cite the source when possible.
- Do not make up information."""

def build_prompt(question: str, context_docs: list[SourceDocument]) -> str:
    context_str = "\n\n---\n\n".join([
        f"[Source: {doc.source}]\n{doc.content}"
        for doc in context_docs
    ])
    return f"""Context:
{context_str}

Question: {question}

Answer based strictly on the context above:"""


async def run_openai(prompt: str) -> str:
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=1024,
    )
    return response.choices[0].message.content


async def run_anthropic(prompt: str) -> str:
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    response = await client.messages.create(
        model="claude-sonnet-4-6",
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024,
    )
    return response.content[0].text


async def query_rag(request: QueryRequest) -> QueryResponse:
    start = time.perf_counter()
    query_id = str(uuid.uuid4())[:8]
    logger.info(f"[{query_id}] Query: {request.question[:80]}...")

    # Retrieve
    retriever = get_retriever()
    docs = retriever.retrieve(request.question, top_k=request.top_k)
    logger.info(f"[{query_id}] Retrieved {len(docs)} docs after reranking")

    if not docs:
        return QueryResponse(
            question=request.question,
            answer="No relevant documents found in the knowledge base.",
            sources=[],
            latency_ms=0,
            model_used=settings.llm_model,
            query_id=query_id,
        )

    # Generate
    prompt = build_prompt(request.question, docs)
    if settings.llm_provider == "anthropic":
        answer = await run_anthropic(prompt)
        model_used = "claude-sonnet-4-6"
    else:
        answer = await run_openai(prompt)
        model_used = settings.llm_model

    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(f"[{query_id}] Answered in {latency_ms:.0f}ms")

    return QueryResponse(
        question=request.question,
        answer=answer,
        sources=docs,
        latency_ms=round(latency_ms, 2),
        model_used=model_used,
        query_id=query_id,
    )
