from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import structlog
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, Field

from src.agents.graph import RAGAgent
from src.cache.redis_client import QueryCache
from src.config import settings

logger = structlog.get_logger(__name__)

# ---------- Prometheus metrics ----------
REQUEST_COUNT = Counter("rag_requests_total", "Total query requests", ["source", "cached"])
LATENCY = Histogram("rag_latency_seconds", "Query latency", buckets=[0.1, 0.3, 0.5, 1, 2, 5])
REWRITE_COUNT = Counter("rag_rewrites_total", "Total query rewrites")
WEB_FALLBACK_COUNT = Counter("rag_web_fallbacks_total", "Times web fallback was triggered")

# ---------- App state (lazy init) ----------
_agent: RAGAgent | None = None
_cache: QueryCache | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent, _cache
    _agent = RAGAgent()
    _cache = QueryCache()
    logger.info("app.started")
    yield
    logger.info("app.shutdown")


app = FastAPI(
    title="Agentic RAG API",
    description="Self-correcting RAG with hybrid search",
    version="0.1.0",
    lifespan=lifespan,
)

# Mount Prometheus metrics
app.mount("/metrics", make_asgi_app())


# ---------- Schemas ----------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    source: Literal["local", "web", "auto"] = "auto"
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    rewrite_attempts: int
    used_web_fallback: bool
    latency_ms: float
    cached: bool


class IngestRequest(BaseModel):
    documents: list[dict]


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    redis: str


# ---------- Endpoints ----------

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    assert _agent is not None and _cache is not None

    cached = _cache.get(req.query, req.source)
    if cached:
        REQUEST_COUNT.labels(source=req.source, cached="true").inc()
        return QueryResponse(**cached, cached=True)

    start = time.perf_counter()
    try:
        result = _agent.run(req.query, source=req.source)
    except Exception as exc:
        logger.error("query.failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Query processing failed") from exc

    latency_ms = (time.perf_counter() - start) * 1000
    LATENCY.observe(latency_ms / 1000)
    REQUEST_COUNT.labels(source=req.source, cached="false").inc()

    if result["rewrite_attempts"] > 0:
        REWRITE_COUNT.inc(result["rewrite_attempts"])
    if result["used_web_fallback"]:
        WEB_FALLBACK_COUNT.inc()

    response_data = {
        "answer": result["answer"],
        "sources": result["sources"],
        "rewrite_attempts": result["rewrite_attempts"],
        "used_web_fallback": result["used_web_fallback"],
        "latency_ms": round(latency_ms, 2),
    }
    _cache.set(req.query, req.source, response_data)
    return QueryResponse(**response_data, cached=False)


@app.post("/ingest", status_code=202)
async def ingest(req: IngestRequest) -> dict:
    assert _agent is not None
    _agent._vector.upsert(req.documents)
    _agent.load_bm25_corpus(req.documents)
    logger.info("ingest.done", count=len(req.documents))
    return {"ingested": len(req.documents)}


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    qdrant_ok = _agent._vector.health() if _agent else False
    redis_ok = _cache.health() if _cache else False
    return HealthResponse(
        status="ok" if (qdrant_ok and redis_ok) else "degraded",
        qdrant="up" if qdrant_ok else "down",
        redis="up" if redis_ok else "down",
    )
