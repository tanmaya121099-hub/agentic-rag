# Architecture Deep Dive

## System Overview

The system implements three interlocking loops that together form a self-correcting RAG pipeline.

## Loop 1: Hybrid Retrieval

```
Query
  │
  ├─► Dense Search (OpenAI embeddings → Qdrant)
  │       top-K results + cosine scores
  │
  └─► Sparse Search (BM25Okapi)
          top-K results + BM25 scores
              │
              ▼
     Reciprocal Rank Fusion
     score = Σ [ 1 / (60 + rank_i) ]
              │
              ▼
     Top-K fused chunks
```

**Why hybrid?**
- Vector alone: misses exact terms ("iPhone 15 Pro Max" ≈ "iPhone 15" semantically)
- BM25 alone: misses paraphrases ("car" ≠ "automobile")
- RRF constant `k=60` is empirically validated standard across literature

## Loop 2: Self-Correction

```
Chunks
  │
  ▼
Grader LLM (GPT-4o-mini, structured output)
  │
  ├── relevant=True  ──────────────────────► Generate
  │
  └── relevant=False
            │
            ├── attempts < 2 ──► Query Rewriter ──► Retrieve again
            │
            └── attempts ≥ 2 ──► Web Search (Tavily) ──► Generate
```

**Circuit breaker**: Hard limit of `MAX_REWRITE_ATTEMPTS=2` prevents infinite loops.

**Grader model choice**: GPT-4o-mini at ~$0.00015/1K tokens (vs $0.003 for GPT-4o).
Each grader call ≈ 400 tokens in + 50 tokens out ≈ $0.00007 per grade.

## Loop 3: Intent Routing

```
Query → Intent Classifier (GPT-4o-mini)
          │
          ├── "local"  → Hybrid Search → [Loop 2]
          ├── "web"    → Tavily Search → Generate
          └── "both"   → Hybrid Search → [Loop 2] (web as fallback)
```

## Data Flow Diagram

```
┌─────────────┐    cache hit    ┌─────────────┐
│  FastAPI    │ ──────────────► │    Redis    │
│  /query     │                 └─────────────┘
└──────┬──────┘
       │ cache miss
       ▼
┌─────────────────────────────────────────────────┐
│              LangGraph State Machine            │
│                                                 │
│  classify_intent → retrieve → grade            │
│                         ↑          │ bad        │
│                         │          ▼            │
│                    rewrite_query (max 2)        │
│                                   │ give up     │
│                                   ▼            │
│                              web_search         │
│                                   │             │
│                                   ▼             │
│                                generate         │
└─────────────────────────────────────────────────┘
       │
       ▼
┌─────────────┐    trace     ┌──────────────────┐
│   Answer +  │ ───────────► │   LangSmith      │
│   Sources   │              │   (every node)   │
└─────────────┘              └──────────────────┘
```

## State Schema

```python
class RAGState(TypedDict):
    query: str               # mutable — rewritten on bad retrieval
    original_query: str      # immutable — used for final generation
    intent: str              # local | web | both
    chunks: list[dict]       # retrieved documents
    grade_relevant: bool     # grader verdict
    grade_reason: str        # grader explanation
    rewrite_attempts: int    # circuit breaker counter
    used_web: bool           # true if web fallback triggered
    answer: str              # final output
    sources: list[str]       # cited sources
```

## Cost Model

| Operation | Model | Cost/Call |
|-----------|-------|-----------|
| Embedding (1 doc) | text-embedding-3-small | ~$0.00002 |
| Intent classification | GPT-4o-mini | ~$0.00005 |
| Retrieval grading | GPT-4o-mini | ~$0.00007 |
| Query rewriting | GPT-4o-mini | ~$0.00006 |
| Answer generation | Claude Sonnet 4.6 | ~$0.003 |
| **Total (no cache)** | | **~$0.004** |
| **Total (cache hit)** | | **$0.000** |

## Failure Modes and Mitigations

| Failure | Mitigation |
|---------|-----------|
| Qdrant down | `health()` check on startup; 503 response |
| Redis down | Cache miss — degrade gracefully, still answer |
| Grader hallucination | Structured output via Pydantic forces valid JSON |
| Infinite rewrite loop | Hard counter limit with web fallback |
| LLM timeout | `tenacity` retry with exponential backoff (3 attempts) |
| No answer in docs or web | Generator told to say "I don't know" rather than hallucinate |
