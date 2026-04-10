# AIntropy Retrieval Layer

Sub-second semantic retrieval middleware with **semantic caching** and **cross-encoder reranking**.
Built as a POC for the AIntropy founding engineer interview.

## What it does

A middleware that sits between a user query and a vector store. For every query:

1. **Embeds** the query with MiniLM
2. **Checks a semantic cache** — if a similar query was asked before (cosine sim > 0.85), returns the cached result
3. Otherwise: **vector search top-50** → **cross-encoder reranks to top-5** → store in cache → return

## Results

| Metric            | Cold query | Cached query | Speedup |
|-------------------|-----------:|-------------:|--------:|
| Avg latency       |    ~324 ms |       ~20 ms |  ~16.5x |
| Cache hit rate    |          — |         100% |     —   |

*(actual numbers from `results/benchmark.json`)*

![Latency breakdown](results/latency_breakdown.png)

## Why semantic cache, not string cache?

Enterprise users phrase the same intent differently:
- "What is our refund policy?"
- "How do I get a refund?"
- "Refund process for customer X?"

A string cache catches none of these. A semantic cache catches all of them. At enterprise scale with repeat-intent workloads, this can serve 40-60% of queries from cache — slashing both latency and infrastructure cost.

## Why fetch 50 then rerank to 5?

- Bi-encoder vector search is fast but coarse — good for narrowing 100M docs to 50.
- Cross-encoder is accurate but ~10x slower per pair — fine for 50 pairs, infeasible for 100M.
- Combine them: cross-encoder quality at bi-encoder cost.

## How to run

```bash
pip install -r requirements.txt
python benchmark.py
```

First run downloads the models (~500MB), takes ~1 minute. Subsequent runs are instant.

## Architecture

```
query
  ├── embed (MiniLM, ~20ms)
  ├── semantic cache lookup
  │     ├── HIT  → return cached result          (~20ms total)
  │     └── MISS → vector search top-50 (<1ms)
  │              → cross-encoder rerank (~300ms)
  │              → store in cache
  │              → return                         (~324ms total)
```

## Scaling notes (how this generalizes to 100M docs)

| Component | POC | Production |
|---|---|---|
| Vector store | numpy in-memory (30 docs) | Sharded Qdrant / Milvus with HNSW |
| Cache | Python list | Redis Vector or FAISS IVF |
| Reranker | CPU MiniLM | Batched GPU inference, or distilled model |
| Embeddings | CPU MiniLM | Same, or larger model with GPU |

The interfaces stay the same. Only the implementations swap.

## Files

- `docs.py` — 30 hardcoded science documents (the corpus)
- `retrieval.py` — Embedder, vector store, reranker, cache, pipeline
- `benchmark.py` — Cold vs cached benchmark + chart generation
