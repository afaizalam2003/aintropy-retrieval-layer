# Open-ended Problem 2: Cross-Modal Understanding

## Enterprise-Scale Search Over 100M Multi-Format Documents

**Author**: Faiz Alam · April 2026
**Context**: AIntropy engineer interview — open-ended problem exploration

---

## 1. Problem Summary

Build a search platform for a client with **100M+ documents** spread across:
- **6 source systems**: Google Drive, SharePoint, Box, Dropbox, Object Store, POSIX file systems
- **5+ formats**: PDF, PPTX, DOCX, Excel, Relational DB
- **Per-document metadata**: filename, description, size, user permissions (read/edit/own)

The platform must answer **any question** about knowledge contained across all documents — including questions requiring **cross-document reasoning** (facts inferred from multiple sources).

### Constraints

| Constraint | Target |
|---|---|
| Corpus size | 100M+ documents |
| Latency | Sub-second knowledge retrieval |
| Accuracy | Higher than ChatGPT (which has no access to proprietary data) |
| Cost | Lower than fine-tuning an LLM/VLM on the entire corpus |
| Security | Results must respect per-user access permissions |

---

## 2. State of the Art

### Current approaches

| Approach | Strengths | Weaknesses |
|---|---|---|
| **Naive RAG** (embed → search → generate) | Simple, works for single-source text | Poor on multi-format, no permission handling, low accuracy at scale |
| **Enterprise search** (Glean, Moveworks, Guru) | Multi-source connectors, good UX | Proprietary, expensive, limited customization |
| **Hybrid retrieval** (dense + sparse) | BM25 catches keyword matches that embeddings miss | Still single-modal, no cross-document reasoning |
| **ColBERT / late interaction** | Better retrieval accuracy via token-level matching | Higher storage cost (per-token vectors), complex indexing |
| **ColPali** (vision-language retrieval) | Embeds document pages as images — no lossy text extraction | New, less mature, high compute for embedding |
| **Fine-tuning LLM on corpus** | Deep knowledge integration | Prohibitively expensive ($100K+), no permission control, can't update incrementally |

### Key insight

No single approach solves the full problem. The opportunity is in **combining** the best elements: multi-source connectors + format-aware parsing + hybrid retrieval + permission-aware filtering + semantic caching (which I demonstrated in my Concrete Problem 2 POC).

---

## 3. Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                          │
│                                                                     │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐          │
│  │  Google    │ │SharePoint │ │  Dropbox  │ │   Box     │  ...     │
│  │  Drive    │ │  (Graph)  │ │   API     │ │   API     │          │
│  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘          │
│        └──────────────┴──────────────┴──────────────┘              │
│                            │                                        │
│                            ▼                                        │
│               ┌────────────────────────┐                           │
│               │  Unified Connector     │  ← Extracts:             │
│               │  Layer                 │    • Raw content          │
│               │                        │    • Metadata             │
│               │                        │    • ACLs (permissions)   │
│               └────────────┬───────────┘                           │
│                            │                                        │
│                            ▼                                        │
│               ┌────────────────────────┐                           │
│               │  Format-Aware Parser   │  ← Unstructured.io       │
│               │  PDF│DOCX│PPTX│XLS│SQL │    handles all formats   │
│               └────────────┬───────────┘                           │
│                            │                                        │
│                            ▼                                        │
│               ┌────────────────────────┐                           │
│               │  Semantic Chunker      │  ← 500 tokens/chunk      │
│               │  + Metadata Attach     │    50 token overlap       │
│               │  + ACL Inheritance     │    ACLs inherited         │
│               └────────────┬───────────┘                           │
│                            │                                        │
│                            ▼                                        │
│               ┌────────────────────────┐                           │
│               │  Embedding Layer       │  ← sentence-transformers  │
│               │  (batch, async)        │    or OpenAI ada-002      │
│               └────────────┬───────────┘                           │
│                            │                                        │
│                            ▼                                        │
│        ┌───────────────────┴───────────────────┐                   │
│        │                                       │                   │
│        ▼                                       ▼                   │
│  ┌──────────────┐                    ┌──────────────────┐          │
│  │ Vector Index │ (Qdrant/Milvus)    │ Metadata Index   │          │
│  │ HNSW, 1B vec │                    │ (Elasticsearch)  │          │
│  │ sharded      │                    │ ACLs, dates,     │          │
│  └──────────────┘                    │ formats, sources │          │
│                                      └──────────────────┘          │
└─────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                         QUERY PIPELINE                              │
│                                                                     │
│  User Query + User Identity                                        │
│        │                                                            │
│        ▼                                                            │
│  ┌──────────────────────┐                                          │
│  │ 1. ACL Pre-Filter    │ ← Resolve user's permissions             │
│  │    (Elasticsearch)   │    Build allowed-doc filter               │
│  └──────────┬───────────┘                                          │
│             │                                                       │
│             ▼                                                       │
│  ┌──────────────────────┐                                          │
│  │ 2. Semantic Cache    │ ← Cosine sim > 0.85 → HIT               │
│  │    (same as my POC)  │    Returns cached top-K instantly        │
│  └──────────┬───────────┘                                          │
│             │                                                       │
│        HIT ─┤── return cached results                              │
│             │                                                       │
│        MISS ▼                                                       │
│  ┌──────────────────────┐                                          │
│  │ 3. Hybrid Search     │ ← Dense (vector) + Sparse (BM25)        │
│  │    Top-100 candidates│    Reciprocal Rank Fusion to merge       │
│  └──────────┬───────────┘                                          │
│             │                                                       │
│             ▼                                                       │
│  ┌──────────────────────┐                                          │
│  │ 4. ACL Post-Filter   │ ← Safety net: remove any results        │
│  │                      │    user shouldn't see                    │
│  └──────────┬───────────┘                                          │
│             │                                                       │
│             ▼                                                       │
│  ┌──────────────────────┐                                          │
│  │ 5. Cross-Encoder     │ ← Rerank top-100 → top-5                │
│  │    Reranker          │    (same as my POC)                      │
│  └──────────┬───────────┘                                          │
│             │                                                       │
│             ▼                                                       │
│  ┌──────────────────────┐                                          │
│  │ 6. LLM Generation    │ ← Top-5 chunks as context               │
│  │    (GPT-4 / Claude)  │    Generates final answer + sources      │
│  └──────────┬───────────┘                                          │
│             │                                                       │
│             ▼                                                       │
│  Return: Answer + Source Documents + Confidence Score               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Deep Dive: Key Design Decisions

### 4.1 Why Hybrid Retrieval (Dense + Sparse)?

Dense retrieval (embeddings) is good at semantic matching: "refund policy" matches "return process". But it misses **exact keyword matches** that matter in enterprise — product codes, employee IDs, legal clause numbers.

Sparse retrieval (BM25) catches these keyword matches but misses semantic equivalences.

**Combining both with Reciprocal Rank Fusion (RRF)** gives the best of both:

```
Dense results:   [Doc A (0.89), Doc C (0.85), Doc F (0.82), ...]
Sparse results:  [Doc B (12.4), Doc A (11.2), Doc D (10.8), ...]

RRF merges:      [Doc A (high in both), Doc B, Doc C, Doc D, Doc F, ...]
```

Doc A ranks high in both → rises to the top. Documents strong in one dimension aren't lost.

### 4.2 Permission-Aware Retrieval (ACL Handling)

This is the **most critical production requirement** — a leak of confidential documents is a security incident.

**Three-layer approach:**

| Layer | When | How | Purpose |
|---|---|---|---|
| **ACL ingestion** | At crawl time | Extract permissions from source API (Drive ACLs, SharePoint roles, etc.) | Know who can see what |
| **Pre-filter** | Before search | Add `user_allowed_docs` filter to vector search query | Reduce search space, prevent leaks |
| **Post-filter** | After search | Re-verify each result's ACL against user identity | Safety net for edge cases |

**Why both pre AND post filter?**

Pre-filter alone can have stale ACLs (permission changed after last sync). Post-filter alone is wasteful (you search 100 results then throw away 80). Hybrid catches both cases.

**ACL sync cadence**: Every 15 minutes for active documents, daily full sweep. Permission changes propagate within 15 minutes — acceptable for most enterprise use cases.

### 4.3 Format-Aware Parsing Strategy

Not all formats are equal. The parsing strategy adapts:

| Format | Strategy | Why |
|---|---|---|
| **PDF (text-heavy)** | PyMuPDF text extraction → semantic chunking | Fast, accurate for text PDFs |
| **PDF (scanned)** | OCR (Tesseract/EasyOCR) → text → chunking | Scanned docs need OCR first |
| **PDF (charts/diagrams)** | ColPali page-image embedding (no text extraction) | Avoids lossy chart-to-text conversion |
| **DOCX / PPTX** | Unstructured.io → preserves headings, tables, slide structure | Maintains document hierarchy |
| **Excel** | Row-level chunking with column headers as context | Each row becomes a searchable "document" |
| **Relational DB** | Schema-aware: table description + sample rows embedded together | Captures both structure and content |

**Key insight**: The parser should be **format-adaptive**, not one-size-fits-all. A PDF with charts needs a completely different pipeline than a PDF with text. Detection → routing → specialized parsing.

### 4.4 Semantic Cache at Enterprise Scale

My Concrete Problem 2 POC demonstrated semantic caching with an in-memory implementation. At 100M-document scale with thousands of concurrent users:

| POC | Production |
|---|---|
| Python list + numpy | Redis Vector Search or FAISS IVF |
| Single-process | Distributed, multi-process |
| No eviction | TTL-based + LRU eviction |
| No invalidation | Invalidate on document update (via webhook from source system) |
| Global cache | Per-tenant cache namespace (multi-tenancy) |

The **interface stays identical**: `cache.lookup(query_vec)` → hit or miss. Only the backend changes.

**Expected impact at scale**: Enterprise workloads show 40-60% semantic repeat rate (support queries, FAQ, dashboard refreshes). At 50% hit rate with 15× speedup on cache hits, the **average query latency drops by ~47%** and **per-query compute cost drops proportionally**.

---

## 5. Measuring Accuracy

### The challenge

Real enterprise corpora don't come with labeled benchmarks. Academic benchmarks (BEIR, MS MARCO) don't reflect proprietary data distributions. We need to **build our own evaluation**.

### Three-layer evaluation framework

| Layer | Method | Measures | Frequency |
|---|---|---|---|
| **Synthetic benchmark** | LLM generates Q&A pairs from known documents | Retrieval recall (Recall@K, NDCG@K) | Every pipeline change (CI/CD) |
| **LLM-as-judge** | GPT-4/Claude scores answer quality (relevance, correctness, completeness) | End-to-end answer quality | Daily / weekly sampling |
| **Human evaluation** | Domain experts review 200-500 query-answer pairs | Ground truth calibration | Quarterly |

### Synthetic benchmark construction

1. Sample 1,000 documents uniformly across formats and sources
2. For each document, prompt an LLM: *"Generate 3 questions that can only be answered using this document"*
3. For cross-document questions: select 2-3 related documents, prompt: *"Generate a question that requires information from all of these documents"*
4. This gives ~3,000 single-doc questions + ~500 cross-doc questions
5. Run retrieval pipeline → measure Recall@5, Recall@20, NDCG@5
6. Target: Recall@5 > 85%, NDCG@5 > 0.75

### Why this beats ChatGPT accuracy

ChatGPT has zero access to proprietary data. Our system retrieves from the actual corpus. Even with imperfect retrieval (85% recall), we're providing **real internal information** that ChatGPT literally cannot access. The accuracy bar is "better than ChatGPT" — which for proprietary questions is effectively "better than guessing."

---

## 6. Cost Estimation

### Index construction (one-time)

| Component | Calculation | Cost |
|---|---|---|
| Parsing 100M docs | ~10 docs/sec with Unstructured.io, ~115 days on 1 machine → parallelize 100× | ~$5,000 (cloud compute) |
| Embedding 1B chunks | OpenAI ada-002: $0.0001/1K tokens × 500 tokens × 1B = $50,000 | $50,000 (API) |
| | Self-hosted (8× A100 GPU): ~3 days | ~$3,000 (GPU rental) |
| Vector DB setup | Qdrant Cloud, 1B vectors, 384-dim | ~$2,000 (initial) |
| Metadata index | Elasticsearch cluster | ~$500 (initial) |
| **Total (self-hosted embedding)** | | **~$10,500** |
| **Total (API embedding)** | | **~$57,500** |

### Per-query cost

| Component | Latency | Cost per query |
|---|---|---|
| ACL pre-filter | ~5ms | negligible |
| Semantic cache (hit) | ~2ms | negligible |
| Vector search (miss) | ~30ms | ~$0.0001 |
| Cross-encoder rerank | ~100ms (GPU) | ~$0.001 |
| LLM generation | ~500ms | ~$0.01-0.05 |
| **Total (cache hit)** | **~33ms** | **~$0.001** |
| **Total (cache miss)** | **~650ms** | **~$0.01-0.05** |

### Monthly operational cost (1M queries/month, 50% cache hit)

| Component | Monthly cost |
|---|---|
| Vector DB hosting (Qdrant Cloud) | ~$3,000-5,000 |
| Elasticsearch | ~$500-1,000 |
| Compute (reranker + API) | ~$2,000-5,000 |
| LLM generation (500K non-cached queries) | ~$5,000-25,000 |
| Redis cache | ~$200-500 |
| **Total** | **~$11,000-36,000/month** |

### Compare: fine-tuning approach

| | Our approach | Fine-tuning |
|---|---|---|
| Setup cost | ~$10K-57K | $100K-500K+ |
| Monthly cost | ~$11-36K | $50K+ (hosting) |
| Update on new docs | Incremental (minutes) | Retrain (days, $$$) |
| Permission control | Per-query filtering | Impossible (baked in weights) |
| Time to deploy | Weeks | Months |

**Our approach is 3-10× cheaper with better operational properties.**

---

## 7. Opportunities for a Fundamentally Different Approach

Beyond incremental improvements, I see two opportunities:

### 7.1 Graph-augmented retrieval

Instead of treating documents as independent chunks, build a **knowledge graph** on top:
- Nodes = entities (people, products, projects, dates)
- Edges = relationships (authored, references, approved, depends-on)
- Query-time: traverse the graph to find related documents that vector search alone would miss

This enables **multi-hop reasoning**: *"Who approved the budget for the project that Q3 revenue depends on?"* requires traversing budget → project → revenue connections.

### 7.2 Adaptive retrieval routing

Not every query needs the full pipeline. A lightweight classifier could route:
- **Simple factual queries** → keyword search only (fast, cheap)
- **Semantic queries** → vector search + reranker
- **Complex reasoning queries** → graph traversal + multi-step retrieval + LLM chain-of-thought

This reduces average cost and latency by matching pipeline complexity to query complexity.

---

## 8. Connection to My Concrete POC

The retrieval layer I built for Concrete Problem 2 (`aintropy-retrieval-layer`) implements **steps 2 and 5** of this architecture:

| Architecture step | POC component |
|---|---|
| Semantic cache | `SemanticCache` class with empirically-tuned threshold |
| Cross-encoder rerank | `Reranker` class using ms-marco-MiniLM |
| Latency instrumentation | `TimingTracker` with per-stage ms breakdown |
| Pipeline orchestration | `RetrievalPipeline.search()` method |

The POC uses numpy in-memory as the vector store (appropriate for 30 docs). In the full architecture, this swaps to sharded Qdrant — but the `RetrievalPipeline` interface stays identical. **The POC is a working blueprint of the production system's core retrieval loop.**

Benchmark results: **422ms cold → 33ms cached (12.5× speedup), 100% cache hit rate on paraphrased queries.**

Repo: [github.com/afaizalam2003/aintropy-retrieval-layer](https://github.com/afaizalam2003/aintropy-retrieval-layer)

---

## 9. Summary

| Aspect | Approach |
|---|---|
| **Ingestion** | Unified connectors → format-aware parsing → semantic chunking → batch embedding |
| **Indexing** | Sharded vector DB (HNSW) + metadata index (Elasticsearch) + ACL store |
| **Search** | ACL pre-filter → semantic cache → hybrid retrieval (dense+sparse) → ACL post-filter → cross-encoder rerank → LLM answer |
| **Permissions** | Three-layer: ingest ACLs + pre-filter + post-filter. 15-min sync cadence. |
| **Accuracy** | Synthetic benchmark (Recall@5 > 85%) + LLM-as-judge + quarterly human eval |
| **Cost** | ~$10K setup + ~$11-36K/month. 3-10× cheaper than fine-tuning. |
| **Key differentiator** | Semantic cache (12.5× speedup proven in POC) + permission-aware hybrid retrieval + format-adaptive parsing |
