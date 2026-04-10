import time
from contextlib import contextmanager

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder


class TimingTracker:
    def __init__(self):
        self.timings = {}

    @contextmanager
    def track(self, stage):
        start = time.perf_counter()
        yield
        self.timings[stage] = round((time.perf_counter() - start) * 1000, 2)

    def total(self):
        return round(sum(self.timings.values()), 2)


class Embedder:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed(self, text):
        return self.model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts):
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


class InMemoryVectorStore:
    def __init__(self, docs, embedder):
        self.docs = docs
        texts = [d["text"] for d in docs]
        self.matrix = embedder.embed_batch(texts)  # (N, 384)

    def search(self, query_vec, top_k=50):
        sims = self.matrix @ query_vec  # cosine sim (vectors normalized)
        top_k = min(top_k, len(self.docs))
        top_idx = np.argsort(-sims)[:top_k]
        return [{"id": self.docs[i]["id"], "text": self.docs[i]["text"], "score": float(sims[i])} for i in top_idx]


class Reranker:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query, candidates, top_k=5):
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)
        for i, c in enumerate(candidates):
            c["rerank_score"] = float(scores[i])
        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]


class SemanticCache:
    def __init__(self, threshold=0.85):
        self.threshold = threshold
        self.vectors = []   # list of np.ndarray
        self.entries = []   # list of {"query": str, "results": list}
        self.hits = 0
        self.misses = 0

    def lookup(self, query_vec):
        if not self.vectors:
            self.misses += 1
            return None
        matrix = np.vstack(self.vectors)
        sims = matrix @ query_vec
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim >= self.threshold:
            self.hits += 1
            return self.entries[best_idx], best_sim
        self.misses += 1
        return None

    def store(self, query, query_vec, results):
        self.vectors.append(query_vec)
        self.entries.append({"query": query, "results": results})

    def stats(self):
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": round(self.hits / total * 100, 1) if total else 0,
        }


class RetrievalPipeline:
    def __init__(self, docs):
        print("Loading embedder...")
        self.embedder = Embedder()
        print("Loading reranker...")
        self.reranker = Reranker()
        print("Building vector store...")
        self.vector_store = InMemoryVectorStore(docs, self.embedder)
        self.cache = SemanticCache(threshold=0.85)
        print("Pipeline ready.\n")

    def search(self, query):
        timer = TimingTracker()

        with timer.track("embed_ms"):
            qvec = self.embedder.embed(query)

        with timer.track("cache_lookup_ms"):
            cached = self.cache.lookup(qvec)

        if cached:
            entry, sim = cached
            return {
                "results": entry["results"],
                "status": "HIT",
                "similarity": round(sim, 4),
                "timing": {**timer.timings, "total_ms": timer.total()},
            }

        with timer.track("vector_search_ms"):
            candidates = self.vector_store.search(qvec, top_k=50)

        with timer.track("rerank_ms"):
            top = self.reranker.rerank(query, candidates, top_k=5)

        with timer.track("cache_store_ms"):
            self.cache.store(query, qvec, top)

        return {
            "results": top,
            "status": "MISS",
            "timing": {**timer.timings, "total_ms": timer.total()},
        }
