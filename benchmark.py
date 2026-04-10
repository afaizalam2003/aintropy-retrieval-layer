import json
import os
import matplotlib.pyplot as plt
from docs import DOCS
from retrieval import RetrievalPipeline

# 10 pairs: (original query, paraphrased version)
QUERY_PAIRS = [
    ("What is the role of TP53 in cancer?",        "How does TP53 function in tumors?"),
    ("Effects of climate change on coral reefs",   "Climate change effects on coral reefs"),
    ("How do mRNA vaccines work?",                 "Mechanism of mRNA vaccines"),
    ("CRISPR gene editing applications",           "Applications of CRISPR gene editing"),
    ("Diabetes type 2 risk factors",               "Risk factors for type 2 diabetes"),
    ("Antibiotic resistance mechanisms",           "Mechanisms of antibiotic resistance"),
    ("Alzheimer's disease biomarkers",             "Biomarkers of Alzheimer's disease"),
    ("Stem cell therapy for heart disease",        "Using stem cells to treat heart conditions"),
    ("COVID-19 long term effects",                 "Long-term effects of COVID-19"),
    ("Microbiome and immune system",               "Microbiome influence on the immune system"),
]

def main():
    pipeline = RetrievalPipeline(DOCS)
    rows = []

    print("=" * 80)
    print("PASS 1: Cold queries (cache empty)")
    print("=" * 80)
    for original, _ in QUERY_PAIRS:
        out = pipeline.search(original)
        rows.append({"query": original, **out["timing"], "status": out["status"]})
        t = out["timing"]
        print(f"  COLD | {t['total_ms']:6.1f}ms | {original[:55]}")

    print()
    print("=" * 80)
    print("PASS 2: Paraphrased queries (should hit cache)")
    print("=" * 80)
    for _, paraphrase in QUERY_PAIRS:
        out = pipeline.search(paraphrase)
        rows.append({"query": paraphrase, **out["timing"], "status": out["status"]})
        t = out["timing"]
        marker = "HIT " if out["status"] == "HIT" else "MISS"
        print(f"  {marker} | {t['total_ms']:6.1f}ms | {paraphrase[:55]}")

    # Summary
    cold = [r for r in rows if r["status"] == "MISS"]
    hot = [r for r in rows if r["status"] == "HIT"]
    cold_avg = sum(r["total_ms"] for r in cold) / len(cold) if cold else 0
    hot_avg = sum(r["total_ms"] for r in hot) / len(hot) if hot else 1

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Cold avg latency:   {cold_avg:.1f} ms")
    print(f"  Cached avg latency: {hot_avg:.1f} ms")
    print(f"  Speedup:            {cold_avg / hot_avg:.1f}x")
    print(f"  Cache stats:        {pipeline.cache.stats()}")

    # Save raw results
    os.makedirs("results", exist_ok=True)
    with open("results/benchmark.json", "w") as f:
        json.dump({
            "cold_avg_ms": cold_avg,
            "cached_avg_ms": hot_avg,
            "speedup": cold_avg / hot_avg if hot_avg else 0,
            "cache_stats": pipeline.cache.stats(),
            "rows": rows,
        }, f, indent=2)

    # Chart
    stages = ["embed_ms", "cache_lookup_ms", "vector_search_ms", "rerank_ms", "cache_store_ms"]
    cold_avgs = [sum(r.get(s, 0) for r in cold) / len(cold) if cold else 0 for s in stages]
    hot_avgs = [sum(r.get(s, 0) for r in hot) / len(hot) if hot else 0 for s in stages]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(stages))
    ax.bar([i - 0.2 for i in x], cold_avgs, 0.4, label="Cold (MISS)", color="#ef4444")
    ax.bar([i + 0.2 for i in x], hot_avgs,  0.4, label="Cached (HIT)", color="#10b981")
    ax.set_xticks(list(x))
    ax.set_xticklabels([s.replace("_ms", "") for s in stages], rotation=15)
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Cold vs Cached — {cold_avg / hot_avg:.1f}x speedup")
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/latency_breakdown.png", dpi=120)
    print(f"\n  Chart saved: results/latency_breakdown.png")

if __name__ == "__main__":
    main()
