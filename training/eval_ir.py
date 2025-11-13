import json, numpy as np
from inference.search import AuroraSearcher

QUERIES = [
    ("Missä opiskella ML ja AI Suomessa", [2]),
    ("rekrymessut Turussa", [0]),
    ("AI-hackathon Turussa", [1]),
]

def mrr_at_k(ranked_ids, relevant, k=5):
    for rank, did in enumerate(ranked_ids[:k], start=1):
        if did in relevant:
            return 1.0 / rank
    return 0.0

def dcg_at_k(gains, k=5):
    return sum((g / np.log2(i+2) for i, g in enumerate(gains[:k])))

def ndcg_at_k(ranked_ids, relevant, k=5):
    gains = [1 if rid in relevant else 0 for rid in ranked_ids]
    ideal = sorted(gains, reverse=True)
    return dcg_at_k(gains, k) / (dcg_at_k(ideal, k) + 1e-9)

def main():
    s = AuroraSearcher(top_k=5)
    mrrs, ndcgs = [], []
    for q, rel in QUERIES:
        res = s.search(q)
        ranked = [r["doc_id"] for r in res]
        mrrs.append(mrr_at_k(ranked, rel, 5))
        ndcgs.append(ndcg_at_k(ranked, rel, 5))
    print(f"MRR@5: {np.mean(mrrs):.3f}  nDCG@5: {np.mean(ndcgs):.3f}")

if __name__ == "__main__":
    main()
