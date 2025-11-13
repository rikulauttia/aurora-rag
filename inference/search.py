import os, json, numpy as np, faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

INDEX_PATH   = "inference/index.faiss"
META_PATH    = "inference/meta.json"
EMB_MODEL    = os.environ.get("EMB_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")  # multilingual

class AuroraSearcher:
    def __init__(self, top_k=5):
        self.emb_model = SentenceTransformer(EMB_MODEL)
        self.index = faiss.read_index(INDEX_PATH)
        self.meta = json.load(open(META_PATH, "r", encoding="utf-8"))
        self.dim = self.index.d
        self.reranker = CrossEncoder(RERANK_MODEL)
        self.top_k = top_k

    def embed(self, text):
        v = self.emb_model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
        return v.astype(np.float32)

    def search(self, query):
        q = self.embed(query)
        k = min(self.top_k * 3, len(self.meta))
        scores, idxs = self.index.search(q, k)
        cands = [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i >= 0]
        pairs = [(query, self.meta[i]["text"]) for i,_ in cands]
        if not pairs:
            return []
        rr = self.reranker.predict(pairs)
        reranked = sorted(zip(cands, rr), key=lambda x: x[1], reverse=True)[:self.top_k]
        out = []
        for ((i, ip), rr_score) in reranked:
            out.append({
                "doc_id": i,
                "text": self.meta[i]["text"],
                "path": self.meta[i]["path"],
                "sim_ip": ip,
                "rerank": float(rr_score)
            })
        return out