import os, glob, json
import faiss, numpy as np
from sentence_transformers import SentenceTransformer

DOC_DIR   = "data/sample_docs"
INDEX_OUT = "inference/index.faiss"
META_OUT  = "inference/meta.json"
EMB_MODEL = os.environ.get("EMB_MODEL", "BAAI/bge-m3")

def read_docs(doc_dir):
    paths = sorted(glob.glob(os.path.join(doc_dir, "*.txt")))
    docs = [open(p, "r", encoding="utf-8").read().strip() for p in paths]
    return paths, docs

def main():
    paths, docs = read_docs(DOC_DIR)
    if not docs:
        raise RuntimeError("No docs in data/sample_docs")
    model = SentenceTransformer(EMB_MODEL)
    emb = model.encode(docs, normalize_embeddings=True, batch_size=32, convert_to_numpy=True)
    emb = emb.astype(np.float32)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, INDEX_OUT)
    with open(META_OUT, "w", encoding="utf-8") as f:
        json.dump([{"path": p, "text": t} for p,t in zip(paths, docs)], f, ensure_ascii=False, indent=2)
    print(f"Indexed {len(docs)} docs with {EMB_MODEL}")

if __name__ == "__main__":
    main()