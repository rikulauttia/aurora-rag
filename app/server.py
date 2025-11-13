from fastapi import FastAPI
from pathlib import Path
from pydantic import BaseModel

# build index if missing
from inference.encode_index import main as build_index
from inference.search import AuroraSearcher
from inference.generate import generate  # <-- NEW

INDEX = Path("inference/index.faiss")
META  = Path("inference/meta.json")

if not (INDEX.exists() and META.exists()):
    Path("data/sample_docs").mkdir(parents=True, exist_ok=True)
    if not any(Path("data/sample_docs").glob("*.txt")):
        Path("data/sample_docs/001.txt").write_text(
            "Root Expo on Turussa järjestettävä IT-ura- ja rekrymessu...", encoding="utf-8"
        )
        Path("data/sample_docs/002.txt").write_text(
            "Since AI Hackathon on 72 tunnin AI-hackathon Turussa...", encoding="utf-8"
        )
        Path("data/sample_docs/003.txt").write_text(
            "Aalto-yliopiston Machine Learning, Data Science and Artificial Intelligence...", encoding="utf-8"
        )
    build_index()

app = FastAPI(title="AuroraRAG API")
searcher = AuroraSearcher(top_k=5)

class Query(BaseModel):
    query: str
    k: int | None = 5

@app.post("/search")
def search(q: Query):
    return {"query": q.query, "results": searcher.search(q.query)}

@app.post("/answer")
def answer(q: Query):
    hits = searcher.search(q.query)
    text = generate(q.query, hits)
    return {"query": q.query, "answer": text, "sources": hits}
