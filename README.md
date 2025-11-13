# AuroraRAG — Eval-First, Multilingual RAG (Local-Ready)

**AuroraRAG** is a clean, production-lean Retrieval-Augmented Generation stack:
- **Embeddings:** `BAAI/bge-m3` (multilingual, strong retrieval)
- **Reranker:** `BAAI/bge-reranker-v2-m3` (multilingual cross-encoder)
- **LLM (local optional):** Qwen2.5 7B Instruct via **Ollama** (or Llama 3.1 8B)

## Why this repo?
- **Eval-first search.** We show both **inner-product** scores and **rerank** scores.
- **Multilingual by default.** Works well in Finnish, English, and more.
- **Local-friendly.** Runs fully offline with Ollama + CPU FAISS.

---

## Quickstart

### 1) Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 2) Build the index
Uses data from `data/sample_docs/*.txt`:
```bash
python inference/encode_index.py
```

### 3) (Optional) Local LLM with Ollama
```bash
# macOS
brew install ollama
ollama serve              # in a separate terminal
ollama pull qwen2.5:7b-instruct-q4_0

export OLLAMA_URL=http://127.0.0.1:11434
export OLLAMA_MODEL=qwen2.5:7b-instruct-q4_0
```

### 4) Run the API
```bash
uvicorn app.server:app --host 0.0.0.0 --port 8080 --reload
```

### 5) Test
```bash
# Search
curl -s -X POST http://127.0.0.1:8080/search \
  -H "Content-Type: application/json" \
  -d '{"query":"Missä voin opiskella koneoppimista Suomessa?"}' | jq .

# Answer (uses reranked hits + LLM if OLLAMA_* is set)
curl -s -X POST http://127.0.0.1:8080/answer \
  -H "Content-Type: application/json" \
  -d '{"query":"Missä voin opiskella koneoppimista Suomessa?"}' | jq .
```

---

## Docker

```bash
docker build -t aurora-rag .
docker run -p 8000:8000 aurora-rag
# API: http://127.0.0.1:8000
```

The image boots FastAPI and auto-builds the FAISS index if missing.

---

## Hugging Face Space (Gradio Demo)

**Space:** [https://huggingface.co/spaces/rikulauttia/aurora-rag-space](https://huggingface.co/spaces/rikulauttia/aurora-rag-space)

### Minimal files:
- `app.py`
- `requirements.txt`
- `index.faiss` + `meta.json` (optional for instant responses; otherwise the Space can build on first run)

### Update the Space:
```bash
cd aurora-rag-space
git add .
git commit -m "Update Space"
git push
```

---

## Project Layout

```
aurora-rag/
├── app/
│   └── server.py          # FastAPI with /search and /answer
├── inference/
│   ├── encode_index.py    # Build FAISS index
│   ├── search.py          # BGE-m3 retriever + multilingual reranker
│   └── generate.py        # LLM wrapper (Ollama)
├── data/
│   └── sample_docs/       # Tiny example docs
│       └── *.txt
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EMB_MODEL` | `BAAI/bge-m3` | Embedding model |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | Reranking model |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen2.5:7b-instruct-q4_0` | Ollama model name |

---

## Notes for Production

- **Scale FAISS:** Swap `IndexFlatIP` → `IVF/HNSW` for larger corpora.
- **Model Persistence:** Cache reranker & embedding models locally (avoid cold downloads).
- **Observability:** Add metrics for latency, top-k distributions, hit-rate, answer-length.
- **Guardrails:** Handle empty hits, refuse answers if context is weak.
- **Attribution:** Add token-level attribution or chunk-level citations for UI.

---

## Git Workflow

### GitHub (main repo)
```bash
git add .
git commit -m "Your message"
git push
```

### HF Space (separate clone)
```bash
cd aurora-rag-space/
git add .
git commit -m "Update Space"
git push
```

> **Note:** The venv only matters when running Python. Git operations work regardless of venv activation.

---

## License

MIT

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## Support

For issues and questions, please open an issue on GitHub.