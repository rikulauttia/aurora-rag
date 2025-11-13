# AuroraRAG

**Evaluation-First, Multilingual RAG Framework with Local Deployment Support**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AuroraRAG is a production-ready Retrieval-Augmented Generation (RAG) framework designed for observable search quality, multilingual support, and flexible deployment options. Built with transparency and measurability at its core, AuroraRAG enables teams to build, evaluate, and deploy RAG systems with confidence.

**Part of the Aurora Series:**  
[AuroraRAG](https://github.com/rikulauttia/aurora-rag) | [Aurora SAR Change Detection](https://github.com/rikulauttia/aurora-sar-change)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Evaluation Framework](#evaluation-framework)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Production Considerations](#production-considerations)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Modern RAG systems often provide subjectively good results but lack measurable performance indicators. AuroraRAG addresses this challenge by making search quality observable and answer generation reproducible through:

- **Clear Separation of Concerns**: Distinct retrieval, reranking, and generation stages
- **Transparent Scoring**: Inner-product similarity (FAISS) combined with cross-encoder reranking
- **Built-in Evaluation**: Information retrieval metrics (MRR, nDCG) for continuous improvement
- **Multilingual Excellence**: Optimized for Finnish and English with BGE-M3 model family
- **Flexible Deployment**: Supports both cloud-based and fully offline operation

### Use Cases

- Internal knowledge base search
- Policy and compliance Q&A systems
- Technical support assistants
- Multilingual document repositories
- Privacy-sensitive applications requiring local deployment

---

## Key Features

### Evaluation-First Design
Built-in information retrieval metrics (Mean Reciprocal Rank, Normalized Discounted Cumulative Gain) enable systematic evaluation of retrieval quality and A/B testing of system components.

### Multilingual Support
Leverages BAAI's BGE-M3 model family for robust multilingual retrieval and reranking, with particular strength in Finnish and English corpora.

### Local Deployment Ready
Operates entirely offline using Ollama for local LLM inference and CPU-optimized FAISS indexing, ensuring data privacy and eliminating cloud dependencies.

### Production-Ready API
FastAPI-based REST API with comprehensive endpoints for search, answer generation, and health monitoring, ready for integration into existing systems.

---

## Architecture

AuroraRAG implements a three-stage pipeline optimizing for both recall and precision:

```
┌─────────────────────────────────────────────────────────────────┐
│                         Query Input                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Dense Embedding Layer                         │
│                      (BGE-M3 Encoder)                           │
└────────────────────────┬────────────────────────────────────────┘
                         │ Vector Representations
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Fast Retrieval Stage                           │
│              (FAISS Inner Product Search)                        │
│                     Returns Top-K                                │
└────────────────────────┬────────────────────────────────────────┘
                         │ Candidate Passages + Similarity Scores
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Precision Reranking                            │
│            (BGE-reranker-v2-M3 Cross-Encoder)                   │
└────────────────────────┬────────────────────────────────────────┘
                         │ Reranked Results + Confidence Scores
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Answer Generation (Optional)                    │
│                     (Ollama LLM / API)                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Structured Response with Citations                  │
└─────────────────────────────────────────────────────────────────┘
```

### Design Rationale

- **BGE-M3 Embeddings**: State-of-the-art multilingual dense retrieval, effective across query lengths
- **BGE-reranker-v2-M3**: Powerful cross-encoder for improving precision on top-K candidates
- **FAISS IndexFlatIP**: Exact search baseline (upgradeable to approximate methods for scale)
- **Evaluation Pipeline**: Quantitative metrics (MRR@K, nDCG@K) over labeled query sets

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- (Optional) Ollama for local LLM inference

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/rikulauttia/aurora-rag.git
   cd aurora-rag
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -U pip
   pip install -r requirements.txt
   ```

### Building the Index

Initialize the FAISS index using the sample documents:

```bash
python inference/encode_index.py
```

This processes documents in `data/sample_docs/` and creates the searchable index.

### Local LLM Setup (Optional)

For offline answer generation with Ollama:

**macOS:**
```bash
brew install ollama
```

**Linux/Windows:**  
Follow instructions at [ollama.ai](https://ollama.ai)

**Configure and run:**
```bash
ollama serve  # Keep this running in a separate terminal
ollama pull qwen2.5:7b-instruct-q4_0
```

**Set environment variables:**
```bash
export OLLAMA_URL=http://127.0.0.1:11434
export OLLAMA_MODEL=qwen2.5:7b-instruct-q4_0
```

### Running the API

Start the FastAPI server:

```bash
uvicorn app.server:app --host 0.0.0.0 --port 8080 --reload
```

The API will be available at `http://127.0.0.1:8080`

### Quick Test

**Search endpoint (retrieval + reranking):**
```bash
curl -X POST http://127.0.0.1:8080/search \
  -H "Content-Type: application/json" \
  -d '{"query":"Missä voin opiskella koneoppimista Suomessa?", "k": 5}' | jq .
```

**Answer endpoint (with LLM generation):**
```bash
curl -X POST http://127.0.0.1:8080/answer \
  -H "Content-Type: application/json" \
  -d '{"query":"Missä voin opiskella koneoppimista Suomessa?"}' | jq .
```

---

## API Reference

### Endpoints

#### `POST /search`

Retrieve and rerank relevant passages without LLM generation.

**Request Body:**
```json
{
  "query": "string",
  "k": 5
}
```

**Response:**
```json
{
  "query": "Missä voin opiskella koneoppimista Suomessa?",
  "hits": [
    {
      "doc_id": 2,
      "text": "...relevant passage excerpt...",
      "path": "data/sample_docs/003.txt",
      "sim_ip": 0.41,
      "rerank": 3.22
    }
  ]
}
```

**Fields:**
- `sim_ip`: Inner product similarity score from FAISS
- `rerank`: Cross-encoder reranking score (higher is better)

#### `POST /answer`

Generate a natural language answer using retrieved passages.

**Request Body:**
```json
{
  "query": "string",
  "k": 5,
  "max_tokens": 512
}
```

**Response:**
```json
{
  "query": "...",
  "answer": "Generated answer with citations",
  "passages": [...],
  "metadata": {
    "model": "qwen2.5:7b-instruct-q4_0",
    "retrieval_count": 5
  }
}
```

#### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

---

## Evaluation Framework

AuroraRAG includes built-in information retrieval evaluation capabilities.

### Metrics

- **MRR@K** (Mean Reciprocal Rank): Measures the rank quality of the first relevant result
- **nDCG@K** (Normalized Discounted Cumulative Gain): Evaluates graded relevance distribution

### Running Evaluations

```bash
python -m training.eval_ir
```

**Example Output:**
```
MRR@5: 0.778
nDCG@5: 0.833
```

### Extending Evaluation

1. Create labeled query-document pairs in `training/eval_ir.py`
2. Define relevance judgments (binary or graded)
3. Run evaluation after system modifications
4. Compare metrics to quantify improvements

**Best Practices:**
- Maintain a diverse test set covering common query patterns
- Include edge cases and multilingual queries
- Re-evaluate after changes to embeddings, reranking, or chunking strategy
- Track metrics over time to detect regressions

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EMB_MODEL` | `BAAI/bge-m3` | Hugging Face embedding model identifier |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranking model |
| `K` | `5` | Number of candidates retrieved from FAISS |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `qwen2.5:7b-instruct-q4_0` | LLM model for answer generation |

### Customizing Models

To use different models, update environment variables before starting the server:

```bash
export EMB_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
export RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

Note: Ensure models are compatible with the expected input/output format.

---

## Deployment

### Docker Deployment

**Build the image:**
```bash
docker build -t aurora-rag .
```

**Run the container:**
```bash
docker run -p 8000:8000 aurora-rag
```

The API will be accessible at `http://127.0.0.1:8000`

The Docker image automatically builds the FAISS index on first run if not pre-built.

### Hugging Face Spaces

For a web-based demo interface, deploy to Hugging Face Spaces:

1. Create a Space repository
2. Include: `app.py`, `requirements.txt`, `index.faiss` (optional), `meta.json` (optional)
3. Push to trigger automatic deployment

**Update Space:**
```bash
cd aurora-rag-space
git add . && git commit -m "Update Space configuration" && git push
```

---

## Production Considerations

### Scaling Retrieval

**Large-scale indexing (>1M documents):**
- Replace `IndexFlatIP` with `IndexIVFFlat` or `IndexHNSWFlat`
- Train index quantizers on representative data samples
- Implement index sharding for distributed search
- Persist FAISS indices to object storage (S3, GCS)

### Optimizing Chunking

- Target 256-512 token chunks with sentence boundary preservation
- Include document metadata (titles, sections) for context-aware reranking
- Experiment with overlapping chunks for long documents
- Maintain chunk-to-source mappings for citation generation

### Reranking Optimization

- Batch query-passage pairs to maximize GPU utilization
- Keep reranker models in memory (avoid reload overhead)
- Consider INT8 quantization for throughput improvements
- Implement caching for frequent query patterns

### LLM Integration

- Use structured prompts with clear instructions and formatting
- Enforce maximum context length to prevent truncation issues
- Implement evidence quality checks (refuse low-confidence answers)
- Add citation validation to ensure answer grounding

### Observability

**Key metrics to monitor:**
- Query latency by pipeline stage (retrieval, reranking, generation)
- Hit rate and median rank of first relevant document
- Cache hit rates for embeddings and results
- Model inference times and resource utilization

### Security and Privacy

- Implement input validation and sanitization
- Avoid indexing personally identifiable information (PII)
- Add content filtering for sensitive domains
- Use allowlists for external API calls
- Implement rate limiting and authentication for production APIs

---

## Project Structure

```
aurora-rag/
├── app/
│   └── server.py              # FastAPI application with REST endpoints
├── inference/
│   ├── encode_index.py        # FAISS index builder
│   ├── search.py              # Retrieval and reranking logic
│   └── generate.py            # LLM inference wrapper
├── training/
│   ├── eval_ir.py             # IR evaluation metrics (MRR, nDCG)
│   └── (additional utilities)
├── data/
│   └── sample_docs/           # Example document corpus
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container definition
└── README.md                  # This file
```

---

## Troubleshooting

### Common Issues

**`ModuleNotFoundError` for `inference` or `training`**

Ensure `__init__.py` files exist in module directories and run modules with the `-m` flag:
```bash
python -m training.eval_ir
```

**Long initial startup time**

First run downloads models from Hugging Face Hub. To mitigate:
- Pin model versions in `requirements.txt`
- Pre-cache models in CI/CD pipeline
- Use Docker images with pre-downloaded models

**Ollama not being used for answer generation**

Verify environment variables are set:
```bash
echo $OLLAMA_URL
echo $OLLAMA_MODEL
```

If not configured, `/answer` endpoint returns synthesized responses from retrieved passages without LLM generation.

**FAISS index not found**

Run the index builder before starting the API:
```bash
python inference/encode_index.py
```

### Getting Help

For additional support:
1. Check existing [GitHub Issues](https://github.com/rikulauttia/aurora-rag/issues)
2. Review API documentation at `/docs` when server is running
3. Create a new issue with:
   - Reproduction steps
   - Error logs
   - Output of `pip freeze`
   - System information (OS, Python version)

---

## Contributing

We welcome contributions! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes with clear commit messages
4. Add or update tests as needed
5. Submit a pull request

For major changes, please open an issue first to discuss the proposed modifications.

### Development Setup

```bash
pip install -r requirements-dev.txt  # If available
pre-commit install                     # If using pre-commit hooks
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

AuroraRAG builds upon excellent open-source projects:

- [BAAI BGE Models](https://github.com/FlagOpen/FlagEmbedding) for multilingual embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [Ollama](https://ollama.ai) for local LLM inference
- [FastAPI](https://fastapi.tiangolo.com) for API framework

---

## Keywords

`rag` `faiss` `information-retrieval` `reranking` `transformers` `fastapi` `gradio` `ollama` `multilingual` `evaluation` `bge` `semantic-search`

---

**Questions about implementation, scaling, or evaluation strategies?**  
Open a [discussion](https://github.com/rikulauttia/aurora-rag/discussions) or reach out via GitHub Issues.