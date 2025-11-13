import os, requests

OLLAMA_URL   = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_0")

SYS = (
    "You are AuroraRAG, a helpful assistant. "
    "Use ONLY the provided CONTEXT to answer. "
    "Cite sources as [1], [2]. If context is insufficient, say so."
)

def _prompt(query: str, hits: list[dict]) -> str:
    ctx = "\n\n".join(f"[{i+1}] {h['text']}" for i, h in enumerate(hits))
    return (
        f"<|system|>{SYS}\n"
        f"<|user|>CONTEXT:\n{ctx}\n\n"
        f"QUESTION: {query}\n\n"
        f"Answer in Finnish when the question is Finnish; otherwise answer in the question's language."
    )

def generate(query: str, hits: list[dict], temperature: float = 0.2) -> str:
    if not hits:
        return "En löytänyt tarpeeksi kontekstia vastaukseen."
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": _prompt(query, hits),
            "options": {"temperature": temperature},
            "stream": False,
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json().get("response", "")