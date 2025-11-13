import gradio as gr
from inference.search import AuroraSearcher

s = AuroraSearcher(top_k=5)

def ui_search(q):
    res = s.search(q)
    lines = []
    for r in res:
        lines.append(f"[rerank={r['rerank']:.3f} | ip={r['sim_ip']:.3f}] {r['text']}")
    return "\n\n".join(lines) if lines else "No results."

with gr.Blocks() as demo:
    gr.Markdown("# AuroraRAG — Eval-First RAG with LoRA Reranking")
    inp = gr.Textbox(label="Your question", value="Missä opiskella ML ja AI Suomessa?")
    out = gr.Textbox(label="Top results (explainable)")
    btn = gr.Button("Search")
    btn.click(ui_search, inputs=inp, outputs=out)

if __name__ == "__main__":
    demo.launch()
