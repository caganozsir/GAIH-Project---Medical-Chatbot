# app_from_artifacts.py
import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import gradio as gr

load_dotenv()  # reads .env from the project root

CFG = {
    "top_k": int(os.getenv("TOP_K", "3")),
    "embedding_model": os.getenv("EMBEDDING_MODEL", "trmteb/turkish-embedding-model"),
    "gemini_model": os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp"),
    "faiss_index_path": os.getenv("FAISS_PATH", "artifacts/medipol_faiss.index"),
    "meta_path_pkl": os.getenv("META_PATH_PKL", "artifacts/medipol_metadata.pkl"),
    "meta_path_jsonl": os.getenv("META_PATH_JSONL", "artifacts/medipol_metadata.jsonl"),
}

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY (e.g., export GEMINI_API_KEY=...)")
genai.configure(api_key=GEMINI_API_KEY)

def load_metadata(meta_pkl: str, meta_jsonl: str):
    if Path(meta_pkl).exists():
        with open(meta_pkl, "rb") as f:
            meta = pickle.load(f)
        if isinstance(meta, dict) and "items" in meta:
            meta = meta["items"]
        if not isinstance(meta, list):
            raise ValueError("Unexpected metadata.pkl structure (not a list).")
        return meta
    if Path(meta_jsonl).exists():
        items = []
        with open(meta_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
    raise FileNotFoundError("Metadata not found. Put medipol_metadata.pkl under artifacts/.")

print("Loading FAISS index:", CFG["faiss_index_path"])
index = faiss.read_index(CFG["faiss_index_path"])

print("Loading metadata:", CFG["meta_path_pkl"])
metadata = load_metadata(CFG["meta_path_pkl"], CFG["meta_path_jsonl"])

if len(metadata) != index.ntotal:
    raise ValueError(f"Index/metadata mismatch: index.ntotal={index.ntotal}, metadata={len(metadata)}")

def _norm(m):
    return {
        "title": m.get("title", "(Başlık Yok)"),
        "url": m.get("url", ""),
        "hospital": m.get("hospital", ""),
        "content": m.get("content", ""),
    }

metadata = [_norm(m) for m in metadata]
chunk_texts = [m["content"] for m in metadata]

embedder = SentenceTransformer(CFG["embedding_model"])

def encode_query(q: str) -> np.ndarray:
    q_emb = embedder.encode([q], normalize_embeddings=True)
    return np.asarray(q_emb, dtype=np.float32)

# sanity check: embedding dim must match index dim
_tmp = encode_query("test")
if _tmp.shape[1] != index.d:
    raise ValueError(f"Dim mismatch: query dim={_tmp.shape[1]} vs index.d={index.d}")

def retrieve(query: str, k: int = CFG["top_k"]):
    q = encode_query(query)
    D, I = index.search(q, k)
    results = []
    for rank, (idx, score) in enumerate(zip(I[0], D[0]), start=1):
        md = metadata[idx]
        results.append({
            "rank": rank,
            "score": float(score),
            "title": md["title"],
            "url": md["url"],
            "hospital": md["hospital"],
            "content": md["content"] if md["content"] else chunk_texts[idx],
        })
    return results

def build_prompt(question: str, contexts):
    ctx_block = "\n\n".join([f"Passaj {i+1}:\n{c['content']}" for i, c in enumerate(contexts)])
    src_block = "\n".join([f"- {c['title']} — {c['url']}" if c["url"] else f"- {c['title']}" for c in contexts])
    return f"""
Aşağıdaki içerik parçalarına dayanarak soruyu yanıtla.
Tıbbi sorular dışında bir soru gelirse "Benim alanım değil ama" dedikten sonra bildiğin kadarıyla cevap ver.
**Kısa değil, kapsamlı** bir yanıt ver: önce 1–2 cümlelik özet, ardından **madde işaretleri** ile detaylar.
Yetersiz bilgi varsa "Bilmiyorum" veya "Belgelerde yeterli bilgi yok" de.
Tıbbi tavsiye verme; genel bilgilendirme yap ve uzman görüşüne yönlendir.
Cevabın sonunda "Kaynaklar:" altında başlık ve URL ver.

Soru:
{question}

Bağlam:
{ctx_block}

Cevap ve ardından "Kaynaklar:":
Kaynaklar:
{src_block}
    """.strip()

def generate_answer(question: str):
    ctxs = retrieve(question, CFG["top_k"])
    prompt = build_prompt(question, ctxs)
    model = genai.GenerativeModel(CFG["gemini_model"])
    resp = model.generate_content(prompt)
    text = resp.text or ""
    return text, ctxs

WELCOME_HTML = """<div class='welcome-msg'>
👋 Merhaba!<br><br>
Ben <b>Medipol Chatbot</b>.<br>
Tıbbi makalelerden derlenmiş bilgilerle sorularınıza yanıt veririm.<br><br>
<i>Örnek sorular:</i><br>
• Bel fıtığı tedavi yöntemleri nelerdir?<br>
• Kütletme sağlıklı mıdır?<br>
• Migren atağı için kanıta dayalı yaklaşımlar neler?<br>
</div>"""

def chat_fn(message, history):
    try:
        answer, _ctxs = generate_answer(message)
        # history is a list of {"role": "...", "content": "..."} dicts with type="messages"
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ]
        return history, gr.update(value="")
    except Exception as e:
        err = f"⚠️ Hata: {e}"
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": err},
        ]
        return history, gr.update(value="")

def clear_chat():
    return [{"role": "assistant", "content": WELCOME_HTML}]

with gr.Blocks(
    title="Medipol RAG Chatbot",
    theme=gr.themes.Soft(),
    css="""
    body { background-color: #0f1218; } /* dark background */
    .app-header {
        display:flex; align-items:center; gap:14px;
        padding:18px 20px; border-radius:16px;
        background:linear-gradient(135deg,#1e88ff 0%,#00b8a9 100%);
        color:white; box-shadow:0 4px 12px rgba(0,0,0,0.25);
        margin-bottom:16px;
    }
    .app-header h1 { font-size:1.4rem; margin:0; color:white; }
    .app-subtle { color:rgba(255,255,255,0.92); font-size:0.95rem; margin:3px 0 0 0; }
    .welcome-msg { text-align:center; line-height:1.6; }
    .welcome-msg b { color:#1e88ff; }
    """
) as demo:
    # Header
    gr.HTML("""
        <div class="app-header">
            <div style="font-size:1.8rem">🏥</div>
            <div>
                <h1>Medipol Medical Articles — Chatbot</h1>
                <div class="app-subtle">Bilgilendirme amaçlıdır; tıbbi tavsiye değildir.</div>
            </div>
        </div>
    """)

    # Chatbot (messages API)
    chat = gr.Chatbot(
        label="Sohbet",
        height=480,
        type="messages",
        value=[{"role": "assistant", "content": WELCOME_HTML}],
        sanitize_html=False,  # allow HTML for centered welcome
        show_copy_button=True
    )

    msg = gr.Textbox(
        label="Soru",
        placeholder="Örn: Bel fıtığı tedavi yöntemleri nelerdir?",
        lines=1,
        max_lines=4
    )

    with gr.Row():
        send_btn = gr.Button("Yanıtla", variant="primary")
        clear_btn = gr.Button("Sohbeti Temizle", variant="secondary")

    msg.submit(chat_fn, inputs=[msg, chat], outputs=[chat, msg])
    send_btn.click(chat_fn, inputs=[msg, chat], outputs=[chat, msg])
    clear_btn.click(fn=clear_chat, inputs=None, outputs=chat)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    # queue BEFORE a single launch
    demo.queue(max_size=64).launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
        share=True
    )
