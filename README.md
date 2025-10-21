# Akbank GenAI RAG (Artifacts-Only)

**Veri seti indirmeden** çalışan Türkçe RAG uygulaması. Hazır **FAISS index** (`medipol_faiss.index`) ve **metadata** (`medipol_metadata.pkl`) ile sorgu anında benzer pasajları bulur ve **Gemini** ile yanıt üretir.

## Yapı
```
akbank-genai-rag-artifacts/
├─ app_from_artifacts.py
├─ requirements.txt
├─ README.md
├─ .gitignore
└─ artifacts/
   ├─ medipol_faiss.index        # sizin dosyanız
   └─ medipol_metadata.pkl       # sizin dosyanız
```

## Kurulum
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Ortam Değişkenleri
```bash
export GEMINI_API_KEY="YOUR_KEY"           # zorunlu
export GEMINI_MODEL="gemini-2.0-flash-exp" # opsiyonel
export EMBEDDING_MODEL="trmteb/turkish-embedding-model"  # query-time
export TOP_K=3                              # opsiyonel
```

## Çalıştırma
1) `artifacts/` klasörüne **kendi** `medipol_faiss.index` ve `medipol_metadata.pkl` dosyalarınızı koyun.
2) Çalıştırın:
```bash
python app_from_artifacts.py
```
Tarayıcı: `http://127.0.0.1:7860`

## Teknik Notlar
- FAISS index tipi: Inner Product (IP). Sorgu embeddingleri L2 normalize edilir → cosine eşdeğeri.
- `len(metadata)` == `index.ntotal` olmalı. Sıra **aynı** kalmalı.
- Metadata öğesi beklenen alanlar: `title`, `url`, `hospital`, `content`.

## Sorun Giderme
- **Dim mismatch**: Embedder modeli farklı → aynı modeli kullanın.
- **Index/metadata mismatch**: Dosyalar uyumsuz → doğru çiftleri yerleştirin.
- **Boş içerik**: `content` yoksa cevap kalitesi düşer.
