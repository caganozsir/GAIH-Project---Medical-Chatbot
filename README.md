# 🏥 Medipol RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot built with **Google Gemini**, **FAISS**, and **Gradio** — fine-tuned for answering medical questions using a dataset consisting verified articles from **İstanbul Medipol University’s health blog**.

This project allows users to ask natural-language questions about health topics, and the system retrieves the most relevant passages from Medipol’s medical articles before generating a comprehensive answer with citations.

---

## 📘 Project Overview

This repository contains:
- **`app_from_artifacts.py`** → The main Gradio web app that loads the precomputed FAISS index and serves the chatbot UI.
- **`Akbank_GenAI_Bootcamp.ipynb`** → The original Colab notebook where the **embeddings** and **FAISS index** were computed.
- **`artifacts/`** → Precomputed data extracted from the Colab environment:
  - `medipol_faiss.index` — FAISS vector index of embedded article chunks.
  - `medipol_metadata.pkl` — Corresponding article metadata (title, URL, content, hospital).

The `.pkl` and `.index` files were **exported from Google Colab** to make the app reproducible without recomputing embeddings.

---

## 🧠 How It Works

### 1. **Retrieval Pipeline**
When a user asks a question:
1. The query is embedded using a **SentenceTransformer** (`trmteb/turkish-embedding-model`).
2. The vector is searched against the **FAISS index** `medipol_faiss.index`.
3. Top `k` most relevant passages are retrieved from `medipol_metadata.pkl`.

### 2. **Prompt Construction**
The retrieved contexts are joined into a formatted prompt containing:
- A **summary request** + **bullet-point details**
- A reminder to avoid giving direct medical advice
- Citations under “Kaynaklar”

This ensures structured, responsible, and well-attributed responses.

### 3. **Generation**
The prompt is sent to **Google Gemini 2.0 Flash** (via the `google-generativeai` API), which generates the final Turkish response with context-grounded information.

### 4. **Interface**
The frontend is built with **Gradio Blocks**, featuring:
- A clean **dark mode UI** 🌙  
- Header banner and chatbot bubbles 💬  
- Example questions preloaded for convenience  
- `Enter` key to send messages  
- “Sohbeti Temizle” button to reset the conversation  

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```
git clone https://github.com/<yourusername>/akbank-genai-rag.git
cd akbank-genai-rag
```
### 2. Create Virtual Environment
```
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# OR
.venv\Scripts\activate           # Windows PowerShell
```
### 3. Install Dependencies
```
pip install -r requirements.txt
```
### 4. Add Your Gemini API Key
Create a .env file in the project root and write:
```
GEMINI_API_KEY=your_api_key_here
```

You can optionally adjust parameters:
```
TOP_K=3
EMBEDDING_MODEL=trmteb/turkish-embedding-model
GEMINI_MODEL=gemini-2.0-flash-exp
```
### 5. Run the App
```
python app_from_artifacts.py
```
Access it at:
👉 http://localhost:7860


## 🧩 Code Structure

| File                           | Description                                                                                                        |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| `app_from_artifacts.py`        | Loads FAISS + metadata, handles retrieval, builds prompts, runs Gemini model, and serves the Gradio interface.     |
| `notebook_medipol_faiss.ipynb` | Preprocessing and embedding of medical articles. Generates `medipol_faiss.index` and `medipol_metadata.pkl` files. |
| `artifacts/`                   | Contains extracted FAISS and metadata files (downloaded from Colab).                                               |
| `.env`                         | Stores your private API key (excluded from GitHub).                                                                |
| `requirements.txt`             | Lists required Python packages.                                                                                    |

## 💡 Example Question

Question: Bel fıtığı tedavi yöntemleri nelerdir?

Chatbot Answer: 

Bel fıtığı tedavisinde hem cerrahi hem de cerrahi olmayan çeşitli yöntemler mevcuttur. Tedavi yaklaşımı genellikle fıtığın şiddetine, hastanın semptomlarına ve genel sağlık durumuna göre belirlenir.

**Cerrahi Olmayan Tedavi Yöntemleri:**
Bel fıtığı vakalarının büyük bir kısmı cerrahi olmayan yöntemlerle tedavi edilebilir. Bu yöntemler genellikle semptomları hafifletmeyi, ağrıyı azaltmayı ve hastanın fonksiyonel yeteneklerini geri kazandırmayı hedefler.

* **İlaç Tedavisi:** Ağrıyı ve inflamasyonu kontrol altına almak için çeşitli ilaçlar kullanılabilir. Bunlar arasında ağrı kesiciler (parasetamol, ibuprofen, naproksen gibi), kas gevşeticiler, non-steroidal antiinflamatuar ilaçlar (NSAID'ler) ve bazı durumlarda opioidler yer alır. Ayrıca sinir ağrısını azaltmaya yardımcı olan antidepresanlar ve antikonvülsanlar da reçete edilebilir. *(Lütfen bu ilaçların sadece doktor tavsiyesiyle kullanılması gerektiğini unutmayın.)*

* **Fizik Tedavi:** Fizyoterapistler, hastaların duruşunu düzeltmek, kasları güçlendirmek, esnekliği artırmak ve ağrıyı azaltmak için çeşitli egzersizler ve teknikler uygularlar. Fizik tedavi programları genellikle kişiye özel olarak tasarlanır ve bel bölgesini destekleyen kasları güçlendirmeye, doğru vücut mekaniklerini öğretmeye ve ağrıyı yönetme stratejileri sunmaya odaklanır.

* **Egzersiz:** Düzenli egzersiz, bel fıtığı semptomlarını hafifletmeye ve nüks riskini azaltmaya yardımcı olabilir. Egzersizler arasında yürüyüş, yüzme, bisiklet sürme gibi düşük etkili aktiviteler, germe egzersizleri ve karın ve sırt kaslarını güçlendiren egzersizler yer alır.

* **Manuel Terapi:** Manuel terapi, omurga ve çevresindeki yumuşak dokuların manipülasyonunu içerir. Bu teknikler, eklem hareketliliğini artırmaya, kas gerginliğini azaltmaya ve ağrıyı hafifletmeye yardımcı olabilir.

* **Enjeksiyonlar:** Bazı durumlarda, doktorlar kortikosteroid enjeksiyonları yaparak iltihabı azaltabilir ve ağrıyı hafifletebilirler. Bu enjeksiyonlar genellikle omurga çevresindeki epidural boşluğa veya sinir köklerinin yakınına yapılır.

* **Ağrı Yönetimi:** Kronik bel fıtığı ağrısı olan hastalar için ağrı yönetimi programları faydalı olabilir. Bu programlar, ilaç tedavisi, fizik tedavi, psikolojik danışmanlık ve alternatif tedavi yöntemlerini içerebilir.

* **Destekleyici Cihazlar:** Bazı hastalar için bel korsesi veya diğer destekleyici cihazlar kullanmak, bel bölgesini destekleyerek ağrıyı azaltmaya yardımcı olabilir.

**Cerrahi Tedavi Yöntemleri:**
Cerrahi tedavi genellikle cerrahi olmayan yöntemlerin başarısız olduğu, semptomların şiddetli olduğu veya sinir hasarının ilerlediği durumlarda düşünülür.

* **Diskektomi:** En sık yapılan cerrahi yöntemlerden biridir. Bu işlemde, fıtıklaşmış diskin sinirlere baskı yapan kısmı çıkarılır. Diskektomi genellikle minimal invaziv tekniklerle yapılabilir.

* **Laminotomi / Laminektomi:** Bu işlemlerde, omurilik kanalını daraltan lamina adı verilen kemik yapılarının bir kısmı veya tamamı çıkarılır. Bu, sinirlere daha fazla alan açarak baskıyı azaltmaya yardımcı olur.

* **Spinal Füzyon:** Omurların birbirine kaynaştırılması işlemidir. Genellikle diskektomi veya laminektomi ile birlikte uygulanır ve omurgayı stabilize etmeye yardımcı olur.

* **Disk Protezi:** Hasarlı diskin yerine yapay bir disk yerleştirilir. Bu, omurganın hareketliliğini korumaya yardımcı olabilir.

**Önemli Not:**
Bel fıtığı tedavisi kişiye özel olmalıdır ve bir uzman hekim tarafından belirlenmelidir. Bu bilgiler sadece genel bilgilendirme amaçlıdır ve tıbbi tavsiye olarak değerlendirilmemelidir. Herhangi bir sağlık sorununuz varsa, lütfen doktorunuza danışın.

**Kaynaklar:**

* [Bel Ağrısı] — https://medipol.com.tr/bilgi-kosesi/adan-zye-saglik/bel-agrisi
* [Bel Fıtığı Nedir? Tedavi Yöntemleri Nelerdir?] — https://medipol.com.tr/bilgi-kosesi/bunlari-biliyor-musunuz/bel-fitigi
* [Ameliyatsız Bel Fıtığı Tedavi Yöntemleri ve Etkili Çözümler] — https://medipol.com.tr/bilgi-kosesi/adan-zye-saglik/ameliyatsiz-bel-fitigi-tedavi-yontemleri-ve-etkili-cozumler


## 🧩 Acknowledgments

Medical Article Dataset at https://huggingface.co/datasets/umutertugrul/turkish-hospital-medical-articles
Developed for the Akbank GenAI Bootcamp (2025).
Author: Çağan Özsır

## 🧾 License
This repository is distributed for educational and research use only.
Do not use the chatbot for real medical advice. Always consult healthcare professionals for medical concerns.
