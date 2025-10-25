# 📚 AskDocs – Your Smart Multimodal Chatbot

**AskDocs** is a multimodal chatbot that lets you **chat with your documents (PDF, TXT, PPTX)** using **Google Gemini 2.5 Flash**, **LangChain**, and **HuggingFace embeddings**.  
It also supports **voice input** through **Faster Whisper** — so you can talk to your documents just like you chat with a human assistant!

---

## 🚀 Features

✅ **Document Q&A** – Upload PDFs, TXT, or PowerPoint files and ask context-aware questions  
✅ **Voice + Text Input** – Type or speak your question  
✅ **Dual Modes**
- **🚀 Fast Mode:** Quick responses (light retrieval)
- **🎯 Accurate Mode:** Context compression with Gemini for detailed answers  
✅ **Semantic Search** – Powered by HuggingFace Sentence Transformers  
✅ **Persistent Vector Store** – Uses Chroma for fast retrieval  
✅ **Memory-Aware Chat** – Keeps previous turns for contextual conversations  
✅ **Clean Streamlit UI** – Sidebar controls, chat layout, and integrated mic  

---

## 🧠 Tech Stack

| Component | Library/Model |
|------------|----------------|
| LLM | Google **Gemini 2.5 Flash** |
| Framework | **LangChain** |
| Embeddings | **HuggingFace Sentence Transformers** |
| Vector Store | **ChromaDB** |
| Voice Transcription | **Faster Whisper** |
| UI | **Streamlit** |
| Environment | **Python 3.10+**, **dotenv** |

---

## 🗂️ Project Structure

```
AskDocs/
│
├── app.py                # Main Streamlit app
├── .env                  # API key and environment variables
├── chroma_db/            # Persistent vector database (auto-created)
├── requirements.txt      # Dependencies
└── README.md             # Documentation
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/AskDocs.git
cd AskDocs
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate       # (Linux/Mac)
venv\Scripts\activate        # (Windows)
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Add Google API Key
Create a `.env` file in the project root:
```bash
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

### 5️⃣ Run the App
```bash
streamlit run app.py
```

---

## 🗣️ How It Works

1. **Upload Documents** (PDF, TXT, or PPTX)  
2. **Choose Mode**
   - 🚀 *Fast Mode* → Quick embedding and retrieval  
   - 🎯 *Accurate Mode* → Context compression via Gemini  
3. **Ask Questions**
   - Type your question  
   - Or record voice (auto-transcribed with Faster Whisper)  
4. **Get Answers**
   - The app retrieves document chunks  
   - Sends context + history to Gemini  
   - Displays detailed answers with explanation & summary  

---

## 🧩 Key Components

- **LangChain Retriever:** Finds relevant chunks from your docs  
- **LLMChainExtractor:** Compresses long contexts intelligently  
- **WhisperModel:** Transcribes speech to text  
- **Chat History:** Keeps last 10 turns for context memory  
- **Streamlit UI:** Clean layout for both text and voice interaction  

---

## 📦 requirements.txt (example)

```
streamlit
langchain
langchain-google-genai
langchain-huggingface
langchain-community
python-dotenv
chromadb
faster-whisper
sentence-transformers
pypdf
python-pptx
unstructured
```

---

## 🪄 Example Use Case

> **Upload:** A company policy PDF  
> **Ask:** “What is the employee leave policy?”  
> **Gemini Responds:** Summarized answer + context citation from document  

Or use voice:  
> 🎙️ “What are the main financial terms in this report?”  
> ✅ Whisper → transcribes → Gemini answers contextually  

---

## 🧾 Future Improvements

- Add **UI mic + send icons** at chat bottom (modern style)  
- Integrate **PDF summarization** mode  
- Store **chat logs** in a database (SQLite/Firebase)  
- Deploy on **Streamlit Cloud / Hugging Face Spaces**  

---

## 👨‍💻 Author

**Muhammad Huzaifa Ghani**  
💼 Machine Learning & AI Engineer  
🌐 Vision: Innovate using ML, DL, NLP, and Generative AI  
📩 Open to collaborations & opportunities  

---

## ⚡ Powered By

**Gemini + LangChain + HuggingFace + Whisper + Streamlit**
