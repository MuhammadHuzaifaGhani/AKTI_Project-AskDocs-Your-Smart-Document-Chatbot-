
# 🧠 AskDocs – Your Smart Document Chatbot

## 📘 Overview
**AskDocs** is a Generative AI-powered chatbot that allows users to **chat with their documents** (PDF, TXT, PPTX).
Built using **LangChain**, **Google Gemini**, and **Streamlit**, it combines the power of **Retrieval-Augmented Generation (RAG)** and **semantic search** to provide accurate, context-aware answers directly from your uploaded files.

---

## 🚀 Features
- 📂 **Multi-format Uploads:** Supports PDF, TXT, and PPTX files.
- 🔍 **Smart Search:** Uses embeddings for semantic retrieval (not just keyword matching).
- 💬 **Chat Interface:** Simple and interactive UI using Streamlit.
- ⚡ **Dual Modes:**
  - **Fast Mode:** Quick retrieval with minimal processing.
  - **Accurate Mode:** Context compression for precise, well-structured answers.
- 🧠 **LLM-powered:** Generates human-like responses using Google Gemini.
- 🗃️ **Persistent Vector Storage:** Stores document embeddings using Chroma DB.

---

## 🧰 Tech Stack

| Component | Tool/Library | Description |
|------------|---------------|--------------|
| **Frontend/UI** | Streamlit | Interactive and user-friendly chatbot interface |
| **Framework** | LangChain | Orchestrates document processing and AI pipeline |
| **Embeddings** | HuggingFace Sentence Transformers | Converts text into semantic vectors |
| **Vector Store** | Chroma | Stores and retrieves embeddings efficiently |
| **LLM** | Google Gemini 2.5 Flash | Generates context-based intelligent responses |
| **File Handling** | PyMuPDF, python-pptx | Extracts text from PDFs and PowerPoint files |

---

## 🧩 How It Works

1. **Upload Document:** The user uploads a PDF, TXT, or PPTX file.
2. **Text Extraction:** Content is read and preprocessed using PyMuPDF or python-pptx.
3. **Embedding Creation:** Text chunks are converted into embeddings using HuggingFace.
4. **Vector Storage:** Chroma stores embeddings for quick retrieval.
5. **Query Processing:** User’s question is embedded and compared to document vectors.
6. **Response Generation:** Relevant context is passed to Gemini, which generates the final answer.

---

## 🛠️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/AskDocs.git
cd AskDocs

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # On macOS/Linux use: source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
Create a `.env` file and add:
GOOGLE_API_KEY=your_gemini_api_key

# 5. Run the app
streamlit run app.py
```

---

## ⚙️ Folder Structure

```
AskDocs/
│
├── app.py                # Main Streamlit app
├── requirements.txt      # Required Python packages
├── .env                  # API key and environment variables
├── data/                 # Uploaded documents
├── vectorstore/          # Chroma DB storage
└── README.md             # Project documentation
```

---

## 💡 Challenges Faced
- Managing **large document context** efficiently for Gemini API.
- Handling **multi-format text extraction** (PDF & PPTX parsing).
- Ensuring **accuracy vs. speed trade-off** between modes.
- Avoiding **token overflow errors** in context-heavy queries.

---

## 🎯 Future Improvements
- 🧩 Add support for DOCX and CSV formats.
- 🌐 Integrate Firebase for user chat history and authentication.
- 🗣️ Include voice-based query input.
- 🤖 Deploy on Streamlit Cloud or Hugging Face Spaces.

---

## 👨‍💻 Author
**Muhammad Huzaifa Ghani**
*Machine Learning & AI Engineer*
📧 muhammadhuzaifaghani101@gmail.com
🔗 [LinkedIn Profile](https://www.linkedin.com/in/muhammad-huzaifa-ghani-285a2a316/)
