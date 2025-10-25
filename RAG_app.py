

import streamlit as st
import os
import shutil
import tempfile
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader,TextLoader,UnstructuredPowerPointLoader,
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ------------------------------
# 🌍 Load Environment
# ------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("⚠️ GOOGLE_API_KEY not found! Please add it to your .env file.")
    st.stop()

# ------------------------------
# ⚙️ Config
# ------------------------------
CHROMA_DIR = "./chroma_db"
MAX_MEMORY_TURNS = 10

# ------------------------------
# 🧩 Load Documents Helper
# ------------------------------
def load_documents(uploaded_files):
    docs = []
    for file in uploaded_files:
        temp_path = os.path.join(tempfile.gettempdir(), file.name)
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())

        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
        elif file.name.endswith(".txt"):
            loader = TextLoader(temp_path, encoding="utf-8")
        elif file.name.endswith(".pptx"):
            loader = UnstructuredPowerPointLoader(temp_path)
        else:
            continue
        docs.extend(loader.load())
    return docs


# ------------------------------
# 🧠 Embeddings
# ------------------------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

embeddings = get_embeddings()

# ------------------------------
# 💾 Session State
# ------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# ------------------------------
# 🧭 Sidebar Controls
# ------------------------------
st.sidebar.header("⚙️ Controls")

mode = st.sidebar.radio("Choose Mode:", ["🚀 Fast Mode", "🎯 Accurate Mode"])

uploaded_files = st.sidebar.file_uploader(
    "📂 Upload PDF/TXT/PPTX files",
    type=["pdf", "txt", "pptx"],
    accept_multiple_files=True,
)

# Clear controls
if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

if st.sidebar.button("🗑️ Clear Database"):
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        st.session_state.retriever = None
        st.session_state.chat_history = []
        st.sidebar.success("✅ Database and chat cleared!")
    else:
        st.sidebar.warning("⚠️ No database found.")

# ------------------------------
# 🧾 Document Processing
# ------------------------------
if uploaded_files:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=250 if mode == "🚀 Fast Mode" else 800,
        chunk_overlap=30 if mode == "🚀 Fast Mode" else 100,
    )

    documents = load_documents(uploaded_files)
    split_docs = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(split_docs, embeddings, persist_directory=CHROMA_DIR)
    vectorstore.persist()

    if mode == "🎯 Accurate Mode":
        llm_for_compression = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.2
        )
        compressor = LLMChainExtractor.from_llm(llm_for_compression)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        )
    else:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    st.session_state.retriever = retriever
    st.sidebar.success(f"✅ Docs indexed in **{mode}** mode!")

# ------------------------------
# 💬 Main Chat Interface
# ------------------------------
st.title("📚AskDocs – Your Smart Multimodal Chatbot ")

# Display previous messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------------------
# 🎙️ Voice Input Section
# ------------------------------
st.subheader("🎙️ Speak Your Question")
voice_input = st.audio_input("🎤 Record your voice")

voice_question = None
if voice_input:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(voice_input.getbuffer())
        tmp_path = tmp.name

    with st.spinner("🎧 Transcribing voice..."):
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(tmp_path)
        voice_question = " ".join([seg.text for seg in segments])

    st.success(f"🗣️ You said: {voice_question}")

# ------------------------------
# 💬 Text Chat Input (Moved Down)
# ------------------------------
st.markdown("---")
user_input = st.chat_input("Type your question here...")

# Prefer voice if available
question = voice_question or user_input

# ------------------------------
# 🤖 Response Generation
# ------------------------------
if question:
    # Display user message
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if not st.session_state.retriever:
        with st.chat_message("assistant"):
            st.error("⚠️ Please upload documents first!")
    else:
        retrieved_docs = st.session_state.retriever.invoke(question)
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        history_text = "\n".join([f"{t['role']}: {t['content']}" for t in st.session_state.chat_history[-MAX_MEMORY_TURNS * 2:]])

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.2
        )

        template = """
        You are a smart and reliable assistant. Use the chat history and retrieved context to answer. 
        If unsure, say so.

        Chat History:
        {history}

        Retrieved Context:
        {context}

        Question:
        {question}

        Answer Instructions:
        - Provide answers in this order:
          1. **LLM Answer:** Direct, concise.
          2. **Explanation:** Detailed reasoning and context.
          3. **Summary:** Short key takeaway.
        - Cite sources when relevant
        Answer:
        """



        prompt = PromptTemplate.from_template(template)
        final_prompt = prompt.invoke({
            "history": history_text,
            "context": context_text,
            "question": question
        })

        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                response = llm.invoke(final_prompt).content
                st.write_stream(iter([response]))

        st.session_state.chat_history.append({"role": "assistant", "content": response})

# ------------------------------
# 🪄 Footer
# ------------------------------
st.divider()
st.caption("⚡ Powered by Gemini + LangChain + HuggingFace + Streamlit + Whisper")

