import streamlit as st
import os
import shutil
from dotenv import load_dotenv
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredPowerPointLoader

# Compression retriever imports
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ------------------------------
# Load API key from .env
# ------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("⚠️ GOOGLE_API_KEY not found! Please add it to your .env file.")
    st.stop()

# ------------------------------
# Config
# ------------------------------
CHROMA_DIR = "./chroma_db"  
MAX_MEMORY_TURNS = 10        

# ------------------------------
# Helper: Load and split documents
# ------------------------------
def load_documents(uploaded_files):
    docs = []
    for file in uploaded_files:
        temp_path = os.path.join(tempfile.gettempdir(), file.name)
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())

        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
            docs.extend(loader.load())
        elif file.name.endswith(".txt"):
            loader = TextLoader(temp_path, encoding="utf-8")
            docs.extend(loader.load())
        elif file.name.endswith(".pptx"):
            loader = UnstructuredPowerPointLoader(temp_path)
            docs.extend(loader.load())
    return docs

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("📚 AskDocs – Your Smart Document Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Embeddings
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

embeddings = get_embeddings()

# Sidebar mode selection
st.sidebar.header("⚙️ Controls")
mode = st.sidebar.radio("Choose Mode:", ["🚀 Fast Mode", "🎯 Accurate Mode"])

# Load existing Chroma
vectorstore = None
if os.path.exists(CHROMA_DIR):
    vectorstore = Chroma(embedding_function=embeddings, persist_directory=CHROMA_DIR)

    if mode == "🎯 Accurate Mode":
        llm_for_compression = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.2)
        compressor = LLMChainExtractor.from_llm(llm_for_compression)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
        )
    else:  # Fast mode
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    st.session_state.retriever = retriever
    st.sidebar.info(f"📂 Loaded Chroma DB in **{mode}**.")

# File uploader
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF/TXT/PPTX files",
    type=["pdf", "txt", "pptx"],
    accept_multiple_files=True
)

if uploaded_files:
    # Use smaller chunks in Fast Mode
    if mode == "🚀 Fast Mode":
        splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=30)
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    documents = load_documents(uploaded_files)
    split_docs = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(split_docs, embeddings, persist_directory=CHROMA_DIR)
    vectorstore.persist()

    if mode == "🎯 Accurate Mode":
        llm_for_compression = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.2)
        compressor = LLMChainExtractor.from_llm(llm_for_compression)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
        )
    else:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    st.session_state.retriever = retriever
    st.sidebar.success(f"✅ Docs uploaded and indexed in **{mode}**!")

# Reset buttons
if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state.chat_history = []
    st.sidebar.success("✅ Chat history cleared.")

if st.sidebar.button("🗑️ Clear Database"):
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        st.session_state.retriever = None
        st.session_state.chat_history = []
        st.sidebar.success("✅ Database and chat history cleared!")
    else:
        st.sidebar.warning("⚠️ No database found.")

# ------------------------------
# Chat Section
# ------------------------------
question = st.text_input("Ask a question:")

if st.button("Submit") and question:
    if not st.session_state.retriever:
        st.warning("⚠️ Please upload documents first!")
    else:
        retrieved_docs = st.session_state.retriever.invoke(question)
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

        recent_history = st.session_state.chat_history[-MAX_MEMORY_TURNS * 2:]
        history_text = "\n".join(
            [f"{turn['role']}: {turn['content']}" for turn in recent_history]
        )

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
        - Cite sources when relevant.

        Answer:
        """
        prompt = PromptTemplate.from_template(template)

        final_prompt = prompt.invoke({
            "history": history_text,
            "context": context_text,
            "question": question
        })

        response = llm.invoke(final_prompt).content

        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.chat_history.append({"role": "assistant", "content": response})

# # ------------------------------
# # Display Chat History
# # ------------------------------
# st.subheader("💬 Chat History")
# for turn in st.session_state.chat_history:
#     if turn["role"] == "user":
#         st.markdown(f"**You:** {turn['content']}")
#     else:
#         st.markdown(f"**Assistant:** {turn['content']}")
# ------------------------------
# Display Chat History (WhatsApp-style)
# ------------------------------
st.subheader("💬 Chat Conversation")

chat_css = """
<style>
.chat-message {
    padding: 0.6rem 1rem;
    border-radius: 1rem;
    margin-bottom: 0.7rem;
    max-width: 80%;
    word-wrap: break-word;
}
.user {
    background-color: #DCF8C6;
    color: black;
    margin-left: auto;
    text-align: right;
}
.assistant {
    background-color: #E6E6E6;
    color: black;
    margin-right: auto;
    text-align: left;
}
.chat-container {
    display: flex;
    flex-direction: column;
}
</style>
"""
st.markdown(chat_css, unsafe_allow_html=True)

st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for turn in st.session_state.chat_history:
    if turn["role"] == "user":
        st.markdown(f'<div class="chat-message user">💬 <b>You:</b> {turn["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message assistant">🤖 <b>Assistant:</b> {turn["content"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
