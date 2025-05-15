import streamlit as st
# app.py

import os
import time
import streamlit as st
from dotenv import load_dotenv

# ─── LangChain & Groq imports ────────────────────────────────────────────────
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

# ─── 1) Load environment & API key ───────────────────────────────────────────
load_dotenv()  # expects a .env in the same folder
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY in your .env file")
    st.stop()
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ─── 2) Initialize Groq LLM & Prompt ─────────────────────────────────────────
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Llama3-8b-8192")
prompt = ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context only.
Please provide the most accurate response.

<context>
{context}
</context>

Question: {input}
"""
)

# ─── 3) Vector‑store builder ──────────────────────────────────────────────────
def create_vector_embedding():
    # Load all PDFs under research_papers/
    loader = PyPDFDirectoryLoader("research_papers")
    docs = loader.load()
    st.info(f"🔖 Loaded {len(docs)} pages from PDF files")

    # Split into ~1,000‑token chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    st.info(f"✂️ Split into {len(chunks)} chunks (~1k tokens each)")

    # Embed locally via the core HuggingFaceEmbeddings (no extra API key needed)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Cache into session state
    st.session_state.vectors = vectorstore
    st.session_state.docs = docs
    st.session_state.chunks = chunks

    st.success("✅ Vector store initialized!")

# ─── 4) Streamlit UI ──────────────────────────────────────────────────────────
st.title("📚 RAG Q&A with Groq & Local HuggingFace Embeddings")

user_question = st.text_input("Enter your question about the research papers:")

if user_question:
    # Build the index on first query
    if "vectors" not in st.session_state:
        create_vector_embedding()

    # Assemble retrieval + generation chain
    doc_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    rag_chain = create_retrieval_chain(retriever, doc_chain)

    # Invoke and time
    start = time.time()
    result = rag_chain.invoke({"input": user_question})
    elapsed = time.time() - start

    st.markdown("**Answer:**")
    st.write(result["answer"])
    st.caption(f"⏱ Generated in {elapsed:.2f} s")

    with st.expander("🔍 Retrieved document chunks"):
        for chunk in result["context"]:
            st.write(chunk.page_content)
            st.markdown("---")

st.title("My RAG Q&A App")
# etc.
