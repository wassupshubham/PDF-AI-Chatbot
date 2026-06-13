import os
import sys

# 1. Core Torch Imports right on top to resolve any upstream library type-hint bugs
import torch
import torch.nn as nn
sys.modules['nn'] = nn

import streamlit as st
from dotenv import load_dotenv
import io

# Load environment variables
load_dotenv()

# --- CORE RAG PIPELINE FUNCTIONS ---
import pdfplumber
from pypdf import PdfReader
import easyocr
from PIL import Image

@st.cache_resource
def load_local_ocr():
    return easyocr.Reader(['en'], gpu=False)

def extract_text_from_pdf(pdf_file):
    text = ""
    reader = load_local_ocr()
    
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text(layout=True)
                if page_text:
                    text += page_text + "\n"
        
        if not text.strip() or len(text.strip()) < 100:
            with st.spinner("🧠 Scanned Layout Detected. Initializing Deep Local OCR Engine..."):
                pdf_file.seek(0)
                pypdf_reader = PdfReader(pdf_file)
                
                for page_num, page in enumerate(pypdf_reader.pages):
                    if "/XObject" in page["/Resources"]:
                        xObject = page["/Resources"]["/XObject"].get_object()
                        for obj in xObject:
                            if xObject[obj]["/Subtype"] == "/Image":
                                data = xObject[obj].get_data()
                                img = Image.open(io.BytesIO(data))
                                results = reader.readtext(img, detail=0)
                                if results:
                                    text += f"\n--- Page {page_num+1} Scan ---\n" + " ".join(results) + "\n"
                                    
    except Exception as e:
        st.error(f"Local Extraction Layer Warning: {e}")
    return text

from langchain_text_splitters import RecursiveCharacterTextSplitter
def split_text(text):
    return RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150).split_text(text)

from langchain_huggingface import HuggingFaceEmbeddings
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 🔥 FIX: Using an explicit Ephemeral Client (Pure RAM In-Memory) to bypass HuggingFace's disk write protection
import chromadb
from langchain_chroma import Chroma

def create_vectorstore(chunks, embeddings):
    # EphemeralClient memory space mein initialize hota hai aur disk par touch nahi karta
    chroma_client = chromadb.EphemeralClient()
    
    return Chroma.from_texts(
        texts=chunks, 
        embedding=embeddings,
        client=chroma_client
    )

from langchain_groq import ChatGroq
def get_llm():
    return ChatGroq(model="llama-3.1-8b-instant", groq_api_key=os.getenv("GROQ_API_KEY"))


# --- SMARTPHONE OPTIMIZED UI INTERFACE ---
st.set_page_config(page_title="InsightDocs AI", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
    .main-title { font-size: 2rem; font-weight: 700; color: #1E3A8A; text-align: center; }
    .sub-title { font-size: 1rem; color: #4B5563; text-align: center; margin-bottom: 15px; }
    .answer-box { background-color: #F3F4F6; padding: 15px; border-radius: 8px; border-left: 5px solid #2563EB; font-size: 1rem; color: #1F2937; }
    div.stButton > button { width: 100% !important; margin-top: 5px; border-radius: 8px; height: 45px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🤖 InsightDocs AI Workspace</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Mobile-optimized deep document and image intelligence dashboard.</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Choose PDF File", type="pdf", label_visibility="collapsed")
with col2:
    if st.button("🔄 Clear System Session/Cache", type="primary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("System memory and in-memory databases wiped clean!")
        st.rerun()

st.markdown("---")

if uploaded_file is not None:
    # Clear session instantly if file name changes to prevent data leak history
    if "current_file_name" in st.session_state:
        if st.session_state.current_file_name != uploaded_file.name:
            if "vectorstore" in st.session_state:
                st.session_state.vectorstore = None
                del st.session_state["vectorstore"]
            st.session_state.current_file_name = uploaded_file.name
    else:
        st.session_state.current_file_name = uploaded_file.name

    if "vectorstore" not in st.session_state or st.session_state.vectorstore is None:
        with st.spinner("⚡ Processing complex document layout layers..."):
            text = extract_text_from_pdf(uploaded_file)
            if not text or not text.strip():
                st.error("❌ System Error: Unable to extract text data from this document layout. Please ensure the file contains valid digital text or images.")
                st.stop()
            chunks = split_text(text)
            st.session_state.vectorstore = create_vectorstore(chunks, get_embeddings())
        st.success(f"Loaded {len(chunks)} contextual nodes successfully via Isolated Local Deep parsing!")

    if "vectorstore" in st.session_state and st.session_state.vectorstore is not None:
        question = st.text_input("💬 Ask anything about this document:", placeholder="Type your question here...")

        if question:
            with st.spinner("Matching similarity nodes..."):
                try:
                    docs = st.session_state.vectorstore.similarity_search(question, k=6)
                    context = "\n\n".join([doc.page_content for doc in docs])

                    prompt = f"""
You are a highly precise Document Intelligence Assistant. Answer the user's question using ONLY the provided context block.

CRITICAL DIRECTIVES:
1. Do NOT reuse general titles, watermarks, or first-page metadata expressions (like \"KOSIS KOI NA KRRE...\") unless explicitly asked.
2. Answer comprehensively, detailing structural nuances, numbers, or metrics present in the matching sections.
3. If the context does not contain the answer, state that the information is missing. Do not hallucinate.

Context:
{context}

Question:
{question}
"""
                    llm = get_llm()
                    response = llm.invoke(prompt)

                    st.markdown("#### 💡 AI Response")
                    st.markdown(f'<div class="answer-box">{response.content}</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Inference Engine Error: {e}")
else:
    if "vectorstore" in st.session_state:
        del st.session_state["vectorstore"]
    if "current_file_name" in st.session_state:
        del st.session_state["current_file_name"]
    st.info("💡 Tap 'Choose PDF File' above to upload a document and activate the workspace.")
