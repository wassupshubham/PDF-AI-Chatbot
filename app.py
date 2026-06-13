import os
import sys
import streamlit as st
from dotenv import load_dotenv

# Load environment variables if configured
load_dotenv()

# --- ALL-IN-ONE RAG FUNCTIONS ---

# 1. PDF Loader Logic
from pypdf import PdfReader

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# 2. Text Splitter Logic
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)

# 3. Embeddings Logic (COMPLETELY CLEANED)
from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# 4. Vector Database Logic
from langchain_chroma import Chroma

def create_vectorstore(chunks, embeddings):
    return Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

# 5. LLM Model Logic
from langchain_groq import ChatGroq

def get_llm():
    # Your verified Groq API key
    api_key = "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"
    
    return ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=api_key
    )


# --- STREAMLIT UI INTERFACE ---

st.set_page_config(page_title="AI PDF Chatbot", layout="centered")

st.title("📚 AI PDF Chatbot")
st.write("Upload a PDF document and chat with it in plain English.")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type="pdf"
)

if uploaded_file is not None:
    # Use Session State so the PDF processing runs only once per upload
    if "vectorstore" not in st.session_state:
        with st.spinner("Processing PDF... Please wait..."):
            try:
                # Step 1: Extract Text
                text = extract_text_from_pdf(uploaded_file)

                # Step 2: Chunk Text
                chunks = split_text(text)

                # Step 3: Get Embedding Model
                embeddings = get_embeddings()

                # Step 4: Create Local Chroma Vector Store
                st.session_state.vectorstore = create_vectorstore(chunks, embeddings)
                st.success("✅ PDF processed and Vector DB created successfully!")
            except Exception as e:
                st.error(f"Error processing file: {e}")

    # Show chat input only if vector database is ready
    if "vectorstore" in st.session_state:
        question = st.text_input(
            "Ask a question about your PDF:",
            placeholder="Type your question here..."
        )

        if question:
            with st.spinner("Searching document chunks and generating answer..."):
                try:
                    # Step 5: Semantic similarity search (Get top 3 matching chunks)
                    docs = st.session_state.vectorstore.similarity_search(question, k=3)

                    # Step 6: Combine matching chunks into one string context
                    context = "\n\n".join([doc.page_content for doc in docs])

                    # Step 7: Build a clean context prompt
                    prompt = f"""
Answer the question using ONLY the context provided below. If you cannot find the answer in the context, say "I cannot find the answer in the uploaded document."

Context:
{context}

Question:
{question}
"""

                    # Step 8: Invoke Updated Groq Llama 3.1 Model
                    llm = get_llm()
                    response = llm.invoke(prompt)

                    # Step 9: Render the answer beautifully
                    st.subheader("💡 Answer")
                    st.write(response.content)

                except Exception as e:
                    st.error(f"Error generating answer: {e}")