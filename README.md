# 🤖 InsightDocs AI - Advanced PDF RAG Chatbot

An enterprise-ready **Retrieval-Augmented Generation (RAG)** assistant designed for document intelligence. This application allows users to upload multi-page PDF documents and extract precise, fact-backed insights instantly using a cutting-edge open-source AI stack.

🚀 **Live Demo:** https://huggingface.co/spaces/wassupshubham/pdf-ai-chatbot

---

## 💡 Core Features

* **Intelligent Document Ingestion:** Extracts and processes unstructured text from multi-page PDF files seamlessly.
* **Semantic Search & Retrieval:** Uses LangChain and ChromaDB to chunk, embed, and map document context in-memory.
* **Zero-Dependency Embeddings:** Utilizes local huggingface embeddings running directly inside the container memory—eliminating external API token dependency or 401 Unauthorized errors.
* **Ultra-Fast Inference:** Powered by Groq Cloud API running **Llama 3.1 (8B-Instant)** for near-zero latency responses.
* **Modern UI Experience:** Built with a highly responsive, clean Streamlit dashboard.

---

## 🛠️ Architecture & Tech Stack

This project implements a classic RAG architecture to ground the LLM's responses and completely eliminate AI hallucinations.

* **Orchestration Framework:** LangChain (Community & Core)
* **Vector Database:** ChromaDB (In-Memory Configuration)
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` via HuggingFaceEmbeddings
* **Large Language Model (LLM):** Llama-3.1-8b-instant via Groq Cloud
* **Frontend UI:** Streamlit Framework
* **Data Parsing:** PyPDF Reader

---

## ⚙️ Installation & Local Setup

Follow these simple steps to spin up the AI matrix on your local machine:

### 1. Clone the Repository
```bash
git clone [https://github.com/wassupshubham/PDF-AI-Chatbot.git](https://github.com/wassupshubham/PDF-AI-Chatbot.git)
cd PDF-AI-Chatbot
