# 🧪 RAG PDF Analysis System: Development & Migration Guide

This document provides a comprehensive step-by-step account of building a Retrieval-Augmented Generation (RAG) system from scratch. It documents the transition from cloud-based models (Google Gemini API) to a completely local, privacy-focused engine using **Ollama** and **Streamlit**.

---

## 📐 1. Architecture Overview

```
 [Uploaded PDF Document]
          │
          ▼
 [PyPDFLoader] ──► Extract Raw Text
          │
          ▼
 [RecursiveCharacterTextSplitter] ──► Chunk Text (Size: 1000, Overlap: 200)
          │
          ▼
 [Nomic-Embed-Text (Ollama)] ──► Generate Vector Embeddings
          │
          ▼
 [ChromaDB] ──► Store Embeddings Locally in `./chroma_db_ollama`
          │
          ▼
 [Retriever] ──► Fetch Top-K Relevant Chunks for User Query
          │
          ▼
 [Llama 3 (Ollama)] ──► Synthesize Final Grounded Answer
```

---

## 🛠️ 2. Environment Setup

### 2.1 Virtual Environment Management
Using Anaconda to manage system dependencies cleanly:

```bash
# Create dedicated Anaconda environment
conda create -n rag-env python=3.10 -y

# Activate environment
conda activate rag-env
```

### 2.2 Installing Dependencies
Install the required packages into your active environment:

```bash
# Core LangChain & Vector Store dependencies
pip install langchain \
            langchain-community \
            langchain-core \
            langchain-text-splitters \
            langchain-chroma \
            chromadb \
            pypdf \
            streamlit \
            python-dotenv

# API Integrations
pip install langchain-google-genai \
            langchain-ollama
```

---

## 🔄 3. Step-by-Step Implementation Journey

### Step 1: PDF Ingestion & Document Chunking (`ingest.py`)
In the initial phase, we extract text from local PDFs and divide it into manageable overlapping chunks.

```python
# ingest.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

if __name__ == "__main__":
    chunks = load_and_split_pdf("sample.pdf")
    print(f"Total chunks created: {len(chunks)}")
```

---

### Step 2: Cloud Vector DB Setup with Gemini API (`vector_store.py`)
Initial implementation utilized Google Gemini for embedding generation and storage in ChromaDB.

```python
# vector_store.py
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from ingest import load_and_split_pdf

load_dotenv()

def create_vector_store(pdf_path, persist_dir="./chroma_db"):
    chunks = load_and_split_pdf(pdf_path)
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    return vector_store

if __name__ == "__main__":
    create_vector_store("sample.pdf")
    print("Vector database built successfully with Gemini Embeddings.")
```

---

### Step 3: API Quota Challenge & Rate Limit Resolution

#### The Problem:
When batch-processing large PDFs or executing multiple queries via Google Gemini API's free tier, the application threw the following HTTP status exception:

```
GoogleGenerativeAIError: Error embedding content (RESOURCE_EXHAUSTED): 429
Quota exceeded for metric: generativelanguage.googleapis.com/embed_content_free_tier_requests, limit: 100
```

#### The Solution:
To achieve zero-cost, unlimited document processing without API rate limits, the architecture was pivoted to **Ollama** for local inference.

---

### Step 4: Configuring Local Ollama Infrastructure

1. **Install Ollama:** Download and install from [ollama.com](https://ollama.com/).
2. **Download Required Open-Source Models:**
   ```powershell
   # Embedding model specialized for retrieval tasks
   ollama pull nomic-embed-text

   # High-performance chat LLM
   ollama pull llama3
   ```
3. **Verify Background Service:** Ensure Ollama is running locally on port `11434`.

---

### Step 5: Terminal Testing Local RAG Pipeline (`rag_chain.py`)

```python
# rag_chain.py
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

PERSIST_DIRECTORY = "./chroma_db_ollama"

# Initialize local embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Load existing vector database
vector_store = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Initialize local LLM
llm = ChatOllama(model="llama3", temperature=0.2)

# Prompt Template
template = """Answer the question based only on the following context:
{context}

Question: {question}

If you do not know the answer based on the context, state that you do not know. Keep the answer clear and concise:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Construct LCEL Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    query = "What medications or treatments was the patient started on?"
    print(f"Asking Question: '{query}'\n--- Answer ---")
    response = rag_chain.invoke(query)
    print(response)
```

---

### Step 6: Full Web Application Interface (`app.py`)

The final user-facing interface built with Streamlit supports drag-and-drop PDF uploading and live offline interactive chat.

```python
# app.py
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

PERSIST_DIRECTORY = "./chroma_db_ollama"

st.set_page_config(page_title="Local PDF RAG (Ollama)", page_icon="🦙", layout="centered")

st.title("🦙 Chat with any PDF locally using Ollama")
st.write("Upload a PDF document below and ask questions completely offline with zero rate limits.")

# Sidebar for PDF Upload & Indexing
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Process PDF Locally"):
            with st.spinner("Processing and embedding PDF with Ollama..."):
                temp_pdf_path = f"./temp_{uploaded_file.name}"
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                loader = PyPDFLoader(temp_pdf_path)
                documents = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(documents)
                
                # Local Ollama Embeddings
                embeddings = OllamaEmbeddings(model="nomic-embed-text")
                vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=PERSIST_DIRECTORY
                )
                
                st.success(f"Successfully processed {len(chunks)} chunks from {uploaded_file.name}!")
                os.remove(temp_pdf_path)

# Main Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask a question about your PDF..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking locally..."):
            try:
                embeddings = OllamaEmbeddings(model="nomic-embed-text")
                vector_store = Chroma(
                    persist_directory=PERSIST_DIRECTORY,
                    embedding_function=embeddings
                )
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                
                # Local Ollama LLM
                llm = ChatOllama(model="llama3", temperature=0.2)
                
                template = """Answer the question based only on the following context:
{context}

Question: {question}

If you do not know the answer based on the context, state that you do not know. Keep the answer clear and concise:"""
                
                prompt = ChatPromptTemplate.from_template(template)
                
                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)
                
                rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                
                response = rag_chain.invoke(query)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"Error: Make sure Ollama is running. Details: {e}")
```

---

## 🚀 4. How to Run

1. Verify Ollama service status:
   ```powershell
   ollama list
   ```
2. Launch the Streamlit Web Application:
   ```powershell
   streamlit run app.py
   ```
3. Access the web client via `http://localhost:8501`.