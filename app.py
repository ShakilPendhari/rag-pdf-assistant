import hashlib
import os
import tempfile
import time

import requests
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------- config
PERSIST_DIRECTORY = "./chroma_db_ollama"
OLLAMA_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"
NUM_CTX = 4096        # keep small on CPU: RAM and latency scale with it
BATCH_SIZE = 32       # smaller than 100: a crash loses less work on CPU
MAX_RETRIES = 3
TOP_K = 4

PROMPT_TEMPLATE = """Answer the question based only on the following context:
{context}

Question: {question}

If you do not know the answer based on the context, state that you do not know. Keep the answer clear and concise:"""


# ------------------------------------------------------------- resources
class NomicEmbeddings(OllamaEmbeddings):
    """nomic-embed-text expects task prefixes; without them retrieval quality drops."""

    def embed_documents(self, texts):
        return super().embed_documents([f"search_document: {t}" for t in texts])

    def embed_query(self, text):
        return super().embed_query(f"search_query: {text}")


@st.cache_resource
def get_embeddings() -> NomicEmbeddings:
    # num_gpu=0 forces CPU per request, independent of how the server was started
    return NomicEmbeddings(model=EMBED_MODEL, num_gpu=0)


@st.cache_resource
def get_vector_store() -> Chroma:
    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=get_embeddings(),
    )


@st.cache_resource
def get_llm() -> ChatOllama:
    return ChatOllama(
        model=LLM_MODEL,
        temperature=0.2,
        num_gpu=0,
        num_ctx=NUM_CTX,
    )


def ollama_is_up() -> bool:
    try:
        return requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).ok
    except requests.RequestException:
        return False


# ------------------------------------------------------------- ingestion
def add_batch_with_retry(store: Chroma, docs, ids) -> None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            store.add_documents(docs, ids=ids)
            return
        except Exception:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2**attempt)  # give Ollama time to recover/restart


def ingest_pdf(data: bytes, filename: str, progress) -> tuple[int, int]:
    """Idempotent, resumable ingestion. Returns (total_chunks, newly_added)."""
    file_hash = hashlib.sha256(data).hexdigest()[:16]

    # Close the handle before reading: Windows can't reopen an open temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        documents = PyPDFLoader(tmp_path).load()
    finally:
        os.remove(tmp_path)

    for doc in documents:
        doc.metadata["source"] = filename

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = [c for c in splitter.split_documents(documents) if c.page_content.strip()]

    # Deterministic IDs: the same PDF always maps to the same IDs, so a re-run
    # after a crash skips everything already stored instead of duplicating it.
    ids = [f"{file_hash}-{i}" for i in range(len(chunks))]

    store = get_vector_store()
    total, added = len(chunks), 0

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch_ids = ids[start:end]
        existing = set(store.get(ids=batch_ids)["ids"])

        todo = [(i, c) for i, c in zip(batch_ids, chunks[start:end]) if i not in existing]
        if todo:
            add_batch_with_retry(store, [c for _, c in todo], [i for i, _ in todo])
            added += len(todo)

        progress.progress(end / total, text=f"Embedded {end}/{total} chunks")

    return total, added


# -------------------------------------------------------------------- UI
st.set_page_config(page_title="Local PDF RAG (Ollama)", page_icon="🦙", layout="centered")
st.title("🦙 Chat with any PDF locally using Ollama")
st.write("Upload a PDF document below and ask questions completely offline with zero rate limits.")

if not ollama_is_up():
    st.error(f"Ollama is not reachable at {OLLAMA_URL}. Start it, then reload this page.")
    st.stop()

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None and st.button("Process PDF Locally"):
        progress = st.progress(0.0, text="Reading PDF...")
        try:
            total, added = ingest_pdf(uploaded_file.getvalue(), uploaded_file.name, progress)
            st.success(
                f"{uploaded_file.name}: {total} chunks total, {added} newly embedded, "
                f"{total - added} already indexed."
            )
        except Exception as e:
            st.error(
                "Ingestion failed after retries. Progress is saved; "
                f"press the button again to resume. Details: {e}"
            )

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
        store = get_vector_store()
        if not store.get(limit=1)["ids"]:
            st.warning("No documents indexed yet. Process a PDF first.")
            st.stop()

        with st.spinner("Thinking locally..."):
            try:
                docs = store.as_retriever(search_kwargs={"k": TOP_K}).invoke(query)
                context = "\n\n".join(d.page_content for d in docs)

                chain = ChatPromptTemplate.from_template(PROMPT_TEMPLATE) | get_llm() | StrOutputParser()
                response = chain.invoke({"context": context, "question": query})

                st.markdown(response)
                with st.expander("Sources"):
                    for d in docs:
                        page = d.metadata.get("page")
                        label = f"p.{page + 1}" if isinstance(page, int) else "?"
                        st.caption(f"{d.metadata.get('source', 'unknown')} ({label})")

                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Generation failed. Check that Ollama is running. Details: {e}")