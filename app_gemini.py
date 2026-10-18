import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
PERSIST_DIRECTORY = "./chroma_db"

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="🤖", layout="centered")

st.title("📄 Chat with any PDF using Gemini RAG")
st.write("Upload a PDF document below and ask questions about its contents instantly.")

# Sidebar for PDF Upload & Indexing
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Process PDF"):
            with st.spinner("Processing and embedding PDF..."):
                # Save uploaded file locally temporarily
                temp_pdf_path = f"./temp_{uploaded_file.name}"
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load & Split
                loader = PyPDFLoader(temp_pdf_path)
                documents = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(documents)
                
                # Embed & Store in ChromaDB
                embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
                vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=PERSIST_DIRECTORY
                )
                
                st.success(f"Successfully processed {len(chunks)} chunks from {uploaded_file.name}!")
                # Clean up temp file
                os.remove(temp_pdf_path)

# Main Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input query
if query := st.chat_input("Ask a question about your PDF..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Load vector store retriever
                embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
                vector_store = Chroma(
                    persist_directory=PERSIST_DIRECTORY,
                    embedding_function=embeddings
                )
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                
                # Initialize Gemini LLM
                llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.2)
                
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
                error_msg = f"Error: Please upload and process a PDF first using the sidebar. Details: {e}"
                st.error(error_msg)