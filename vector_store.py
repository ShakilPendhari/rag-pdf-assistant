import os
from dotenv import load_dotenv
from ingest import process_pdf
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables from .env file
load_dotenv()

PERSIST_DIRECTORY = "./chroma_db"

def create_vector_store(pdf_path: str):
    """Embeds PDF chunks using Gemini and stores them in ChromaDB."""
    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY not found in environment variables or .env file.")
    
    # Get the chunks from Phase 1
    chunks = process_pdf(pdf_path)
    
    print("Generating embeddings using Gemini and storing in ChromaDB...")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    
    # Create and persist the vector store locally
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    
    print(f"Vector store successfully created and saved at: {PERSIST_DIRECTORY}")
    return vector_store

def load_vector_store():
    """Loads an existing ChromaDB vector store from disk."""
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )
    return vector_store
    
if __name__ == "__main__":
    pdf_file = "OMC_Report_Sample_Cardio.pdf"
    if os.path.exists(pdf_file):
        vector_store = create_vector_store(pdf_file)
        
        # Test similarity search
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        query = "What is the main finding in the cardio report?"
        results = retriever.invoke(query)
        
        print(f"\nTest Query: '{query}'")
        print(f"Retrieved {len(results)} relevant chunks:")
        for i, res in enumerate(results):
            print(f"\n--- Chunk {i+1} ---")
            print(res.page_content[:300] + "...")
    else:
        print(f"File {pdf_file} not found.")