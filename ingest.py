import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_pdf(file_path: str):
    """Loads a PDF and splits it into optimal chunks for retrieval."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}")
    
    print(f"Loading PDF: {file_path}...")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    print(f"Loaded {len(documents)} pages. Splitting into chunks...")
    
    # Chunking strategy: 1000 characters per chunk with a 200 character overlap
    # This ensures context isn't cut off mid-sentence between chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Successfully generated {len(chunks)} text chunks.")
    return chunks

if __name__ == "__main__":
    # Test it with a sample PDF path (replace with your actual file path)
    sample_pdf = "OMC_Report_Sample_Cardio.pdf" 
    if os.path.exists(sample_pdf):
        chunks = process_pdf(sample_pdf)
    else:
        print(f"Place a PDF named 'sample.pdf' in this directory to test text processing.")