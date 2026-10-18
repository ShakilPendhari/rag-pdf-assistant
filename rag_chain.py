import os
from dotenv import load_dotenv
from vector_store import load_vector_store
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def ask_question(question: str):
    """Queries the RAG system and returns an LLM-generated answer based on the PDF."""
    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY not found.")
    
    # 1. Load the existing vector store retriever
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # 2. Initialize the Gemini LLM
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.2)
    
    # 3. Define the RAG prompt template (enforcing groundedness)
    template = """Answer the question based only on the following context:
{context}

Question: {question}

If you do not know the answer based on the context, state that you do not know. Keep the answer clear and concise:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Helper function to format retrieved documents into a single text block
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # 4. Build the LCEL (LangChain Expression Language) chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # 5. Invoke the chain
    print(f"\nAsking Question: '{question}'\n")
    response = rag_chain.invoke(question)
    return response

if __name__ == "__main__":
    test_question = "What medications or treatments was the patient started on?"
    answer = ask_question(test_question)
    print("--- Answer ---")
    print(answer)