# utils/rag_pipeline.py
import os
import shutil 
from typing import List, Dict, Any
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Define paths
KB_PATH = "data/insurance_kb.md"
CHROMA_DB_DIR = "vectorstore/chroma_db"

def load_documents(file_path: str) -> List[Document]:
    """Loads documents from a given file path."""
    # Adjust path to be relative to the project root for execution from main.py
    abs_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", file_path)
    
    if abs_file_path.endswith(".md"):
        loader = TextLoader(abs_file_path, encoding="utf-8")
    elif abs_file_path.endswith(".pdf"):
        loader = PyPDFLoader(abs_file_path) # Requires 'pypdf'
    else:
        raise ValueError(f"Unsupported file type for {abs_file_path}")
    return loader.load()

def split_documents(documents: List[Document]) -> List[Document]:
    """Splits documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def create_vector_store(documents: List[Document], embeddings: GoogleGenerativeAIEmbeddings) -> Chroma:
    """Creates and persists a Chroma vector store from documents."""
    # Adjust path to be relative to the project root
    abs_db_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", CHROMA_DB_DIR)

    if os.path.exists(abs_db_dir):
        print(f"Removing existing Chroma DB at {abs_db_dir} for clean creation...")
        shutil.rmtree(abs_db_dir)
        
    db = Chroma.from_documents(documents, embeddings, persist_directory=abs_db_dir)
    db.persist()
    return db

def get_persisted_vector_store(embeddings: GoogleGenerativeAIEmbeddings) -> Chroma:
    """Retrieves an existing Chroma vector store."""
    # Adjust path to be relative to the project root
    abs_db_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", CHROMA_DB_DIR)

    if not os.path.exists(abs_db_dir) or not os.listdir(abs_db_dir):
        raise FileNotFoundError(f"Chroma DB not found at {abs_db_dir}. Please run 'ingest_documents' first.")
    
    return Chroma(persist_directory=abs_db_dir, embedding_function=embeddings)


def ingest_and_get_vector_store(embeddings: GoogleGenerativeAIEmbeddings, kb_path: str = KB_PATH) -> Chroma:
    """
    Loads, splits, and creates/updates the vector store for the knowledge base.
    This function rebuilds the DB completely if it already exists.
    """
    print(f"--- Ingesting documents from {kb_path} ---")
    documents = load_documents(kb_path)
    chunks = split_documents(documents)
    
    db = create_vector_store(chunks, embeddings)
        
    print(f"--- Document ingestion complete. {len(chunks)} chunks stored. ---")
    return db

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from config import GOOGLE_API_KEY
    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY is not set. Please set it in your .env file.")
        exit(1)

    print("--- Running RAG pipeline ingestion test ---")
    
    embeddings_test = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

    vector_db = ingest_and_get_vector_store(embeddings_test)
    print(f"Vector store has {vector_db._collection.count()} items.")

    print("\n--- Testing RAG retrieval ---")
    
    test_queries = [
        "What does comprehensive auto insurance cover?",
        "Explain different types of life insurance.",
        "What does health insurance cover?",
        "What is a premium?",
        "Do I need liability insurance?",
        "What is an insurance deductible?",
        "Car warranties?" 
    ]

    for q in test_queries:
        print(f"\nQUERY: {q}")
        relevant_docs = None
        try:
            vector_db_retrieval = get_persisted_vector_store(embeddings_test) 
            docs = vector_db_retrieval.similarity_search(q, k=5)
            if docs:
                relevant_docs = "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"Error during retrieval test: {e}")

        if relevant_docs:
            print(f"RETRIEVED:\n{relevant_docs[:500]}...")
        else:
            print("No relevant documents found.")