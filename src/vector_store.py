from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

STORE_PATH = "faiss_store"

def get_embeddings():
    """Initializes the embedding model."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_vector_store(documents, embeddings):
    """Builds the FAISS vector store from documents (in-memory)."""
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    print("Building vector store...")
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    
    # Simple, direct approach
    store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    
    print("Vector store built successfully.")
    return store

def load_vector_store(embeddings):
    """Loads an existing FAISS vector store."""
    if os.path.exists(STORE_PATH):
        print(f"Loading vector store from {STORE_PATH}...")
        store = FAISS.load_local(
            STORE_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True 
        )
        print("Vector store loaded.")
        return store
    else:
        print("No vector store found.")
        return None