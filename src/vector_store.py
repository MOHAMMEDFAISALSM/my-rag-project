from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

STORE_PATH = "faiss_store"

def get_embeddings():
    """
    Initializes the embedding model.
    """
    # This will download the model "all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_vector_store(documents, embeddings):
    """
    Builds and saves the FAISS vector store from documents.
    """
    print("Chunking documents...")
    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    print("Building vector store...")
    # Create the FAISS vector store from the chunks
    store = FAISS.from_documents(docs, embeddings)
    
    print(f"Saving vector store to {STORE_PATH}...")
    # Save the store locally
    store.save_local(STORE_PATH)
    print("Vector store built and saved.")
    return store

def load_vector_store(embeddings):
    """
    Loads an existing FAISS vector store.
    """
    if os.path.exists(STORE_PATH):
        print(f"Loading vector store from {STORE_PATH}...")
        # Add allow_dangerous_deserialization=True
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