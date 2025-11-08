from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    TextLoader,
)
import os

def load_documents(data_dir="data"):
    """
    Loads all supported documents from the specified directory.
    
    Supports: .pdf, .txt, .md
    """
    print(f"Loading documents from {data_dir}...")
    
    # Define the loaders for different file types
    pdf_loader = DirectoryLoader(
        data_dir, 
        glob="**/*.pdf",  # Search all subdirectories
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    
    txt_loader = DirectoryLoader(
        data_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True
    )
    
    md_loader = DirectoryLoader(
        data_dir,
        glob="**/*.md",
        loader_cls=TextLoader, # TextLoader works for .md
        show_progress=True
    )
    
    # Load all documents
    pdf_docs = pdf_loader.load()
    txt_docs = txt_loader.load()
    md_docs = md_loader.load()
    
    all_documents = pdf_docs + txt_docs + md_docs
    
    print(f"Loaded {len(all_documents)} documents ({len(pdf_docs)} PDFs, {len(txt_docs)} TXTs, {len(md_docs)} MDs).")
    return all_documents