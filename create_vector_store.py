#!/usr/bin/env python3
"""
Create the vector store from all documents
Run this script to build the RAG pipeline and create the vector store
"""
import os
import shutil
from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    PyPDFLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def create_vector_store():
    """Create the vector store from all documents"""
    
    print("="*70)
    print("CREATING RAG PIPELINE - VECTOR STORE")
    print("="*70)
    
    # Step 1: Clean existing vector store
    print("\n[Step 1] Cleaning existing vector store...")
    chroma_db_path = Path("data/chroma_db")
    if chroma_db_path.exists():
        print(f"  Deleting existing ChromaDB at: {chroma_db_path}")
        shutil.rmtree(chroma_db_path)
        print("  ✓ Deleted successfully")
    else:
        print(f"  No existing vector store found, proceeding...")
    
    # Step 2: Load documents
    print("\n[Step 2] Loading documents...")
    text_files_path = "data/text_files"
    pdf_files_path = "data/pdf"
    
    # Load text files
    text_documents = []
    if os.path.exists(text_files_path):
        print(f"  Loading text files from: {text_files_path}")
        text_loader = DirectoryLoader(
            text_files_path,
            glob="*.txt",
            loader_cls=TextLoader,
            show_progress=True
        )
        text_documents = text_loader.load()
        print(f"  ✓ Loaded {len(text_documents)} text documents")
    else:
        print(f"  ✗ Directory not found: {text_files_path}")
    
    # Load PDF files
    pdf_documents = []
    if os.path.exists(pdf_files_path):
        print(f"  Loading PDF files from: {pdf_files_path}")
        pdf_loader = DirectoryLoader(
            pdf_files_path,
            glob="*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        pdf_documents = pdf_loader.load()
        print(f"  ✓ Loaded {len(pdf_documents)} PDF documents")
    else:
        print(f"  ✗ Directory not found: {pdf_files_path}")
    
    # Combine all documents
    documents = text_documents + pdf_documents
    print(f"\n  ✓ Total documents loaded: {len(documents)}")
    
    if len(documents) == 0:
        print("\n❌ ERROR: No documents were loaded!")
        print("   Please check that data/pdf and data/text_files directories exist and contain files.")
        return False
    
    # Step 3: Chunk documents
    print("\n[Step 3] Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=175,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"  Original documents: {len(documents)}")
    print(f"  Total chunks created: {len(chunks)}")
    print(f"  ✓ Chunking complete")
    
    if len(chunks) == 0:
        print("\n❌ ERROR: No chunks were created!")
        return False
    
    # Step 4: Initialize embeddings
    print("\n[Step 4] Initializing embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    print("  ✓ Embeddings model loaded")
    
    # Step 5: Create vector store
    print("\n[Step 5] Creating vector store (this may take a few minutes)...")
    print("  This will generate embeddings for all chunks and store them in ChromaDB...")
    
    try:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(chroma_db_path)
        )
        print("  ✓ Vector store created successfully!")
        
        # Verify embeddings were stored
        collection = vector_store._collection
        embedding_count = collection.count()
        
        print(f"\n  ✓ Total embeddings stored: {embedding_count}")
        
        if embedding_count == 0:
            print("\n❌ ERROR: Vector store has 0 embeddings!")
            return False
        
        print(f"\n{'='*70}")
        print("✓ SUCCESS! Vector store is ready for queries")
        print(f"{'='*70}")
        print(f"\nYou can now search using:")
        print(f"  python query_search.py 'your query here'")
        print(f"\nOr use the notebook: notebook/document.ipynb")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR creating vector store: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = create_vector_store()
    exit(0 if success else 1)









