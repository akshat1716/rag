#!/usr/bin/env python3
"""
Query script for RAG pipeline - Search the vector store for relevant documents
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def search_vector_store(query: str, k: int = 5):
    """
    Search the vector store for relevant documents
    
    Args:
        query: Search query string
        k: Number of results to return
    """
    # Check if vector store exists
    chroma_path = project_root / "data" / "chroma_db"
    if not chroma_path.exists():
        print("❌ Vector store not found!")
        print(f"   Expected location: {chroma_path}")
        print("\n   Please run the notebook first to create the vector store.")
        print("   The notebook should be at: notebook/document.ipynb")
        return
    
    print("Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    print("Loading vector store from disk...")
    try:
        vector_store = Chroma(
            persist_directory=str(chroma_path),
            embedding_function=embeddings
        )
        
        # Verify it loaded
        collection = vector_store._collection
        embedding_count = collection.count()
        print(f"✓ Loaded vector store with {embedding_count} embeddings\n")
        
        if embedding_count == 0:
            print("⚠️  WARNING: Vector store has 0 embeddings!")
            print("   Please recreate the vector store by running the notebook.")
            return
        
    except Exception as e:
        print(f"❌ Error loading vector store: {e}")
        return
    
    # Perform the search
    print("="*70)
    print("SEMANTIC SEARCH")
    print("="*70)
    print(f"Query: '{query}'")
    print(f"\nSearching for relevant documents...\n")
    
    # Perform similarity search
    results = vector_store.similarity_search_with_score(query, k=k)
    
    print(f"Found {len(results)} relevant results:\n")
    
    if len(results) == 0:
        print("⚠️  WARNING: No results found!")
        print("This may indicate:")
        print("  - The query doesn't match any content in the documents")
        print("  - The vector store was not created correctly")
    else:
        for i, (doc, score) in enumerate(results, 1):
            print(f"{'='*70}")
            print(f"Result {i} (Similarity Score: {score:.4f})")
            print(f"{'='*70}")
            
            # Extract source file name
            source = doc.metadata.get('source', 'N/A')
            source_name = os.path.basename(source) if source != 'N/A' else 'N/A'
            
            # Extract page number
            page = doc.metadata.get('page', 'N/A')
            
            print(f"Source File: {source_name}")
            print(f"Page Number: {page}")
            print(f"\nContent Snippet:")
            print(f"{doc.page_content[:500]}...")
            print()


if __name__ == "__main__":
    # Default query
    query = "List any two challenges in implementing a web crawling system"
    
    # Allow query to be passed as command line argument
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    
    search_vector_store(query, k=5)









