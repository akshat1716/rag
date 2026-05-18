#!/usr/bin/env python3
"""
Simple CLI script for semantic search in the vector database.
Performs similarity search without LLM generation (faster for testing retrieval).
"""
import os
import sys
from pathlib import Path

# Use langchain_community for compatibility with existing chromadb setup
# langchain_chroma has compatibility issues with chromadb 0.4.x
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# Configuration
CHROMA_PATH = "./data/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    """Perform semantic search on the vector database"""
    
    if len(sys.argv) < 2:
        print("Usage: python query_search.py '<your query>'")
        print("Example: python query_search.py 'machine learning algorithms'")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    
    print(f"Query: {query}\n")
    
    # Check if database exists
    if not os.path.exists(CHROMA_PATH):
        print(f"Error: Database not found at {CHROMA_PATH}")
        print("Please run: python create_vector_store.py")
        sys.exit(1)
    
    try:
        # Load embeddings and connect to database
        print("Loading embeddings and connecting to database...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        try:
            db = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=embeddings
            )
        except Exception as e:
            print(f"Error loading database: {e}")
            print("The database may be corrupted or incompatible.")
            print("Please try recreating it: python create_vector_store.py")
            sys.exit(1)
        
        # Check if database has documents by trying a simple search
        # This is safer than accessing _collection directly
        try:
            # Try a simple similarity search to verify database is usable
            test_results = db.similarity_search("test", k=1)
            # Get actual count by searching with a very high k
            all_results = db.similarity_search("", k=10000)  # Large k to get all
            collection_count = len(all_results) if all_results else 0
        except Exception as e:
            print(f"Error checking database: {e}")
            print("The database may be corrupted. Try recreating it:")
            print("  python create_vector_store.py")
            sys.exit(1)
        
        if collection_count == 0:
            print(f"Error: Vector database is empty at {CHROMA_PATH}")
            print("Please run: python create_vector_store.py")
            sys.exit(1)
        
        print(f"✓ Connected to database with {collection_count} embeddings\n")
        
        # Perform similarity search
        print("Searching for relevant documents...\n")
        results = db.similarity_search_with_score(query, k=5)
        
        if not results:
            print("No results found.")
            return
        
        # Display results
        print(f"Found {len(results)} relevant documents:\n")
        print("=" * 70)
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n[Result {i}] Similarity Score: {score:.4f}")
            print("-" * 70)
            
            # Extract metadata
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            source_name = os.path.basename(source) if source != 'Unknown' else 'Unknown'
            
            print(f"Source: {source_name}")
            if page != 'N/A':
                print(f"Page: {page}")
            
            # Display content snippet (first 300 characters)
            content = doc.page_content.strip()
            snippet = content[:300] + "..." if len(content) > 300 else content
            print(f"\nContent:\n{snippet}\n")
            
            if i < len(results):
                print("=" * 70)
        
        print("\n" + "=" * 70)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

