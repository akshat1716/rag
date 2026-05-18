
import os
import time
# Use langchain_community for compatibility with existing chromadb setup
# langchain_chroma has compatibility issues with chromadb 0.4.x
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from typing import Any, Dict

class SimpleRetrievalQA:
    """A simple RetrievalQA implementation using the current LangChain API"""
    
    def __init__(self, llm, retriever, return_source_documents: bool = True):
        self.llm = llm
        self.retriever = retriever
        self.return_source_documents = return_source_documents
    
    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the QA chain"""
        query = inputs.get("query", "").strip()
        
        if not query:
            raise ValueError("Query cannot be empty")
        
        try:
            # Retrieve relevant documents
            relevant_docs = self.retriever.get_relevant_documents(query)
            
            if not relevant_docs:
                return {
                    "result": "No relevant documents found in the database for this query.",
                    "source_documents": []
                }
            
            # Combine the documents into a context
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Create the prompt
            prompt = f"""Use the following context to answer the question. If you cannot find the answer in the context, say so.

Context: {context}

Question: {query}

Answer:"""
            
            # Get response from the LLM
            response = self.llm.invoke(prompt)
            
            # Prepare result
            result = {
                "result": response.content if hasattr(response, 'content') else str(response)
            }
            
            if self.return_source_documents:
                result["source_documents"] = relevant_docs
                
            return result
        except Exception as e:
            raise RuntimeError(f"Error processing query: {str(e)}") from e

# --- CONFIGURATION ---
CHROMA_PATH = "./data/chroma_db"
# This MUST match what you used in document.ipynb
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" 
LLM_MODEL = "llama3.2"

def main():
    print(f"--- 🚀 Starting Local RAG Pipeline (Model: {LLM_MODEL}) ---")
    
    try:
        # 1. SETUP EMBEDDINGS
        # We use the same HuggingFace model so the math matches your database
        print("Loading embedding model...")
        embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        # 2. CONNECT TO LOCAL DATABASE
        print("Connecting to Vector Database...")
        if not os.path.exists(CHROMA_PATH):
            print(f"Error: Database not found at {CHROMA_PATH}")
            print("Please run: python create_vector_store.py")
            return

        try:
            db = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=embedding_function
            )
        except Exception as e:
            print(f"Error loading database: {e}")
            print("The database may be corrupted or incompatible.")
            print("Please try recreating it: python create_vector_store.py")
            return
        
        # Check if database has documents by trying a simple search
        # This is safer than accessing _collection directly
        try:
            # Try a simple similarity search to verify database is usable
            test_results = db.similarity_search("test", k=1)
            # Get actual count by searching with a very high k
            all_results = db.similarity_search("", k=10000)  # Large k to get all
            collection_count = len(all_results) if all_results else 0
            
            if collection_count == 0:
                print(f"Error: Vector database is empty at {CHROMA_PATH}")
                print("Please run: python create_vector_store.py")
                return
            print(f"✓ Connected to database with {collection_count} embeddings")
        except Exception as e:
            print(f"Error accessing database: {e}")
            print("The database may be corrupted. Try recreating it:")
            print("  python create_vector_store.py")
            return

        # 3. SETUP LOCAL LLM (Ollama)
        # This connects to Ollama running locally
        print("Initializing Local LLM...")
        try:
            llm = ChatOllama(model=LLM_MODEL, temperature=0)
            # Test connection with a simple query
            llm.invoke("test")
            print(f"✓ LLM connection successful")
        except Exception as e:
            print(f"Error: Could not connect to Ollama. Is Ollama running?")
            print(f"Please start Ollama and ensure model '{LLM_MODEL}' is installed:")
            print(f"  ollama pull {LLM_MODEL}")
            return

        # 4. CREATE THE CHAIN
        retriever = db.as_retriever(search_kwargs={"k": 3}) # Get top 3 pages
        qa_chain = SimpleRetrievalQA(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        # 5. RUN THE INTERACTIVE LOOP
        print("\n✅ System Ready! Type 'exit' to quit.\n")
        
        while True:
            query = input("\n👉 Your Question: ").strip()
            if query.lower() in ["exit", "quit", "q"]:
                break
            
            if not query:
                print("Please enter a non-empty query.")
                continue
            
            print("\n🤖 Thinking...")
            start_time = time.time()
            
            try:
                # Ask the question
                response = qa_chain.invoke({"query": query})
                
                end_time = time.time()
                
                # Print Answer
                print("\n=== ANSWER ===")
                print(response['result'])
                print(f"\n[⏱️ Time taken: {end_time - start_time:.2f}s]")

                # Print Sources (Proof it read the file)
                if response.get('source_documents'):
                    print("\n📚 Sources Used:")
                    for doc in response['source_documents']:
                        source_name = doc.metadata.get('source', 'Unknown')
                        print(f" - {os.path.basename(source_name)}")
                else:
                    print("\n⚠️ No sources found")
            except Exception as e:
                print(f"\n❌ Error processing query: {e}")
                print("Please try again or type 'exit' to quit.")
    
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
