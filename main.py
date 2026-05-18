#!/usr/bin/env python3
"""
Main entry point for the RAG project.
Provides a simple CLI menu to choose between different interfaces.
"""
import sys
import os


def show_menu():
    """Display the main menu"""
    print("\n" + "=" * 60)
    print("RAG Project - Main Menu")
    print("=" * 60)
    print("1. Interactive CLI Query (full RAG with LLM)")
    print("2. Streamlit Web App")
    print("3. Simple Semantic Search (no LLM, faster)")
    print("4. Create/Update Vector Store")
    print("5. Exit")
    print("=" * 60)


def main():
    """Main entry point"""
    show_menu()
    
    while True:
        try:
            choice = input("\nSelect an option (1-5): ").strip()
            
            if choice == "1":
                print("\nStarting Interactive CLI Query...")
                from final_rag_fixed import main as rag_main
                rag_main()
                show_menu()
                
            elif choice == "2":
                print("\nStarting Streamlit Web App...")
                print("The app will open in your browser.")
                os.system("streamlit run byob_app.py")
                show_menu()
                
            elif choice == "3":
                query = input("\nEnter your search query: ").strip()
                if query:
                    os.system(f'python query_search.py "{query}"')
                else:
                    print("Query cannot be empty.")
                show_menu()
                
            elif choice == "4":
                print("\nCreating/updating vector store...")
                from create_vector_store import create_vector_store
                create_vector_store()
                show_menu()
                
            elif choice == "5":
                print("\nExiting...")
                sys.exit(0)
                
            else:
                print("Invalid choice. Please select 1-5.")
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)
        except ImportError as e:
            print(f"\nError importing module: {e}")
            print("Please ensure all dependencies are installed:")
            print("  pip install -r requirements.txt")
            sys.exit(1)
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            show_menu()


if __name__ == "__main__":
    main()
