# RAG (Retrieval-Augmented Generation) Project

A local RAG system that allows you to query your documents using semantic search and LLM-based question answering. This project uses ChromaDB for vector storage, HuggingFace embeddings, and Ollama for local LLM inference.

## Features

- 📄 **Document Ingestion**: Load PDF and text files into a vector database
- 🔍 **Semantic Search**: Fast similarity search across your documents
- 🤖 **Question Answering**: Interactive Q&A using local LLM (Ollama)
- 🖥️ **Multiple Interfaces**: CLI and Streamlit web app
- 💾 **Persistent Storage**: Vector database is saved locally for reuse

## Prerequisites

1. **Python 3.13+** (or Python 3.10+)
2. **Ollama** installed and running
   - Download from: https://ollama.ai
   - Install the model: `ollama pull llama3.2`

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd rag2
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or if using `uv`:
   ```bash
   uv pip install -r requirements.txt
   ```

4. **Verify Ollama is running:**
   ```bash
   ollama list
   ```
   
   If `llama3.2` is not listed, install it:
   ```bash
   ollama pull llama3.2
   ```

## Quick Start

### Step 1: Create the Vector Store

First, ingest your documents into the vector database:

```bash
python create_vector_store.py
```

This script will:
- Load all PDF files from `data/pdf/`
- Load all text files from `data/text_files/`
- Chunk and embed the documents
- Create a persistent ChromaDB at `data/chroma_db/`

**Note:** The vector store only needs to be created once, unless you add new documents.

### Step 2: Query Your Documents

You have three options to query your documents:

#### Option 1: Interactive CLI (Recommended for Q&A)

Full RAG pipeline with LLM-based answers:

```bash
python final_rag_fixed.py
```

This launches an interactive session where you can ask questions and get answers with source citations.

#### Option 2: Streamlit Web App

User-friendly web interface:

```bash
streamlit run byob_app.py
```

The app will open in your browser with:
- PDF upload and ingestion
- Query interface
- Source document display
- Database management

#### Option 3: Simple Semantic Search (Fast, No LLM)

Quick similarity search without LLM generation:

```bash
python query_search.py "your query here"
```

Example:
```bash
python query_search.py "machine learning algorithms"
```

#### Option 4: Main Menu

Launch the main menu to choose between all options:

```bash
python main.py
```

## Project Structure

```
rag/
├── data/
│   ├── chroma_db/          # Vector database (created after ingestion)
│   ├── pdf/                # PDF documents to ingest
│   └── text_files/         # Text documents to ingest
├── notebook/
│   └── document.ipynb      # Jupyter notebook for exploration
├── byob_app.py             # Streamlit web application
├── create_vector_store.py  # Script to build vector database
├── final_rag_fixed.py      # Interactive CLI RAG interface
├── query_search.py         # Simple semantic search (no LLM)
├── main.py                 # Main entry point with menu
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project configuration
└── README.md               # This file
```

## Configuration

Default settings (can be modified in the source files):

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM Model**: `llama3.2` (Ollama)
- **Vector Store Path**: `./data/chroma_db`
- **Chunk Size**: 900 tokens
- **Chunk Overlap**: 175 tokens
- **Retrieval Count**: 3 documents per query

## Usage Examples

### Adding New Documents

1. Place PDF files in `data/pdf/` or text files in `data/text_files/`
2. Run `python create_vector_store.py` to rebuild the vector database

### Querying Examples

**Interactive CLI:**
```bash
$ python final_rag_fixed.py
👉 Your Question: What is information retrieval?
🤖 Thinking...
=== ANSWER ===
Information retrieval is the process of...
📚 Sources Used:
 - document1.pdf
 - document2.pdf
```

**Command-line search:**
```bash
$ python query_search.py "information retrieval systems"
Query: information retrieval systems
Found 5 relevant documents...
```

## Troubleshooting

### "Database not found" Error

Run `python create_vector_store.py` to create the vector database first.

### "Could not connect to Ollama" Error

1. Ensure Ollama is running: `ollama list`
2. Install the model: `ollama pull llama3.2`
3. Verify Ollama service is running on your system

### "Vector database is empty" Error

The database exists but has no documents. Re-run `create_vector_store.py`.

### Import Errors

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Performance Issues

- First run will be slower (downloading embedding model)
- LLM inference speed depends on your hardware
- Consider using a smaller/faster model if needed

## Technology Stack

- **LangChain**: Framework for building RAG applications
- **ChromaDB**: Vector database for storing embeddings
- **HuggingFace**: Embedding models
- **Ollama**: Local LLM inference
- **Streamlit**: Web interface
- **PyPDF**: PDF parsing

## Notes

- The vector database persists between sessions
- You only need to rebuild the database when adding new documents
- All processing happens locally (no external API calls)
- The embedding model is downloaded on first use (~90MB)

## LicenseThis project is provided as-is for educational and development purposes.
