import os
import time
import shutil
import tempfile
import streamlit as st

from typing import Dict, Any, List
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Use langchain_community for compatibility with chromadb 0.4.x
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


# ======================
# CONFIG
# ======================
CHROMA_PATH = "./data/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"


# ======================
# SIMPLE RAG QA
# ======================
class SimpleRetrievalQA:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        query = inputs["query"]

        docs = self.retriever.get_relevant_documents(query)
        context = "\n\n".join(d.page_content for d in docs)

        prompt = f"""
Use the context below to answer the question.
If the answer is not present, say you do not know.

Context:
{context}

Question:
{query}

Answer:
"""

        response = self.llm.invoke(prompt)

        return {
            "result": response.content if hasattr(response, "content") else str(response),
            "source_documents": docs
        }


# ======================
# PDF INGESTION
# ======================
def extract_documents(pdf_path: str) -> List[Document]:
    reader = PdfReader(pdf_path)
    docs = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": os.path.basename(pdf_path),
                        "page": i + 1
                    }
                )
            )
    return docs


def ingest_pdfs(files):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for file in files:
            path = os.path.join(tmpdir, file.name)
            with open(path, "wb") as f:
                f.write(file.read())

            docs = extract_documents(path)
            chunks.extend(splitter.split_documents(docs))

    if not chunks:
        raise RuntimeError("No text extracted from PDFs")

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    db.add_documents(chunks)
    # ChromaDB with persist_directory automatically persists, but we can try to persist explicitly
    # Note: persist() may not exist in all versions, so we catch the exception
    try:
        if hasattr(db, 'persist'):
            db.persist()
    except AttributeError:
        pass  # persist() not available, which is fine if using persist_directory


# ======================
# LOAD QA CHAIN
# ======================
def load_qa_chain():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    try:
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
    except Exception as e:
        raise RuntimeError(f"Error loading database: {e}. The database may be corrupted.")

    # Check if database has documents by trying a simple search
    # This is safer than accessing _collection directly
    try:
        # Try a simple similarity search to verify database is usable
        test_results = db.similarity_search("test", k=1)
        # Get actual count by searching with a very high k
        all_results = db.similarity_search("", k=10000)  # Large k to get all
        collection_count = len(all_results) if all_results else 0
    except Exception as e:
        raise RuntimeError(f"Error accessing database: {e}. The database may be corrupted.")
    
    if collection_count == 0:
        raise RuntimeError("Vector database is empty. Ingest PDFs first.")

    retriever = db.as_retriever(search_kwargs={"k": 3})

    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0
    )

    return SimpleRetrievalQA(llm, retriever)


# ======================
# STREAMLIT UI
# ======================
st.set_page_config(page_title="Local PDF RAG", layout="wide")
st.title("📄 Local PDF RAG (Stable)")
st.caption("Upload PDFs → Embed → Retrieve → Answer (Chroma + Ollama)")

st.divider()

with st.sidebar:
    st.header("Database")
    if st.button("Clear Vector Database"):
        shutil.rmtree(CHROMA_PATH, ignore_errors=True)
        st.success("Vector DB cleared")

# Upload
st.header("1️⃣ Upload & Ingest PDFs")

files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Ingest PDFs"):
    if not files:
        st.error("Upload at least one PDF.")
    else:
        with st.spinner("Ingesting PDFs..."):
            ingest_pdfs(files)
            st.success("PDFs ingested successfully.")

st.divider()

# QA
st.header("2️⃣ Ask a Question")

query = st.text_input("Enter your question")

if query:
    try:
        with st.spinner("Thinking..."):
            qa = load_qa_chain()
            start = time.time()
            response = qa.invoke({"query": query})
            end = time.time()

        st.subheader("Answer")
        st.write(response["result"])
        st.caption(f"⏱️ {end - start:.2f}s")

        with st.expander("Sources"):
            if response.get("source_documents"):
                for d in response["source_documents"]:
                    st.write(
                        f"- {d.metadata.get('source')} (page {d.metadata.get('page')})"
                    )
            else:
                st.write("No sources found")
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        st.info("Make sure Ollama is running and the vector database is not empty.")
