"""
Run this ONCE after dropping your slide PDFs into data/slides/:

    python setup_index.py

It loads the PDFs, chunks them, embeds them, and saves a FAISS index to
data/vectorstore/. The Streamlit app and pipeline then just load this index
instead of rebuilding it on every run.
"""

from backend.ingestion import load_and_chunk_slides
from backend.embeddings import build_vectorstore

if __name__ == "__main__":
    print("Loading and chunking slides...")
    chunks = load_and_chunk_slides()

    print("Building FAISS index (this downloads the embedding model on first run)...")
    build_vectorstore(chunks)

    print("\n[DONE] Done. You can now run: streamlit run frontend/app.py")
