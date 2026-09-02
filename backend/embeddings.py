"""
Step 3 & 4 of the RAG pipeline: Embed chunks and build/load a FAISS index.

    Chunks
       |
    Embeddings (sentence-transformers, local & free)
       |
    FAISS Index
       |
    Retriever
"""

from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from backend.config import EMBEDDING_MODEL_NAME, VECTORSTORE_DIR, TOP_K

_embeddings_instance: Optional[HuggingFaceEmbeddings] = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Singleton loader so the embedding model is only loaded into memory once."""
    global _embeddings_instance
    if _embeddings_instance is None:
        print(f"[embeddings] Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _embeddings_instance = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embeddings_instance


def build_vectorstore(
    chunks: List[Document], save_path: Path = VECTORSTORE_DIR
) -> FAISS:
    """Embed all chunks and build a fresh FAISS index, saving it to disk."""
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(save_path))
    print(f"[embeddings] Built and saved FAISS index -> {save_path}")
    return vectorstore


def load_vectorstore(save_path: Path = VECTORSTORE_DIR) -> FAISS:
    """Load a previously built FAISS index from disk."""
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        str(save_path), embeddings, allow_dangerous_deserialization=True
    )
    print(f"[embeddings] Loaded FAISS index from {save_path}")
    return vectorstore


def get_or_build_vectorstore(chunks: Optional[List[Document]] = None) -> FAISS:
    """
    Load the FAISS index if it already exists on disk; otherwise build it
    from `chunks` (which must be provided the first time).
    """
    index_file = Path(VECTORSTORE_DIR) / "index.faiss"
    if index_file.exists():
        return load_vectorstore()

    if chunks is None:
        raise ValueError(
            "No existing FAISS index found and no chunks were provided to build one."
        )
    return build_vectorstore(chunks)


def get_retriever(vectorstore: FAISS, k: int = TOP_K):
    """Return a retriever configured for top-k semantic search."""
    return vectorstore.as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    from backend.ingestion import load_and_chunk_slides

    chunks = load_and_chunk_slides()
    vs = build_vectorstore(chunks)
    retriever = get_retriever(vs)
    results = retriever.invoke("What is this course about?")
    for r in results:
        print("---", r.metadata.get("source_file"), "---")
        print(r.page_content[:200], "\n")
