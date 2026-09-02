"""
Step 1 & 2 of the RAG pipeline: Load slide PDFs and split them into chunks.

    Slides (PDF)
        |
    Document Loader
        |
    Slide-Aware Splitter (Page & Header Context preservation)
        |
    List[Document]  (chunks, ready for embedding)
"""

import re
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import SLIDES_DIR

# Default chunking parameters
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


def extract_slide_title(page_text: str) -> str:
    """Extract a concise title/header from the top lines of a slide page."""
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    if not lines:
        return "Untitled Slide"
    
    # Take first 1 or 2 meaningful lines as title summary
    title_candidates = []
    for line in lines[:3]:
        # Skip pure page numbers or slide numbers
        if re.match(r"^\d+$", line) or re.match(r"^slide\s*\d+$", line, re.IGNORECASE):
            continue
        title_candidates.append(line)
        if len(" - ".join(title_candidates)) >= 40:
            break
            
    if not title_candidates:
        return lines[0][:60]
    return " - ".join(title_candidates)[:80]


def load_slide_pdfs(slides_dir: Path = SLIDES_DIR) -> List[Document]:
    """Load every PDF in `slides_dir` into a flat list of page-level Documents."""
    pdf_paths = sorted(Path(slides_dir).glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDFs found in {slides_dir}. Drop your slide PDFs there first."
        )

    all_docs: List[Document] = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page_idx, page in enumerate(pages):
            page.metadata["source_file"] = pdf_path.name
            page.metadata["page"] = page_idx + 1  # 1-indexed page number
        all_docs.extend(pages)

    print(f"[ingestion] Loaded {len(pdf_paths)} PDF(s) -> {len(all_docs)} raw page(s).")
    return all_docs


def chunk_documents_slide_aware(
    documents: List[Document],
    max_slide_len: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Slide-Aware Chunking:
    1. Keeps slide page boundaries intact whenever possible (1 slide = 1 chunk).
    2. Extracts slide titles and prepends contextual header to each chunk:
       [Source: filename | Slide X: Title]
    3. Sub-splits long slides (> max_slide_len) while maintaining the slide header context.
    4. Filters out empty/trivial slides.
    """
    processed_chunks: List[Document] = []
    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_slide_len,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunk_id = 0
    for doc in documents:
        raw_text = doc.page_content.strip()
        # Filter out empty slides or tiny page numbers
        if len(raw_text) < 15:
            continue

        source_file = doc.metadata.get("source_file", "unknown")
        page_num = doc.metadata.get("page", doc.metadata.get("page_number", 1))
        slide_title = extract_slide_title(raw_text)

        header = f"[Source: {source_file} | Slide {page_num}: {slide_title}]"

        # If slide content fits within max length, treat entire slide as one single chunk
        if len(raw_text) <= max_slide_len:
            content = f"{header}\n{raw_text}"
            metadata = {
                **doc.metadata,
                "source_file": source_file,
                "page": page_num,
                "slide_title": slide_title,
                "chunk_id": chunk_id,
            }
            processed_chunks.append(Document(page_content=content, metadata=metadata))
            chunk_id += 1
        else:
            # Sub-split long slide content while injecting slide header into each fragment
            parts = sub_splitter.split_text(raw_text)
            for part_idx, part in enumerate(parts):
                part_header = f"{header} (Part {part_idx + 1}/{len(parts)})"
                content = f"{part_header}\n{part}"
                metadata = {
                    **doc.metadata,
                    "source_file": source_file,
                    "page": page_num,
                    "slide_title": slide_title,
                    "part": part_idx + 1,
                    "total_parts": len(parts),
                    "chunk_id": chunk_id,
                }
                processed_chunks.append(Document(page_content=content, metadata=metadata))
                chunk_id += 1

    print(f"[ingestion] Slide-aware chunking generated {len(processed_chunks)} chunk(s).")
    return processed_chunks


def chunk_documents_standard(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """Standard character-level splitter (fallback strategy)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    print(f"[ingestion] Standard character chunking generated {len(chunks)} chunk(s).")
    return chunks


def chunk_documents(
    documents: List[Document],
    strategy: str = "slide_aware",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """Main chunking entry point supporting 'slide_aware' and 'standard' strategies."""
    if strategy == "slide_aware":
        return chunk_documents_slide_aware(documents, max_slide_len=chunk_size, chunk_overlap=chunk_overlap)
    else:
        return chunk_documents_standard(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def load_and_chunk_slides(strategy: str = "slide_aware") -> List[Document]:
    """Convenience wrapper: load all slide PDFs then chunk them."""
    docs = load_slide_pdfs()
    return chunk_documents(docs, strategy=strategy)


if __name__ == "__main__":
    chunks = load_and_chunk_slides(strategy="slide_aware")
    print("\nSample chunk [0]:\n", chunks[0].page_content[:400])

