"""
Evaluation script for the RAG Slide Assistant.
Calculates Precision@k, Recall@k, Hit Rate@k, and MRR (Mean Reciprocal Rank).

Run with:
    python evaluate.py
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any

from backend.ingestion import load_and_chunk_slides
from backend.embeddings import get_or_build_vectorstore, get_retriever
from backend.config import BASE_DIR

# --- Ground Truth Benchmark Dataset for Course Slides ---
BENCHMARK_DATASET = [
    {
        "id": 1,
        "query": "What is the Open-Closed Principle in SOLID?",
        "expected_source": "CS251_Lecture_Notes_Spring2026.pdf",
        "expected_keywords": ["Open-Closed", "SOLID", "modifying existing code", "flexible"],
        "target_pages": [708, 709, 710],
        "total_relevant_estimate": 3,
    },
    {
        "id": 2,
        "query": "What are the prescribed textbooks for the software engineering course?",
        "expected_source": "CS251_Lecture_Notes_Spring2026.pdf",
        "expected_keywords": ["Sommerville", "Engineering Software Products", "O'Regan", "textbooks"],
        "target_pages": [11],
        "total_relevant_estimate": 1,
    },
    {
        "id": 3,
        "query": "What are the main software process activities?",
        "expected_source": "CS251_Lecture_Notes_Spring2026.pdf",
        "expected_keywords": ["Software Process Activities", "Software Specification", "Development", "Validation"],
        "target_pages": [51, 52, 53, 54, 55],
        "total_relevant_estimate": 3,
    },
    {
        "id": 4,
        "query": "What is the Single Responsibility Principle?",
        "expected_source": "CS251_Lecture_Notes_Spring2026.pdf",
        "expected_keywords": ["Single Responsibility", "SOLID", "reason to change"],
        "target_pages": [700, 701, 702, 703, 704, 705, 706, 707],
        "total_relevant_estimate": 3,
    },
    {
        "id": 5,
        "query": "What are the main challenges of Large Language Models (LLMs)?",
        "expected_source": "LLM & LangChain.pdf",
        "expected_keywords": ["Bias", "Hallucination", "Computational Cost", "Ethical", "Safety"],
        "target_pages": [6],
        "total_relevant_estimate": 1,
    },
    {
        "id": 6,
        "query": "How does data flow through LangChain prompt templates and chains?",
        "expected_source": "LLM & LangChain.pdf",
        "expected_keywords": ["Flow of Data", "Prompt Template", "Chains", "LLM Interface"],
        "target_pages": [11],
        "total_relevant_estimate": 1,
    },
    {
        "id": 7,
        "query": "What is covered in the LangChain agenda?",
        "expected_source": "LLM & LangChain.pdf",
        "expected_keywords": ["Agenda", "LLM foundations", "LangChain architecture", "practical use cases"],
        "target_pages": [2],
        "total_relevant_estimate": 1,
    },
]


def is_chunk_relevant(chunk: Any, item: Dict[str, Any]) -> bool:
    """Check if a retrieved chunk matches the ground truth for a given test item."""
    source_file = chunk.metadata.get("source_file", "")
    if item["expected_source"] and item["expected_source"].lower() not in source_file.lower():
        return False

    page_num = chunk.metadata.get("page", 0)
    target_pages = item.get("target_pages", [])
    if target_pages and page_num in target_pages:
        return True

    # Keyword matching in page content
    text_lower = chunk.page_content.lower()
    matches = sum(1 for kw in item["expected_keywords"] if kw.lower() in text_lower)
    return matches >= 1


def evaluate_retrieval(top_k: int = 5) -> Dict[str, Any]:
    print("=" * 80)
    print(f"[EVALUATION] Running Retrieval Evaluation (Top-K = {top_k})")
    print("=" * 80)

    # Ensure index exists
    chunks = load_and_chunk_slides(strategy="slide_aware")
    vectorstore = get_or_build_vectorstore(chunks)
    retriever = get_retriever(vectorstore, k=top_k)

    results_per_query = []
    total_precision = 0.0
    total_recall = 0.0
    total_hit = 0.0
    total_mrr = 0.0

    print(f"\n{'Q#':<3} | {'Query':<45} | {'P@' + str(top_k):<6} | {'R@' + str(top_k):<6} | {'Hit@' + str(top_k):<6} | {'MRR':<6}")
    print("-" * 80)

    for item in BENCHMARK_DATASET:
        query = item["query"]
        retrieved_docs = retriever.invoke(query)

        relevant_flags = [is_chunk_relevant(doc, item) for doc in retrieved_docs]
        num_relevant_retrieved = sum(relevant_flags)

        # Metrics calculation
        precision_at_k = num_relevant_retrieved / top_k
        
        rel_estimate = item.get("total_relevant_estimate", 1)
        recall_at_k = min(1.0, num_relevant_retrieved / rel_estimate)
        
        hit_at_k = 1.0 if num_relevant_retrieved > 0 else 0.0

        # MRR
        mrr = 0.0
        for rank_idx, rel in enumerate(relevant_flags):
            if rel:
                mrr = 1.0 / (rank_idx + 1)
                break

        total_precision += precision_at_k
        total_recall += recall_at_k
        total_hit += hit_at_k
        total_mrr += mrr

        query_disp = query[:43] + ".." if len(query) > 45 else query
        print(f"{item['id']:<3} | {query_disp:<45} | {precision_at_k:.2f}   | {recall_at_k:.2f}   | {hit_at_k:.2f}   | {mrr:.2f}")

        results_per_query.append(
            {
                "id": item["id"],
                "query": query,
                "precision_at_k": precision_at_k,
                "recall_at_k": recall_at_k,
                "hit_at_k": hit_at_k,
                "mrr": mrr,
                "num_relevant_retrieved": num_relevant_retrieved,
                "retrieved_sources": [
                    f"{doc.metadata.get('source_file')} (Slide {doc.metadata.get('page')})"
                    for doc in retrieved_docs
                ],
            }
        )

    num_queries = len(BENCHMARK_DATASET)
    avg_precision = total_precision / num_queries
    avg_recall = total_recall / num_queries
    avg_hit = total_hit / num_queries
    avg_mrr = total_mrr / num_queries

    print("-" * 80)
    print(f"[SUMMARY] AVERAGE METRICS (k={top_k}):")
    print(f"   * Mean Precision@{top_k} : {avg_precision:.4f} ({avg_precision*100:.1f}%)")
    print(f"   * Mean Recall@{top_k}    : {avg_recall:.4f} ({avg_recall*100:.1f}%)")
    print(f"   * Hit Rate@{top_k}       : {avg_hit:.4f} ({avg_hit*100:.1f}%)")
    print(f"   * MRR               : {avg_mrr:.4f}")
    print("=" * 80)

    summary = {
        "top_k": top_k,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_hit_rate": avg_hit,
        "avg_mrr": avg_mrr,
        "queries_evaluated": num_queries,
        "details": results_per_query,
    }

    # Save metrics JSON
    output_path = BASE_DIR / "data" / "evaluation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[DONE] Evaluation results saved to {output_path}")

    return summary


if __name__ == "__main__":
    k_val = 5
    if len(sys.argv) > 1:
        try:
            k_val = int(sys.argv[1])
        except ValueError:
            pass
    evaluate_retrieval(top_k=k_val)

