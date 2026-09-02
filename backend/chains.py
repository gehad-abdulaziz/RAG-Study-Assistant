"""
Chapter 6/8 content: the 4 Chain strategies.

    Simple Sequential : Question -> Answer -> Short Summary
    Sequential         : Extract concepts -> Explain -> Quiz -> Format
    MapReduce          : chunk -> summary (map) -> combine (reduce)
    Refine             : answer1 -> refine with chunk2 -> refine with chunk3...


Every chain here takes the SAME core inputs (a built prompt from prompts.py,
retrieved context, question, history) so the pipeline layer can swap chains
in/out based on the user's UI selection without special-casing each one.
"""

from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from backend.llm import get_llm
from backend.prompts import MAP_PROMPT, REDUCE_PROMPT, REFINE_INITIAL_PROMPT, REFINE_PROMPT


def run_simple_answer(prompt: PromptTemplate, history: str, context: str, question: str) -> str:
    """The baseline single LLM call: used directly, or as a building block."""
    llm = get_llm()
    filled = prompt.format(history=history, context=context, question=question)
    return llm.invoke(filled).strip()


def run_simple_sequential_chain(
    prompt: PromptTemplate, history: str, context: str, question: str
) -> str:
    """
    Simple Sequential Chain: Question -> Explanation -> Short Summary.
    Output of step 1 feeds directly into step 2, single input/output each.
    """
    llm = get_llm()
    explanation = run_simple_answer(prompt, history, context, question)

    summarize_prompt = f"Summarize this explanation in 1-2 short sentences:\n\n{explanation}\n\nSummary:"
    summary = llm.invoke(summarize_prompt).strip()

    return f"{explanation}\n\n**TL;DR:** {summary}"


def run_sequential_chain(
    prompt: PromptTemplate, history: str, context: str, question: str
) -> str:
    """
    Sequential Chain: multiple named steps, each can use outputs from
    earlier steps. Here: extract key concepts -> explain -> mini quiz.
    """
    llm = get_llm()

    extract_prompt = (
        f"From this context, list the 2-3 key concepts relevant to the "
        f"question '{question}':\n\n{context}\n\nKey concepts:"
    )
    key_concepts = llm.invoke(extract_prompt).strip()

    explanation = run_simple_answer(prompt, history, context, question)

    quiz_prompt = (
        f"Based on these key concepts: {key_concepts}\n\n"
        f"Write ONE short quiz question (with the answer) to test understanding."
    )
    quiz = llm.invoke(quiz_prompt).strip()

    return (
        f"**Key concepts:** {key_concepts}\n\n"
        f"**Explanation:** {explanation}\n\n"
        f"**Quick check:** {quiz}"
    )


def run_mapreduce_chain(chunks: List[Document], question: str) -> str:
    """
    MapReduce: summarize each chunk independently (map), then combine all
    partial summaries into one final answer (reduce). Best for broad
    requests like 'summarize the whole lecture'.
    """
    llm = get_llm()

    partial_summaries = []
    for chunk in chunks:
        filled = MAP_PROMPT.format(text=chunk.page_content)
        partial_summaries.append(llm.invoke(filled).strip())

    combined_text = "\n\n".join(partial_summaries)
    reduce_filled = REDUCE_PROMPT.format(text=combined_text)
    final_summary = llm.invoke(reduce_filled).strip()

    return final_summary


def run_refine_chain(chunks: List[Document], question: str) -> str:
    """
    Refine: build an answer from the first chunk, then progressively refine
    it as each additional chunk is folded in. Best for questions that need
    a comprehensive answer built across many slides.
    """
    llm = get_llm()

    if not chunks:
        return "No relevant slide content found."

    first_context = f"Question: {question}\n\nContext:\n{chunks[0].page_content}"
    filled = REFINE_INITIAL_PROMPT.format(text=first_context)
    answer = llm.invoke(filled).strip()

    for chunk in chunks[1:]:
        filled = REFINE_PROMPT.format(existing_answer=answer, text=chunk.page_content)
        answer = llm.invoke(filled).strip()

    return answer


def run_chain(
    chain_type: str,
    prompt: PromptTemplate,
    history: str,
    context: str,
    question: str,
    chunks: List[Document],
) -> str:
    """
    Dispatcher used by the pipeline layer.
    chain_type: 'simple-sequential' | 'sequential' | 'mapreduce' | 'refine'
    """
    chain_type = chain_type.lower().strip()

    if chain_type == "simple-sequential":
        return run_simple_sequential_chain(prompt, history, context, question)
    if chain_type == "sequential":
        return run_sequential_chain(prompt, history, context, question)
    if chain_type == "mapreduce":
        return run_mapreduce_chain(chunks, question)
    if chain_type == "refine":
        return run_refine_chain(chunks, question)

    raise ValueError(f"Unknown chain type: {chain_type}")
