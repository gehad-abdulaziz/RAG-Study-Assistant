"""
Chapter 3 + Chapter 5 content: Prompt Templates, Zero-Shot vs Few-Shot.

Each function returns a LangChain PromptTemplate ready to be dropped into
a chain. Keeping them here (separate from chains.py) means every chain can
reuse the exact same prompting strategy without duplicating template text.
"""

from langchain_core.prompts import PromptTemplate

# A small, reusable set of Few-Shot examples. In a real course you'd tailor
# these to the subject; keep them short so they don't eat the context window.
FEW_SHOT_EXAMPLES = """Example 1:
Question: What is an RNN?
Answer: A Recurrent Neural Network (RNN) is a neural network architecture
that processes sequential data by maintaining a hidden state that carries
information from previous time steps.

Example 2:
Question: What is LSTM?
Answer: An LSTM (Long Short-Term Memory) is a type of RNN designed to
better capture long-range dependencies using gates that control what
information is kept, forgotten, or output.
"""


def zero_shot_prompt() -> PromptTemplate:
    """Direct instruction, no examples given to the model."""
    template = """You are a helpful study assistant. Use ONLY the context
below (from the course slides) to answer the student's question. If the
answer is not in the context, say you don't have that information in the
slides.

Conversation so far:
{history}

Context from slides:
{context}

Question:
{question}

Answer:"""
    return PromptTemplate(
        input_variables=["history", "context", "question"], template=template
    )


def few_shot_prompt() -> PromptTemplate:
    """Same task, but primed with worked examples before the real question."""
    template = f"""You are a helpful study assistant. Use ONLY the context
below (from the course slides) to answer the student's question, following
the style of the examples.

{FEW_SHOT_EXAMPLES}

Conversation so far:
{{history}}

Context from slides:
{{context}}

Now answer:
Question: {{question}}
Answer:"""
    return PromptTemplate(
        input_variables=["history", "context", "question"], template=template
    )


def get_prompt(strategy: str) -> PromptTemplate:
    """strategy: 'zero-shot' | 'few-shot'"""
    strategy = strategy.lower().strip()
    if strategy == "zero-shot":
        return zero_shot_prompt()
    if strategy == "few-shot":
        return few_shot_prompt()
    raise ValueError(f"Unknown prompting strategy: {strategy}")


# --- Task-specific prompts (used by the chain layer) ------------------

MAP_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="Summarize this slide excerpt in 2-3 sentences:\n\n{text}\n\nSummary:",
)

REDUCE_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=(
        "Combine the following partial summaries into one coherent, "
        "well-structured summary of the lecture:\n\n{text}\n\nFinal Summary:"
    ),
)

REFINE_INITIAL_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="Answer the question as best you can using this context:\n\n{text}\n\nAnswer:",
)

REFINE_PROMPT = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=(
        "Here is an existing answer:\n{existing_answer}\n\n"
        "Refine it (improve or extend it, don't just repeat it) using this "
        "additional context if useful:\n{text}\n\nRefined answer:"
    ),
)
