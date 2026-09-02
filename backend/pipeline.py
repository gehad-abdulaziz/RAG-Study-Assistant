"""
The orchestrator: wires together Retriever + Memory + Prompt + Chain.

                 User Question
                       |
          -------------+-------------
          |                         |
      Retriever                  Memory
          |                         |
     Relevant Chunks          Conversation
          |                         |
          -------------+-------------
                       |
                    Prompt
                       |
                     Chain
                       |
                    Answer

This is the "Pipeline Inspector" data structure too: every call returns not
just the answer, but which pipeline was actually used, so the frontend can
show it to the user for transparency.
"""

from dataclasses import dataclass, field
from typing import List

from langchain_core.documents import Document

from backend.embeddings import get_or_build_vectorstore, get_retriever
from backend.memory import get_memory, BaseMemory
from backend.prompts import get_prompt
from backend.chains import run_chain
from backend.config import TOP_K


@dataclass
class PipelineConfig:
    prompting_strategy: str = "zero-shot"   # 'zero-shot' | 'few-shot'
    memory_strategy: str = "buffer"          # 'buffer' | 'buffer-window' | 'summary' | 'entity'
    chain_type: str = "simple-sequential"    # 'simple-sequential' | 'sequential' | 'mapreduce' | 'refine'
    top_k: int = TOP_K



@dataclass
class PipelineResult:
    answer: str
    retrieved_chunks: List[Document]
    pipeline_steps: List[str] = field(default_factory=list)


class StudyAssistantPipeline:
    """
    Stateful per-session object: holds the memory instance so a Streamlit
    session (or notebook cell) can keep asking follow-up questions.
    """

    def __init__(self, config: PipelineConfig, session_id: str = "default"):
        self.config = config
        self.session_id = session_id
        self.memory: BaseMemory = get_memory(config.memory_strategy)
        self.memory.load_from_disk(session_id)

        self.vectorstore = get_or_build_vectorstore()
        self.retriever = get_retriever(self.vectorstore, k=config.top_k)

    def ask(self, question: str) -> PipelineResult:
        steps = ["User Question"]

        # 1. Retrieval
        retrieved_chunks = self.retriever.invoke(question)
        steps.append(f"Retriever (top {self.config.top_k} chunks from slides)")

        context = "\n\n".join(c.page_content for c in retrieved_chunks)

        # 2. Memory
        history = self.memory.get_history_text()
        steps.append(f"{self.memory.name.replace('_', ' ').title()} Memory")

        # 3. Prompt
        prompt = get_prompt(self.config.prompting_strategy)
        steps.append(f"{self.config.prompting_strategy.title()} Prompt")

        # 4. Chain
        answer = run_chain(
            chain_type=self.config.chain_type,
            prompt=prompt,
            history=history,
            context=context,
            question=question,
            chunks=retrieved_chunks,
        )
        steps.append(f"{self.config.chain_type.replace('-', ' ').title()} Chain")
        steps.append("LLM")
        steps.append("Answer")

        # 5. Update memory with this turn
        self.memory.save_turn(question, answer)
        self.memory.save_to_disk(self.session_id)

        return PipelineResult(
            answer=answer, retrieved_chunks=retrieved_chunks, pipeline_steps=steps
        )


if __name__ == "__main__":
    config = PipelineConfig(
        prompting_strategy="few-shot",
        memory_strategy="summary",
        chain_type="refine",
    )
    pipeline = StudyAssistantPipeline(config, session_id="demo")

    result = pipeline.ask("What is self-attention?")
    print("PIPELINE:", " -> ".join(result.pipeline_steps))
    print("\nANSWER:\n", result.answer)
