"""
Loads the free Hugging Face LLM used across the whole pipeline.

By default this loads a small instruction-tuned model (flan-t5-base) locally
on CPU so the project runs with zero API keys. If HUGGINGFACEHUB_API_TOKEN is
set in .env, it switches to the hosted HF Inference API instead (lighter on
your machine, same interface).
"""

from typing import Optional

from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint
from langchain_core.language_models import BaseLanguageModel

from backend.config import LLM_MODEL_NAME, USE_HF_INFERENCE_API, HUGGINGFACEHUB_API_TOKEN

_llm_instance: Optional[BaseLanguageModel] = None


def get_llm() -> BaseLanguageModel:
    """Singleton loader so the (potentially large) model loads only once."""
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    if USE_HF_INFERENCE_API:
        print(f"[llm] Using HF Inference API model: {LLM_MODEL_NAME}")
        _llm_instance = HuggingFaceEndpoint(
            repo_id=LLM_MODEL_NAME,
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
            model=LLM_MODEL_NAME,
            max_new_tokens=512,
            temperature=0.3,
        )
    else:
        print(f"[llm] Loading local HF pipeline model: {LLM_MODEL_NAME}")
        is_seq2seq = any(k in LLM_MODEL_NAME.lower() for k in ["t5", "bart", "pegasus"])
        task = "text2text-generation" if is_seq2seq else "text-generation"
        _llm_instance = HuggingFacePipeline.from_model_id(
            model_id=LLM_MODEL_NAME,
            task=task,
            pipeline_kwargs={"max_new_tokens": 512, "temperature": 0.3, "do_sample": True},
        )

    return _llm_instance


if __name__ == "__main__":
    llm = get_llm()
    print(llm.invoke("Explain what a neural network is in one sentence."))
