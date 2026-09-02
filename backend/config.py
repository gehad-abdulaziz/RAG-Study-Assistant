"""
Central configuration for the Adaptive RAG Study Assistant.
All paths and model names are defined here so every other module
imports from a single source of truth.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
SLIDES_DIR = BASE_DIR / "data" / "slides"
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore"
MEMORY_STORE_DIR = BASE_DIR / "data" / "memory_sessions"

SLIDES_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_STORE_DIR.mkdir(parents=True, exist_ok=True)

# --- Models --------------------------------------------------------------
# Free, local, CPU-friendly defaults. Swap via .env without touching code.
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

# Whether to call the hosted HF Inference API instead of loading the model
# locally. Local is the default so the project runs with zero API keys.
USE_HF_INFERENCE_API = bool(HUGGINGFACEHUB_API_TOKEN)

# --- Chunking --------------------------------------------------------------
CHUNK_SIZES = 2000
OVERLAPS = 200

# --- Retrieval -------------------------------------------------------------
TOP_K = 5

# --- Memory ------------------------------------------------------------
BUFFER_WINDOW_K = 20  # how many past exchanges "Buffer Window" memory keeps
