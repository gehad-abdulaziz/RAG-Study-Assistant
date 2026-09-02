"""
Streamlit frontend for the Adaptive RAG Study Assistant.

Run with:
    streamlit run frontend/app.py

The user configures Prompting / Memory / Chain, then asks questions that
are answered strictly from the pre-loaded course slides (RAG, not fine-tuning).
"""

import sys
from pathlib import Path

# Allow running `streamlit run frontend/app.py` from the project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st

from backend.pipeline import StudyAssistantPipeline, PipelineConfig
from backend.config import SLIDES_DIR

st.set_page_config(page_title="Adaptive RAG Study Assistant", page_icon="📚", layout="wide")

st.title("📚 Adaptive RAG Study Assistant")
st.caption(
    "Answers are grounded in your course slides (Retrieval-Augmented Generation), "
    "not a fine-tuned model."
)

# --- Sidebar: pipeline configuration -----------------------------------
with st.sidebar:
    st.header("⚙️ Study Mode Configuration")

    prompting_strategy = st.selectbox(
        "🧠 Prompting Strategy", ["zero-shot", "few-shot"], index=0
    )
    memory_strategy = st.selectbox(
        "💾 Memory Type", ["buffer", "buffer-window", "summary", "entity"], index=0
    )
    chain_type = st.selectbox(
        "⛓️ Chain Type",
        ["simple-sequential", "sequential", "mapreduce", "refine"],
        index=0,
        help=(
            "MapReduce/Refine work best for broad questions like "
            "'summarize the lecture'. Use Simple Sequential/Sequential "
            "for direct Q&A."
        ),
    )

    st.divider()
    pdf_count = len(list(Path(SLIDES_DIR).glob("*.pdf")))
    st.caption(f"📄 {pdf_count} slide PDF(s) detected in `data/slides/`")

    if st.button("🔄 Reset conversation"):
        st.session_state.pop("pipeline", None)
        st.session_state.pop("chat_history", None)
        st.rerun()

# --- Build / rebuild pipeline when config changes ------------------------
config_key = (prompting_strategy, memory_strategy, chain_type)
if "pipeline_config_key" not in st.session_state or st.session_state.pipeline_config_key != config_key:
    config = PipelineConfig(
        prompting_strategy=prompting_strategy,
        memory_strategy=memory_strategy,
        chain_type=chain_type,
    )
    with st.spinner("Setting up pipeline (loading models / index)..."):
        st.session_state.pipeline = StudyAssistantPipeline(config, session_id="streamlit_session")
    st.session_state.pipeline_config_key = config_key
    st.session_state.setdefault("chat_history", [])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Chat history ---------------------------------------------------------
for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        with st.expander("🔍 Pipeline Inspector"):
            st.code(" → ".join(turn["pipeline_steps"]))
            st.caption("Retrieved slide excerpts:")
            for chunk in turn["chunks"]:
                st.text(f"[{chunk.metadata.get('source_file', '?')}] {chunk.page_content[:150]}...")

# --- New question -----------------------------------------------------
question = st.chat_input("Ask a question about the course slides...")
if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.pipeline.ask(question)
        st.markdown(result.answer)
        with st.expander("🔍 Pipeline Inspector"):
            st.code(" → ".join(result.pipeline_steps))
            st.caption("Retrieved slide excerpts:")
            for chunk in result.retrieved_chunks:
                st.text(f"[{chunk.metadata.get('source_file', '?')}] {chunk.page_content[:150]}...")

    st.session_state.chat_history.append(
        {
            "question": question,
            "answer": result.answer,
            "pipeline_steps": result.pipeline_steps,
            "chunks": result.retrieved_chunks,
        }
    )
