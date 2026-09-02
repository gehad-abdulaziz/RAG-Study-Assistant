# 📚 Adaptive RAG Study Assistant

> An interactive Retrieval-Augmented Generation (RAG) study assistant grounded in a fixed collection of course slides.

The **Adaptive RAG Study Assistant** is a capstone project that integrates Prompt Engineering, Memory, Chains, Embeddings, Semantic Retrieval, and Large Language Models into a single practical application.

The system allows users to ask questions about a predefined collection of course slides and experiment with different prompting, memory, and chain configurations to observe their impact on the generated answers.

> **Important:** This project does **not** fine-tune an LLM on the course slides. The slides are used as a retrieval knowledge base and are provided to the LLM as context at inference time.
 
---

## 🎯 Project Purpose

This project is the **capstone project for the Prompt Engineering course by Abu Bakr**.

Its main goal is to apply and integrate the concepts learned throughout the course into a practical, end-to-end **Retrieval-Augmented Generation (RAG) system**.

Rather than implementing each concept separately, the project combines them into one interactive study assistant, demonstrating how different Prompt Engineering and LLM application techniques can work together in a complete system.

The project applies concepts including:

* Prompt Templates
* Zero-shot Prompting
* Few-shot Prompting
* Conversation Memory
* Sequential Chains
* MapReduce Chains
* Refine Chains
* Embeddings
* Semantic Retrieval
* FAISS Vector Search
* Retrieval-Augmented Generation
* LLM-based Generation
* Retrieval Evaluation

> **From learning individual concepts to building a complete LLM application.**

---

# ✨ Features

* 📄 **PDF Course Material Ingestion**

  * Load predefined course slides from PDF files.

* ✂️ **Document Chunking**

  * Split course material into manageable chunks for retrieval.

* 🧠 **Local Semantic Embeddings**

  * Generate embeddings using Sentence Transformers.

* 🔎 **FAISS Vector Search**

  * Efficiently retrieve the most relevant course chunks.

* 🎯 **Zero-shot & Few-shot Prompting**

  * Compare different prompting strategies.

* 💾 **Multiple Memory Strategies**

  * Buffer Memory
  * Window Memory
  * Summary Memory
  * Entity Memory

* 🔗 **Multiple Chain Architectures**

  * Simple Sequential
  * Sequential
  * MapReduce
  * Refine

* 💬 **Multi-turn Conversations**

  * Maintain conversational context across questions.

* 🔬 **Pipeline Inspector**

  * Inspect the configuration used to generate each answer.

* 📊 **Retrieval Evaluation**

  * Precision@K
  * Recall@K
  * Hit Rate@K
  * Mean Reciprocal Rank (MRR)

* 🛡️ **Grounded Responses**

  * Answers are generated using retrieved course material rather than relying solely on the model's pretrained knowledge.

---

# 🏗️ Architecture

```text
                     COURSE SLIDES
                          │
                          ▼
                 ┌─────────────────┐
                 │ PDF Loader      │
                 │ + Chunking      │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Sentence        │
                 │ Transformers    │
                 │ Embeddings      │
                 └────────┬────────┘
                          │
                          ▼
                     ┌────────┐
                     │ FAISS  │
                     └────┬───┘
                          │
                          │
                    User Question
                          │
                          ▼
                     Retriever
                          │
                          ▼
                  Relevant Chunks
                          │
             ┌────────────┼────────────┐
             │            │            │
             ▼            ▼            ▼
          Prompt        Memory       Chain
             │            │            │
             └────────────┼────────────┘
                          │
                          ▼
                         LLM
                          │
                          ▼
                       Answer
```

---

# 🔄 How the RAG Pipeline Works

The system processes each user question through the following stages:

### 1. User Question

The user asks a question related to the course material.

### 2. Embedding & Retrieval

The question is converted into an embedding and searched against the FAISS vector index.

The most relevant chunks from the course slides are retrieved.

### 3. Conversation Memory

For multi-turn conversations, the selected memory strategy provides relevant conversation history.

Available strategies include:

* Buffer
* Window
* Summary
* Entity

### 4. Prompt Construction

The system constructs a prompt using:

* User question
* Retrieved course context
* Conversation history
* Selected prompting strategy

The available prompting strategies are:

* Zero-shot
* Few-shot

### 5. Chain Execution

The selected chain determines how the information is processed before final generation.

Available chains include:

* Simple Sequential
* Sequential
* MapReduce
* Refine

### 6. LLM Generation

The final prompt is passed to the configured Hugging Face language model.

### 7. Pipeline Inspector

The application exposes the selected pipeline configuration so users can understand how the answer was produced.

---

# 🧩 Pipeline Configuration

The project allows different RAG configurations to be selected and compared.

| Component    | Available Options                                   |
| ------------ | --------------------------------------------------- |
| Prompt       | Zero-shot / Few-shot                                |
| Memory       | Buffer / Window / Summary / Entity                  |
| Chain        | Simple Sequential / Sequential / MapReduce / Refine |
| Embeddings   | Sentence Transformers                               |
| Vector Store | FAISS                                               |
| Generation   | Hugging Face LLM                                    |

This makes the project more than a simple chatbot.

It provides an **interactive environment for experimenting with different LLM application techniques** and observing their effect on the final response.

---

# 🧠 Course Concepts → Implementation

| Concept                          | Implementation                            |
| -------------------------------- | ----------------------------------------- |
| Base vs Instruction-Tuned Models | `backend/llm.py`                          |
| Tokenization / Context Window    | `backend/config.py`                       |
| API Keys / Environment Variables | `.env.example`                            |
| Prompt Templates                 | `backend/prompts.py`                      |
| Zero-shot Prompting              | `backend/prompts.py`                      |
| Few-shot Prompting               | `backend/prompts.py`                      |
| Chatbots / Multi-turn            | `frontend/app.py`                         |
| Output Structuring               | `PipelineResult` in `backend/pipeline.py` |
| Buffer Memory                    | `backend/memory.py`                       |
| Window Memory                    | `backend/memory.py`                       |
| Summary Memory                   | `backend/memory.py`                       |
| Entity Memory                    | `backend/memory.py`                       |
| Sequential Chains                | `backend/chains.py`                       |
| MapReduce                        | `backend/chains.py`                       |
| Refine                           | `backend/chains.py`                       |
| Document Loading                 | `backend/ingestion.py`                    |
| Chunking                         | `backend/ingestion.py`                    |
| Embeddings                       | `backend/embeddings.py`                   |
| Semantic Search                  | `backend/embeddings.py`                   |
| Vector Indexing                  | FAISS                                     |
| Retrieval                        | `backend/embeddings.py`                   |
| Full RAG Pipeline                | `backend/pipeline.py`                     |

---

# 🛠️ Tech Stack

### Core

* Python
* LangChain
* Hugging Face Transformers
* Sentence Transformers
* FAISS
* Streamlit
* PyPDF
* python-dotenv

### Embedding Model

```text
sentence-transformers/all-MiniLM-L6-v2
```

A lightweight embedding model that can run locally on CPU.

### Default Generation Model

```text
google/flan-t5-base
```

The default generation model is an instruction-tuned model that can run locally without requiring a paid API.

Larger models or hosted Hugging Face models can be configured if more computational resources are available.

---

# 📊 Current Evaluation Results

The current retrieval evaluation was conducted on **7 manually annotated queries** over the course-slide knowledge base.

## Overall Retrieval Performance

| Metric               |      Score | Interpretation                                                                |
| -------------------- | ---------: | ----------------------------------------------------------------------------- |
| **Mean Recall@5**    | **1.0000** | 100% of the annotated relevant chunks were retrieved within the top 5 results |
| **Hit Rate@5**       | **1.0000** | Every evaluated query retrieved at least one relevant chunk in the top 5      |
| **MRR**              | **0.9286** | The first relevant result was typically ranked first                          |
| **Mean Precision@5** | **0.6571** | Approximately 65.7% of the retrieved top-5 chunks were relevant on average    |

## 🔎 Per-Query Retrieval Metrics

|  # | Query                                          |  P@5 |  R@5 | Hit@5 |  MRR |
| -: | ---------------------------------------------- | ---: | ---: | ----: | ---: |
|  1 | What is the Open-Closed Principle in SOLID?    | 1.00 | 1.00 |  1.00 | 1.00 |
|  2 | What are the prescribed textbooks for course?  | 0.40 | 1.00 |  1.00 | 1.00 |
|  3 | What are the main software process activities? | 0.80 | 1.00 |  1.00 | 1.00 |
|  4 | What is the Single Responsibility Principle?   | 1.00 | 1.00 |  1.00 | 1.00 |
|  5 | What are the main challenges of LLMs?          | 0.40 | 1.00 |  1.00 | 0.50 |
|  6 | How does data flow through LangChain prompts?  | 0.80 | 1.00 |  1.00 | 1.00 |
|  7 | What is covered in the LangChain agenda?       | 0.20 | 1.00 |  1.00 | 1.00 |

## 📌 Interpretation

The current results indicate that the retrieval component has **strong coverage of the relevant course material**:

* **Recall@5 = 100%** — all annotated relevant chunks were retrieved within the top 5.
* **Hit Rate@5 = 100%** — every evaluated query returned at least one relevant result.
* **MRR = 92.86%** — the first relevant result was generally ranked very highly.
* **Precision@5 = 65.71%** — the main opportunity for improvement is reducing irrelevant chunks among the top-5 results.

Overall, the current retrieval configuration is **recall-oriented**: it successfully retrieves the required evidence, while also returning some additional non-relevant chunks.

> These metrics evaluate **retrieval quality**, not the correctness of the final LLM-generated answer.

---

# 🧪 Evaluation Methodology

The evaluation separates the RAG system into two major components:

```text
                 RAG Evaluation
                       │
            ┌──────────┴──────────┐
            │                     │
            ▼                     ▼
     Retrieval Quality      Generation Quality
            │                     │
            ▼                     ▼
     Precision@K             Correctness
     Recall@K                Faithfulness
     Hit Rate@K              Groundedness
     MRR                     Citation Accuracy
```

### Retrieval Evaluation

Measures whether the retriever finds the relevant course chunks.

Current metrics:

* Precision@5
* Recall@5
* Hit Rate@5
* MRR

### Generation Evaluation

Generation-level evaluation is planned to measure whether the LLM correctly uses the retrieved evidence.

Potential metrics include:

* Answer Correctness
* Faithfulness / Groundedness
* Context Relevance
* Citation Accuracy

A high retrieval score does **not** automatically mean that the final generated answer is correct. The retrieved context must still be correctly interpreted and used by the LLM.

---

# 🗂️ Project Structure

```text
study_assistant/
│
├── backend/
│   ├── config.py
│   ├── ingestion.py
│   ├── embeddings.py
│   ├── llm.py
│   ├── prompts.py
│   ├── memory.py
│   ├── chains.py
│   └── pipeline.py
│
├── frontend/
│   └── app.py
│
├── notebooks/
│   └── experiments.ipynb
│
├── data/
│   ├── slides/
│   └── vectorstore/
│
├── setup_index.py
├── evaluate.py
├── requirements.txt
└── .env.example
```

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd study_assistant
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📄 Add Course Slides

Place your course PDF files inside:

```text
data/slides/
```

For example:

```text
data/
└── slides/
    ├── lecture_01.pdf
    ├── lecture_02.pdf
    ├── lecture_03.pdf
    └── lecture_04.pdf
```

The application is designed around a **fixed course knowledge base**.

Users do not upload documents during interaction. The course slides are prepared and indexed beforehand.

---

# 🔐 Environment Variables

The project works with default local settings.

If you want to use a hosted Hugging Face model, create a `.env` file based on:

```text
.env.example
```

and provide your Hugging Face access token.

Never commit your actual `.env` file or access tokens to GitHub.

---

# 🗂️ Build the Vector Index

Run:

```bash
python setup_index.py
```

This process:

1. Loads the course PDFs
2. Extracts their text
3. Splits the documents into chunks
4. Generates embeddings
5. Builds the FAISS index
6. Saves the vector store

Run this command again whenever the course slides are modified.

---

# 📊 Run Retrieval Evaluation

Run:

```bash
python evaluate.py
```

The evaluation script reports retrieval metrics including:

* Precision@5
* Recall@5
* Hit Rate@5
* MRR

---

# ▶️ Launch the Application

Run:

```bash
streamlit run frontend/app.py
```

The Streamlit interface provides an interactive way to ask questions and experiment with different pipeline configurations.

---

# 🔬 Experiments

The notebook:

```text
notebooks/experiments.ipynb
```

provides a step-by-step walkthrough of the major components.

Suggested experiments include:

### Experiment 1 — Document Processing

Inspect PDF loading, chunking, and metadata.

### Experiment 2 — Semantic Retrieval

Inspect which chunks are retrieved for different questions.

### Experiment 3 — Zero-shot vs Few-shot

Compare answer generation with and without examples.

### Experiment 4 — Memory Strategies

Compare:

```text
Buffer
Window
Summary
Entity
```

in multi-turn conversations.

### Experiment 5 — Chain Architectures

Compare:

```text
Simple Sequential
Sequential
MapReduce
Refine
```

for different question types.

### Experiment 6 — Retrieval Configuration

Experiment with:

* Different embedding models
* Different chunk sizes
* Different overlap values
* Different values of K
* Similarity thresholds

### Experiment 7 — End-to-End Pipeline

Compare complete configurations and investigate how retrieval, prompting, memory, and chains affect final answer quality.

---

# 🛡️ Grounding & Out-of-Scope Questions

The assistant is designed to ground its responses in the indexed course material.

If the retrieved context does not provide sufficient evidence for a question, the system should avoid presenting unsupported information as though it came from the course material.

For example:

```text
User:
What is the capital of Japan?

Assistant:
I couldn't find sufficient evidence in the course material
to answer this question.
```

This behavior helps reduce hallucination and keeps the assistant aligned with its predefined knowledge based comparison of different pipeline configurations

---

# 🎓 Learning Outcomes

This project demonstrates the practical integration of concepts from the **Prompt Engineering course by Abu Bakr** into a complete LLM application.

The project connects individual concepts into an end-to-end workflow:

```text
Prompt Engineering
        +
Memory
        +
Chains
        +
Embeddings
        +
Semantic Retrieval
        +
RAG
        +
LLM Generation
        ↓
Complete LLM Application
```

The main objective is to move from **understanding individual concepts to applying them together in a practical system**.

---

# 📌 Key Takeaway

The project is not intended to be just a chatbot.

It is an **interactive RAG experimentation platform and study assistant** designed to demonstrate how different Prompt Engineering, Memory, Chain, and Retrieval strategies can be combined and evaluated within a real-world-style LLM application.

---

# 👩‍💻 Project Information

**Project:** Adaptive RAG Study Assistant
**Type:** Capstone Project
**Course:** Prompt Engineering — Abu Bakr
**Domain:** Retrieval-Augmented Generation / LLM Applications
**Language:** Python

---

## ⭐ Acknowledgment

Developed as part of the **Prompt Engineering course by Abu Bakr**, with the goal of applying the concepts learned throughout the course in a practical end-to-end project.
