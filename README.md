# Hybrid RAG Study Assistant

A grounded hybrid Retrieval-Augmented Generation (RAG) system that combines dense and sparse retrieval with citation enforcement and abstention control.

---

## 🚀 Overview

This project implements a hybrid retrieval pipeline that integrates:

- **Dense retrieval** (BGE embeddings + FAISS)
- **Sparse retrieval** (BM25)
- **Hybrid score interpolation**
- **Grounded answer generation**
- **Citation enforcement**
- **Abstention mechanism for low-confidence responses**

The system is designed to reduce hallucinations while improving retrieval robustness.

---

## 🏗 System Architecture

### Offline Phase
1. PDF ingestion
2. Text chunking (configurable size + overlap)
3. Sparse indexing (BM25)
4. Dense embedding generation (BGE)
5. FAISS vector indexing

### Online Phase
1. Query embedding
2. Hybrid score computation:
   
   hybrid_score = α * dense_score + (1 − α) * sparse_score

3. Top‑k retrieval
4. Grounded prompt construction
5. Citation‑enforced answer generation
6. Abstention if insufficient evidence

---

## 📊 Experimental Results

Hybrid retrieval outperformed standalone sparse retrieval in evaluation experiments.

| Alpha | Retrieval Mode | Relative Performance |
|--------|----------------|----------------------|
| 0.0 | Sparse (BM25) | Baseline |
| 0.25 | Hybrid | Improved |
| 0.5 | Hybrid | Strong |
| 0.75 | Hybrid | Strong |
| 1.0 | Dense (BGE) | Competitive |

The evaluation framework supports configurable alpha sweeps for retrieval comparison.

---

## 🧠 Key Features

- Dense + Sparse hybrid ranking
- FAISS vector search
- SQLite metadata storage
- Modular pipeline design
- Evaluation module for retrieval experiments
- Citation-grounded response generation
- Abstention when evidence is weak

---

## 📂 Project Structure

core/ → Retrieval, embedding, generation logic
ingestion/ → PDF processing & chunking
storage/ → Database & schema
evaluation/ → Experimental evaluation framework

---

## ⚙️ Installation

```bash
git clone https://github.com/parkjw5/hybrid-rag-study-assistant.git
cd hybrid-rag-study-assistant
pip install -r requirements.txt
```

▶️ Run the Application
bash
python app.py

🎯 Future Improvements
Cross-encoder reranking
Streaming response support
Web interface (Streamlit / FastAPI)
Retrieval benchmarking on larger corpora
Deployment-ready API version

📌 Tech Stack
Python
FAISS
BM25
SQLite
HuggingFace embeddings
Large Language Models (LLMs)
Pandas

📄 License
MIT License
---

# ✅ After Updating README

1. Save the file
2. Run:

```bash
git add README.md
git commit -m "Improve README with architecture and documentation"
git push
```

