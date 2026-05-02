# ==============================
# MODELS
# ==============================

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
LLM_MODEL = "qwen2.5:14b-instruct"


# ==============================
# CHUNKING
# ==============================

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


# ==============================
# RETRIEVAL
# ==============================

TOP_K = 5
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


# ==============================
# STORAGE PATHS
# ==============================

DB_PATH = "data/rag.sqlite"
FAISS_INDEX_PATH = "data/faiss.index"
PDF_FOLDER = "data/pdfs"