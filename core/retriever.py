import numpy as np
import faiss
from rank_bm25 import BM25Okapi

from config import FAISS_INDEX_PATH
from storage.database import Database
from core.embedding import EmbeddingEngine


class HybridRetriever:
    def __init__(self):
        self.db = Database()
        self.engine = EmbeddingEngine()

        # Load all chunks
        self.rows = self.db.get_all_chunks()

        self.chunk_ids = []
        self.texts = []
        self.metadata = []

        for row in self.rows:
            self.chunk_ids.append(row["id"])
            self.texts.append(row["content"])
            self.metadata.append({
                "title": row["title"],
                "page": row["page_number"]
            })

        # Build BM25
        tokenized_corpus = [text.split() for text in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Load FAISS index
        self.index = faiss.read_index(FAISS_INDEX_PATH)

    # -----------------------------------
    # Normalize scores to 0-1
    # -----------------------------------
    def _normalize(self, scores):
        scores = np.array(scores)

        if scores.max() > scores.min():
            return (scores - scores.min()) / (scores.max() - scores.min())
        else:
            return scores

    # -----------------------------------
    # Hybrid Retrieval
    # -----------------------------------
    def retrieve(self, query: str, alpha: float = 0.5, top_k: int = 5):

        # Dense search
        query_vec = self.engine.embed_query(query).astype("float32")
        dense_scores, dense_indices = self.index.search(
            np.array([query_vec]),
            len(self.texts)
        )

        dense_scores = dense_scores[0]
        dense_indices = dense_indices[0]
        dense_scores = self._normalize(dense_scores)

        dense_dict = {
            dense_indices[i]: dense_scores[i]
            for i in range(len(dense_indices))
        }

        # Sparse search
        tokenized_query = query.split()
        sparse_scores = self.bm25.get_scores(tokenized_query)
        sparse_scores = self._normalize(sparse_scores)

        # Combine
        final_scores = []

        for i in range(len(self.texts)):
            dense = dense_dict.get(i, 0.0)
            sparse = sparse_scores[i]

            score = alpha * dense + (1 - alpha) * sparse
            final_scores.append(score)

        final_scores = np.array(final_scores)

        ranked_indices = final_scores.argsort()[::-1][:top_k]

        results = []

        for idx in ranked_indices:
            results.append({
                "chunk_id": self.chunk_ids[idx],
                "text": self.texts[idx],
                "score": float(final_scores[idx]),
                "metadata": self.metadata[idx]
            })

        return results