from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL


class EmbeddingEngine:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.dimension = self.model.get_sentence_embedding_dimension()

    # -----------------------------------
    # Embed multiple documents
    # -----------------------------------
    def embed_documents(self, texts):
        """
        texts: list[str]
        returns: numpy array (n, dimension)
        """
        return self.model.encode(
            texts,
            normalize_embeddings=True
        )

    # -----------------------------------
    # Embed single query
    # -----------------------------------
    def embed_query(self, query: str):
        """
        query: str
        returns: numpy array (dimension,)
        """
        return self.model.encode(
            [query],
            normalize_embeddings=True
        )[0]