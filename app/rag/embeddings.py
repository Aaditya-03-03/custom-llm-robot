import logging
from typing import List
from app.rag.config import rag_settings

logger = logging.getLogger("custom_llm_robot.rag.embeddings")

class EmbeddingManager:
    """
    Sentence-Transformer embeddings wrapper using BAAI/bge-small-en-v1.5.
    Produces 384-dimensional normalized vector embeddings.
    """

    def __init__(self, model_name: str = rag_settings.EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            logger.info(f"Loading SentenceTransformer model '{self.model_name}'...")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info("SentenceTransformer embedding model loaded successfully.")
        return self._model

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single string."""
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a batch of strings."""
        if not texts:
            return []
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()
