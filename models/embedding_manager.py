import torch
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Handles embedding generation and similarity calculations"""

    def __init__(self, model_id: str = "google/embeddinggemma-300M", cache_dir: str = "models_cache"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache_dir = cache_dir
        self.model_id = model_id

        # Set Hugging Face token
        os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

        logger.info(f"Loading embedding model {model_id} on {self.device}")

        try:
            # Load model with caching
            self.model = SentenceTransformer(
                model_id,
                cache_folder=cache_dir,
                use_auth_token=os.getenv("HF_TOKEN")
            ).to(device=self.device)

            # Print model info
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Model loaded on: {self.model.device}")
            logger.info(f"Total parameters: {total_params:,}")

        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise

    def encode_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query"""
        try:
            embedding = self.model.encode(
                query,
                prompt_name="Retrieval-query",
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embedding
        except Exception as e:
            logger.error(f"Error encoding query: {e}")
            raise

    def encode_document(self, text: str, title: str = None) -> np.ndarray:
        """Generate embedding for a document"""
        try:
            if title and title.strip():
                embedding = self.model.encode(
                    text,
                    prompt=f"title: {title} | text: ",
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
            else:
                embedding = self.model.encode(
                    text,
                    prompt="title: none | text: ",
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
            return embedding
        except Exception as e:
            logger.error(f"Error encoding document: {e}")
            raise

    def calculate_similarity(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between query and document embeddings"""
        try:
            return cosine_similarity([query_embedding], doc_embeddings)[0]
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            raise
