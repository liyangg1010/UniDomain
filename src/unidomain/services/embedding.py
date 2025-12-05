"""
Text Embedding Service
======================

This module provides a thread-safe singleton API for generating text embeddings.
It wraps the "sentence-transformers" library to convert text strings into
dense vector representations, which are essential for semantic similarity calculations.

Architecture Position:
    [Services / Infrastructure] -> Used by domain_fusion for similarity checks.
"""

import os
import threading
from typing import List

from numpy.typing import NDArray

from unidomain.configs.constants import EnvVar

__all__ = ["TextEmbeddingAPI"]

# Default model if not specified in environment variable
_DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

class TextEmbeddingAPI:
    """Singleton API class for text embedding using sentence-transformers.
    
    This class ensures that the heavy embedding model is loaded only once per process,
    even when accessed by multiple threads in the domain fusion pipeline.
    """

    _load_lock = threading.Lock()
    _shared_model = None

    def __init__(self) -> None:
        """Initialize the API wrapper.
        
        Note: The actual model loading is deferred until the first call to `model`.
        """
        pass

    @property
    def model(self):
        """Thread-safe property to access the shared SentenceTransformer instance.
        
        Implements the Double-Checked Locking pattern for lazy initialization.
        """
        if TextEmbeddingAPI._shared_model is None:
            with TextEmbeddingAPI._load_lock:
                if TextEmbeddingAPI._shared_model is None:
                    self._load_model()
        return TextEmbeddingAPI._shared_model
    
    def text_embedding(self, texts: List[str]) -> NDArray:
        """Generate embeddings for a list of texts.

        Args:
            texts (List[str]): A list of input strings.

        Returns:
            NDArray: A numpy array of shape (n_texts, embedding_dim).
        """
        return self.model.encode(texts)

    def _load_model(self):
        """Internal method to load the model from disk or HuggingFace Hub."""
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Missing dependencies for embedding service. "
                "Please install them via: 'pip install -e .[fusion]'"
            )
        
        # Priority:
        # 1. Environment Variable (Highest)
        # 2. Default Constant
        model_path = os.getenv(EnvVar.EMBEDDING_MODEL_PATH)
        if not model_path:
            model_path = _DEFAULT_MODEL_NAME

        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model '{model_path}' on {device}...")
        try:
            TextEmbeddingAPI._shared_model = SentenceTransformer(str(model_path), device=device)
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model from '{model_path}': {e}")
