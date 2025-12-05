"""
Services & Infrastructure Module
================================

This module provides the foundational AI services required by the system.
It encapsulates external model interactions (LLMs, VLMs, Embeddings) behind
unified, thread-safe, and robust interfaces.

Module Overview
---------------
- **llm_agent.py**: The central gateway for Large Language Model interactions.
  It handles API communication (OpenAI/Azure), multimodal input formatting,
  automatic retries, cost tracking, and response validation against Pydantic schemas.

- **embedding.py**: A singleton service for generating text embeddings using
  Sentence Transformers. It ensures thread-safety and efficient resource usage
  across parallel processes (e.g., during domain fusion).
"""

from unidomain.services.embedding import TextEmbeddingAPI
from unidomain.services.llm_agent import LLMAgent, execute_llm_task

__all__ = ["TextEmbeddingAPI", "LLMAgent", "execute_llm_task", "setup_client"]
