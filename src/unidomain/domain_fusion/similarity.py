"""
Similarity Calculation Utilities
================================

This module provides low-level utility functions for calculating semantic
similarity between strings using vector embeddings. It serves as the foundational
computation layer for higher-level merging modules.

Functions:
    filter_and_sort_strings: Filters and sorts a list of strings based on
        their semantic similarity to a query string.
"""

from typing import List, Union

import numpy as np
from numpy.typing import NDArray

from unidomain.services.embedding import TextEmbeddingAPI

__all__ = ["filter_and_sort_strings"]


def _cosine_similarity(vector_a: NDArray, vector_b: NDArray) -> np.float64:
    """Compute the cosine similarity between two vectors.

    Args:
        vector_a (NDArray): The first vector.
        vector_b (NDArray): The second vector.

    Returns:
        np.float64: The cosine similarity score (-1.0 to 1.0).
            Returns 0.0 if either vector has zero norm to avoid division by zero.
    """
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return np.float64(0.0)
    
    return np.dot(vector_a, vector_b) / (norm_a * norm_b)


def _preprocess_action(action: Union[str, List[str]]) -> Union[str, List[str]]:
    """Preprocess action strings by normalizing delimiters.

    Replace hyphens and underscores with spaces to improve embedding semantic matching.

    Args:
        action (Union[str, List[str]]): The action string or list of action strings.

    Returns:
        Union[str, List[str]]: The preprocessed action(s).

    Raises:
        TypeError: If the input is neither a string nor a list of strings.
    """
    if isinstance(action, str):
        return action.replace("-", " ").replace("_", " ")
    elif isinstance(action, list):
        return [a.replace("-", " ").replace("_", " ") for a in action]
    else:
        raise TypeError(f"Expected str or List[str], got {type(action)}")


def _semantic_similarity(
    name: str,
    candidate_names: List[str],
) -> List[np.float64]:
    """Compute semantic similarity between a query string name and a list of candidate string names.

    Args:
        name (str): The string name to compare.
        candidate_names (List[str]): A list of candidate string names.
            
    Returns:
        List[np.float64]: A list of cosine similarity scores corresponding to candidate names.
    """

    # Note: Preprocessing is currently disabled based on original design,
    # but the function `_preprocess_action` is retained for potential future use.
    # name = _preprocess_action(name)
    # candidate_names = _preprocess_action(candidate_names)


    # Using the same loaded model for this function, from API class.
    text_embedding_api = TextEmbeddingAPI()

    name = text_embedding_api.text_embedding([name])[0]
    candidate_names = text_embedding_api.text_embedding(candidate_names)

    similarities = [
        _cosine_similarity(name, candidate_name)
        for candidate_name in candidate_names
    ]

    return similarities


def filter_and_sort_strings(query: str, candidates: List[str], threshold: float):
    """Filter strings by similarity threshold and sorts them by score.

    This function calculates the semantic similarity between the query string
    and each candidate. Only candidates with a score strictly greater than
    the threshold are returned, sorted in descending order of similarity.

    Args:
        query (str): The string to compare.
        candidates (List[str]): The list of strings to filter and sort.
        threshold (float): The minimum similarity score required.

    Returns:
        List[str]: A list of strings that meet the threshold, sorted by similarity.
    """
    if not candidates:
        return []
    
    scored_strings = []
    similarities = _semantic_similarity(query, candidates)

    # Filter out candidates with similarity below the threshold
    scored_strings = [(candidates[i], similarities[i]) for i in range(len(candidates))]
    scored_strings = [k for k in scored_strings if k[1] > threshold]

    # Sort the remaining candidates by similarity, in descending order
    scored_strings = sorted(scored_strings, key=lambda k: k[1], reverse=True)

    return [k[0] for k in scored_strings]
