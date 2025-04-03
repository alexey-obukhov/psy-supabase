"""
embedding_utils.py

This module provides utility functions for formatting embedding vectors to be compatible with PostgreSQL's
pgvector extension. These functions ensure that embeddings, which may come in various formats (e.g., lists,
numpy arrays, PyTorch tensors), are properly converted into the string format required for storage and retrieval
in a PostgreSQL database.

Key Features:
- Handles embeddings in different formats (lists, numpy arrays, PyTorch tensors).
- Converts embeddings into a compact, space-free string format compatible with pgvector.
- Ensures consistency and compatibility for database operations involving vector embeddings.

Functions:
- format_embedding_for_db_obs(embedding): Formats an embedding vector for PostgreSQL pgvector.
- format_embedding_for_db(embedding): Formats an embedding vector into a compact string for pgvector.

Usage:
    from psy_supabase.utilities.embedding_utils import format_embedding_for_db

    # Example embedding (list format)
    embedding = [0.1, 0.2, 0.3]

    # Format the embedding for PostgreSQL
    formatted_embedding = format_embedding_for_db(embedding)
    print(formatted_embedding)  # Output: [0.1,0.2,0.3]

    # Example embedding (numpy array)
    import numpy as np
    embedding_np = np.array([0.1, 0.2, 0.3])

    # Format the numpy embedding
    formatted_embedding_np = format_embedding_for_db(embedding_np)
    print(formatted_embedding_np)  # Output: [0.1,0.2,0.3]
"""
from typing import List, Dict

def format_embedding_for_db(embedding):
    """
    Format an embedding vector for Postgres pgvector.

    Args:
        embedding: Embedding vector (could be list, numpy array, torch tensor, etc.)

    Returns:
        Properly formatted string for pgvector
    """
    # Handle torch tensors
    if hasattr(embedding, 'tolist') and callable(getattr(embedding, 'tolist')):
        # Convert torch tensor or numpy array to list
        embedding = embedding.tolist()

    # Handle complex number tensors by taking real part only (if needed)
    if any(isinstance(x, complex) for x in embedding):
        embedding = [float(x.real) for x in embedding]
    else:
        # Ensure all values are floats (not numpy.float32 or similar)
        embedding = [float(x) for x in embedding]

    # Format as string with square brackets for pgvector
    return '[' + ','.join(str(x) for x in embedding) + ']'

def format_embedding_for_db_old(embedding):
    """Format embedding vector for PostgreSQL's pgvector extension."""
    if hasattr(embedding, 'tolist') and callable(getattr(embedding, 'tolist')):
        embedding = embedding.tolist()

    # Format as [0.1,0.2,0.3,...] - no spaces
    return '[' + ','.join(str(float(x)) for x in embedding) + ']'

def detect_repetition_pattern(original_question: str, current_question: str, similar_questions: List[Dict]) -> Dict:
    """
    Analyze repetition patterns in similar questions to detect psychological fixation.

    Args:
        original_question: The first occurrence of this question
        current_question: The current question
        similar_questions: List of similar questions identified

    Returns:
        Dictionary with repetition pattern data
    """
    import re
    from psy_supabase.utilities.keep_words import keep_words

    # Count occurrences of highly similar questions with threshold 0.7 from supabase queries
    high_similarity_count = sum(1 for q in similar_questions if q['similarity'] > 0.7)

    # Extract key terms that appear in both original and current question
    original_terms = set(re.findall(r'\b\w+\b', original_question.lower()))
    current_terms = set(re.findall(r'\b\w+\b', current_question.lower()))

    recurring_terms = original_terms.intersection(current_terms)

    significant_terms = [term for term in recurring_terms if term not in keep_words and len(term) > 2]

    return {
        'count': high_similarity_count,
        'recurring_terms': list(significant_terms),
        'is_fixation': high_similarity_count >= 3,  # Consider it fixation if asked 3+ times
        'intensity': min(high_similarity_count / 5, 1.0)  # Scale intensity, max 1.0
    }
