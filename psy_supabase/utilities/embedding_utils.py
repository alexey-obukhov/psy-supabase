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
def format_embedding_for_db_obs(embedding):
    """
    Format an embedding vector for PostgreSQL pgvector.

    Args:
        embedding: Embedding vector (list, numpy array, or tensor)

    Returns:
        str: Formatted embedding string for PostgreSQL
    """
    # Handle different embedding types
    if hasattr(embedding, 'tolist') and callable(getattr(embedding, 'tolist')):
        # Handle numpy arrays or PyTorch tensors
        embedding_list = embedding.tolist()
    elif isinstance(embedding, list):
        # Already a list
        embedding_list = embedding
    else:
        # Try conversion to list
        embedding_list = list(embedding)

    # Convert to string format expected by pgvector
    return str(embedding_list).replace(' ', '')

def format_embedding_for_db(embedding):
    """Format embedding vector for PostgreSQL's pgvector extension."""
    if hasattr(embedding, 'tolist') and callable(getattr(embedding, 'tolist')):
        embedding = embedding.tolist()

    # Format as [0.1,0.2,0.3,...] - no spaces
    return '[' + ','.join(str(float(x)) for x in embedding) + ']'
