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
