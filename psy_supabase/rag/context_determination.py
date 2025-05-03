"""
Context determination for Retrieval-Augmented Generation (RAG).

This module handles the context retrieval and formatting process for the RAG system,
enabling the generation of more relevant, personalized, and coherent responses by
providing past conversation context to the language model.

Key features:
- Semantic search for relevant past interactions using vector similarity
- Session-based context filtering to maintain conversation continuity
- Robust formatting of interaction history for LLM consumption
- Graceful error handling with detailed logging

The module integrates with:
1. Database manager for retrieving interaction records
2. Embedding provider for vector representation of text
3. Vector utilities for similarity search operations
4. Response generation pipeline that consumes the prepared context

This context-aware approach allows the system to:
- "Remember" previous conversations with users
- Provide more consistent and personalized responses
- Better understand user intent by considering conversation history
"""

import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from prismalog.log import get_logger
from typeguard import typechecked

from psy_supabase.core.model_manager import get_embedding_provider
from psy_supabase.utilities.vector_utils import find_similar_interactions

if TYPE_CHECKING:
    from psy_supabase.core.database import DatabaseManager


logger = get_logger(__name__)


def determine_context(
    db_manager: "DatabaseManager", user_input: str, session_id: Optional[str] = None, limit: int = 5
) -> str:
    """
    Determine relevant context for RAG based on user input and past interactions.

    This function coordinates the end-to-end context retrieval process:
    1. Generating an embedding for the current user input
    2. Finding semantically similar past interactions
    3. Formatting those interactions as context for the language model

    The function intelligently handles errors to ensure the conversation can
    continue even if context retrieval fails.

    Args:
        db_manager: Database manager object with access to interaction records
        user_input: Current user input text requiring context
        session_id: Optional session ID to filter interactions by specific conversation
        limit: Maximum number of past interactions to include in context

    Returns:
        str: Formatted context string ready for inclusion in LLM prompt,
             or empty string if context retrieval fails
    """
    try:
        # Get embedding provider
        provider = get_embedding_provider()
        if not provider:
            logger.error("Failed to get embedding provider in determine_context.")
            return ""

        # Generate embedding for user input
        embedding = provider.generate_embedding(user_input)
        if not embedding:
            logger.error("Failed to generate embedding for user input in determine_context.")
            return ""

        # Get context from similar interactions
        context = create_context_from_similar_interactions(
            db_manager=db_manager, embedding=embedding, session_id=session_id, limit=limit
        )

        return context
    except Exception as e:
        logger.error("Error determining context: %s", e)
        logger.error(traceback.format_exc())
        return ""


def extract_relevant_interactions(
    db_manager: "DatabaseManager", embedding: List[float], session_id: Optional[str] = None, limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Extract relevant past interactions based on semantic similarity.

    This function performs a vector similarity search to find past interactions
    that are semantically related to the current user input. It uses the vector
    embedding of the user's input to find conversations that discussed similar
    topics or contained similar questions.

    The search can be filtered by session ID to prioritize continuity within
    the same conversation. The similarity threshold is intentionally set lower
    than in other search operations to capture a wider range of potential matches.

    Args:
        db_manager: Database manager with access to interaction records
        embedding: Vector embedding representing the current user input
        session_id: Optional session ID to filter interactions by conversation
        limit: Maximum number of interactions to return

    Returns:
        list: List of relevant interaction records sorted by similarity,
              each containing question, answer, and metadata fields
              Returns empty list if retrieval fails
    """
    try:
        # Use vector_utils to find similar interactions
        interactions: List[Dict[str, Any]] = find_similar_interactions(
            db_manager=db_manager,
            embedding=embedding,
            session_id=session_id,
            limit=limit,
            threshold=0.6,  # Lower threshold for more potential matches
        )

        return interactions
    except Exception as e:
        logger.error("Error extracting relevant interactions: %s", e)
        logger.error(traceback.format_exc())
        return []


def format_interactions_as_context(interactions: List[Dict[str, Any]]) -> str:
    """
    Format a list of interactions as a context string for RAG.

    Transforms raw interaction records into a structured, readable format
    that can be included in prompts to the language model. The formatting
    preserves the question-answer pairs and their sequence, providing
    conversation flow context to the model.

    The function safely handles potential missing fields and empty interactions
    by implementing defensive programming techniques to prevent errors like
    "'list' object has no attribute 'get'".

    Args:
        interactions: List of interaction records containing at minimum
                     'question' and 'answer' fields

    Returns:
        str: Formatted context string with numbered interactions and
             clear question/answer delineation, or empty string if
             no valid interactions are provided
    """
    if not interactions:
        return ""

    context_parts: List[str] = []

    for i, interaction in enumerate(interactions):
        # Extract fields safely
        question = interaction.get("question", "")
        answer = interaction.get("answer", "")

        if not question and not answer:
            continue

        # Format as context entry
        context_part = f"Interaction {i+1}:\nQuestion: {question}\nAnswer: {answer}"
        context_parts.append(context_part)

    context = "\n\n".join(context_parts)

    return context


@typechecked
def create_context_from_similar_interactions(
    db_manager: "DatabaseManager", embedding: List[float], session_id: Optional[str] = None, limit: int = 5
) -> str:
    """
    Create a complete context string from semantically similar past interactions.

    This convenience function combines the extraction and formatting steps
    to generate ready-to-use context from vector embeddings. It encapsulates
    the entire retrieval and formatting process for easier integration with
    the response generation pipeline.

    Args:
        db_manager: Database manager with access to interaction records
        embedding: Vector embedding representing the current user input
        session_id: Optional session ID to filter interactions by conversation
        limit: Maximum number of interactions to include in the context

    Returns:
        str: Fully formatted context string ready for inclusion in LLM prompts,
             or empty string if no relevant interactions are found
    """
    # Get similar interactions
    interactions = extract_relevant_interactions(
        db_manager=db_manager, embedding=embedding, session_id=session_id, limit=limit
    )

    # Format as context
    context = format_interactions_as_context(interactions)

    return context
