"""
dynamic_rag.py

This module implements the DynamicRAGRetriever class, which provides dynamic retrieval and analysis capabilities
for a psychological AI system. It enables efficient and relevant responses by fetching only the necessary
knowledge, past interactions, and related concepts during model generation.

Key Features:
- Dynamic knowledge retrieval based on user queries, with support for schema-specific searches and similarity-based document retrieval.
- Retrieval of past user interactions, optionally filtered by topic, to maintain conversational context.
- Analysis of related psychological concepts using vector similarity for enhanced therapeutic insights.
- Emotion and topic analysis to better understand user input and provide tailored responses.
- Detection of recurring pain points in user interactions to identify key areas of concern and recommend therapeutic approaches.
- Caching to avoid redundant database queries and improve performance.

Classes:
- DynamicRAGRetriever: The main class that provides methods for dynamic retrieval, analysis, and caching.

Dependencies:
- psy_supabase.core.database.DatabaseManager: Handles database operations such as embedding creation and document retrieval.
- prismalog.log.ColoredLogger: Provides enhanced logging capabilities for debugging and monitoring.

Usage:
    db_manager = DatabaseManager(...)
    retriever = DynamicRAGRetriever(db_manager, session_id="user_session_123")
    knowledge = retriever.get_knowledge_by_query("anxiety management")
    past_interactions = retriever.get_past_interactions(topic="anxiety")
    emotion_analysis = retriever.analyze_emotion("I'm feeling very stressed lately.")
    pain_point = retriever.get_pain_point()
"""

import json
import traceback
from json import JSONDecodeError
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from prismalog.log import get_logger

from psy_supabase.memory.associative_memory import AssociativeMemory
from psy_supabase.utilities.stop_words import stop_words

if TYPE_CHECKING:
    from psy_supabase.core.database import DatabaseManager

    from .rag_processor import RAGProcessor

logger = get_logger(__name__)


class DynamicRAGRetriever:
    """
    Dynamic RAG retriever that delegates to RAGProcessor methods.
    """

    def __init__(
        self,
        db_manager: "DatabaseManager",
        session_id: Optional[str] = None,
        persona: Optional[Any] = None,
        query_mode: Optional[Any] = None,
        embedding_provider: Optional[Any] = None,
        rag_processor: Optional["RAGProcessor"] = None,
    ):
        """
        Initialize the dynamic RAG retriever.

        Args:
            db_manager: Database manager instance
            session_id: Optional session ID for context filtering
            embedding_provider: Optional custom embedding provider
            persona: Optional persona to use for context retrieval
            query_mode: Optional query mode (e.g., 'semantic', 'hybrid')
            **kwargs: Additional parameters (to capture unexpected params)
        """
        self.db_manager = db_manager
        self.session_id = session_id
        self.persona = persona
        self.query_mode = query_mode or "hybrid"
        self._embedding_provider = embedding_provider
        self.rag_processor = rag_processor

        # Initialize associative memory component for enhanced retrieval
        self.associative_memory = AssociativeMemory()
        self.memory_initialized = False

        self.query_cache: Dict[str, str] = {}  # Cache for query results
        self._last_raw_results: List[Dict[str, Any]] = []  # Store raw results for testing

    def get_conversation_context(self, limit: int = 3) -> str:
        """
        Get conversation context for the current session.

        Args:
            limit: Maximum number of turns to include

        Returns:
            String representation of conversation context
        """
        try:
            if not self.session_id:
                return ""

            # Use the existing get_past_interactions method which already works with our database
            interactions = self.get_past_interactions(session_id=self.session_id, limit=limit)

            if not interactions:
                return ""

            # Format the conversation history into a string
            context = ""
            for interaction in interactions:
                q = interaction.get("question", "")
                a = interaction.get("answer", "")
                if q and a:
                    context += f"User: {q}\nAssistant: {a}\n\n"

            return context.strip()
        except Exception as e:
            logger.error("Error getting conversation context: '%s'", e)
            return ""

    def get_knowledge_by_query(
        self, query: str, associative_memory: bool = False, min_similarity: float = 0.1, **kwargs: Any
    ) -> str:
        """
        Get knowledge relevant to a query using vector similarity search.

        Args:
            query: The text query to search for relevant knowledge
            associative_memory: Whether to use associative memory for enhanced retrieval
            min_similarity: Minimum similarity threshold for results
            **kwargs: Additional parameters like session_id and limit

        Returns:
            String containing relevant knowledge from the database
        """
        try:
            session_id = kwargs.get("session_id", self.session_id)
            limit = kwargs.get("limit", 5)

            # Create cache key
            cache_key = f"{query}_{session_id}_{associative_memory}_{min_similarity}"
            if cache_key in self.query_cache:
                return self.query_cache[cache_key]

            # Generate embedding
            query_embedding = self.db_manager.create_embedding(query)
            if not query_embedding:
                return "Failed to generate embedding for query"

            # Search for similar interactions
            similar = self.db_manager.find_similar_interactions_by_embedding(
                embedding=query_embedding, session_id=session_id, limit=limit, threshold=min_similarity
            )

            if not similar:
                return "No relevant interactions found."

            # Process results
            result_parts = []

            # Store raw results for testing and internal use
            self._last_raw_results = []

            for interaction in similar:
                question = interaction.get("question", "")
                answer = interaction.get("answer", "")
                similarity = interaction.get("similarity", 0)

                if answer:
                    # Just add the answer text (cleaner for user display)
                    result_parts.append(answer)

                    # Store the full interaction with debug info
                    self._last_raw_results.append(
                        {
                            "question": question,
                            "answer": answer,
                            "similarity": similarity,
                            "interaction_id": interaction.get("interaction_id", ""),
                        }
                    )

            # Combine results for user display
            combined_results = "\n\n".join(result_parts)

            # Cache the result
            self.query_cache[cache_key] = combined_results

            return combined_results

        except Exception as e:
            logger.error("Error in get_knowledge_by_query: %s", e)
            logger.error(traceback.format_exc())
            return f"Error retrieving knowledge: {str(e)}"

    def analyze_emotion(self, text: str) -> Dict:
        """
        Analyze the emotion expressed in the text.

        Args:
            text: The text to analyse

        Returns:
            Dictionary with emotion analysis
        """
        try:
            # Simple keyword-based analysis for testing
            text = text.lower()

            # Define emotion keywords
            emotion_keywords = {
                "anger": ["angry", "furious", "mad", "upset", "irritated", "annoyed"],
                "sadness": ["sad", "depressed", "down", "unhappy", "miserable", "lonely"],
                "anxiety": ["anxious", "worried", "nervous", "stressed", "tense", "afraid"],
                "fear": ["scared", "terrified", "frightened", "panicked", "afraid", "fearful"],
                "joy": ["happy", "joyful", "delighted", "pleased", "glad", "excited"],
                "gratitude": ["thankful", "grateful", "appreciative", "blessed", "fortunate"],
            }

            # Count emotion keywords
            emotion_counts = {}
            for emotion, keywords in emotion_keywords.items():
                count = sum(1 for keyword in keywords if keyword in text)
                if count > 0:
                    emotion_counts[emotion] = count

            # Calculate sentiment
            negative_emotions = ["anger", "sadness", "anxiety", "fear"]
            positive_emotions = ["joy", "gratitude"]

            negative_score = sum(emotion_counts.get(emotion, 0) for emotion in negative_emotions)
            positive_score = sum(emotion_counts.get(emotion, 0) for emotion in positive_emotions)

            total_score = positive_score - negative_score
            sentiment = total_score / (positive_score + negative_score) if (positive_score + negative_score) > 0 else 0

            # Find dominant emotion
            dominant_emotion = None
            max_count = 0
            for emotion, count in emotion_counts.items():
                if count > max_count:
                    max_count = count
                    dominant_emotion = emotion

            return {
                "sentiment": sentiment,
                "emotions": emotion_counts,
                "dominant_emotion": dominant_emotion,
                "confidence": min(max_count * 0.2, 0.9) if dominant_emotion else 0.0,
            }

        except Exception as e:
            logger.error("Error analysing emotion: %s", e)
            return {"sentiment": 0, "emotions": {}, "dominant_emotion": None, "confidence": 0.0}

    def _initialize_memory_from_db(self) -> None:
        """Load relevant session data into associative memory."""
        if self.memory_initialized:
            return

        if not self.session_id:
            logger.warning("Cannot initialize memory without a session ID.")
            return

        try:

            # Get conversation history instead of non-existent session documents
            history = self.db_manager.get_conversation_history(self.session_id)

            # Add each interaction to associative memory with topics
            for interaction in history:
                # Use 'answer' as the primary content, fallback to 'question'
                content = interaction.get("answer", "")
                if not content:
                    content = interaction.get("question", "")  # Fallback if answer is empty
                if not content:
                    continue  # Skip if both are empty

                # Extract topics from metadata (assuming metadata might be present)
                metadata = interaction.get("metadata", {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except JSONDecodeError:  # Catch specific JSONDecodeError if possible
                        metadata = {}

                topics = []
                # Extract topics from various metadata fields
                if "topics" in metadata:
                    if isinstance(metadata["topics"], list):
                        topics.extend(metadata["topics"])
                    elif isinstance(metadata["topics"], str):
                        topics.extend([t.strip() for t in metadata["topics"].split(",")])

                if "related_topics" in metadata:
                    if isinstance(metadata["related_topics"], list):
                        topics.extend(metadata["related_topics"])
                    elif isinstance(metadata["related_topics"], str):
                        topics.extend([t.strip() for t in metadata["related_topics"].split(",")])

                if "category" in metadata:
                    topics.append(metadata["category"])

                # If no topics found, extract keywords from content
                if not topics:
                    topics = self._extract_keywords(content)

                # Add to associative memory
                self.associative_memory.add_memory(content, topics, metadata)

            self.memory_initialized = True
            logger.info("Initialized associative memory with %d interactions", len(history))

        except Exception as e:
            logger.error("Error initializing associative memory: %s", e)
            logger.error(traceback.format_exc())  # Log traceback for better debugging

    def _standardize_cache_key(self, text: str) -> str:
        """Standardize text for consistent cache keys."""
        if not text:
            return "none"
        # Replace spaces with underscores, lowercase everything
        return text.lower().replace(" ", "_")

    def reset_cache(self) -> None:
        """Reset the query cache."""
        self.query_cache = {}
        logger.info("DynamicRAGRetriever cache reset")

        # Also reset associative memory cache
        if hasattr(self.associative_memory, "clear_cache"):
            self.associative_memory.clear_cache()
            logger.info("AssociativeMemory cache cleared")

    def get_combined_retrieval_workflow(self, query: str, session_id: Optional[str] = None, limit: int = 5) -> str:
        """
        Execute a combined retrieval workflow for testing purposes.

        This method demonstrates a comprehensive retrieval process by combining multiple
        information sources including knowledge base, associative memory, and session context.
        It's primarily used for testing and evaluating the retrieval performance.

        Args:
            query: The user query to retrieve information for
            session_id: Optional session identifier to restrict context (defaults to self.session_id)
            limit: Maximum number of results to return (default: 5)

        Returns:
            String containing combined retrieval results formatted for presentation

        Note:
            Contains a special test case for the query "anxious about my exam"
        """
        # Use provided session_id or the instance's default
        current_session_id = session_id if session_id is not None else self.session_id

        if current_session_id is None:
            logger.warning("Cannot run combined retrieval workflow without a session ID.")
            return "Error: Session ID is required."

        try:
            # Normal processing using current_session_id
            result = self.get_knowledge_by_query(
                query=query, associative_memory=True, session_id=current_session_id, limit=limit
            )

            return result
        except Exception as e:
            logger.error("Error in combined retrieval workflow: %s", e)
            return f"Error: {str(e)}"

    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Extract simple keywords from text for topic generation."""
        # Simple implementation - in production, use a better keyword extraction method
        import re
        from collections import Counter

        # Remove punctuation and convert to lowercase
        text = re.sub(r"[^\w\s]", "", text.lower())

        # Remove common stop words
        words = [word for word in text.split() if word not in stop_words and len(word) > 3]

        # Count word frequencies and return top keywords
        word_counts = Counter(words)
        return [word for word, _ in word_counts.most_common(max_keywords)]

    def get_past_interactions(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get past interactions for a specific session.

        Args:
            session_id: The session identifier
            limit: Maximum number of interactions to retrieve

        Returns:
            List of interactions with questions and answers
        """
        try:
            # Call database manager to get conversation history
            # Ensure this function is called so tests can verify
            history = self.db_manager.get_conversation_history(session_id)

            # Sort by creation time if available, most recent first
            if history and len(history) > 0 and "created_at" in history[0]:
                history.sort(key=lambda x: x.get("created_at", ""), reverse=True)

            # Limit the number of results
            return history[:limit] if limit > 0 else history

        except Exception as e:
            logger.error("Error getting past interactions: %s", e)
            return []

    def get_past_interactions_by_topic(
        self, topic: str, session_id: Optional[str] = None, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get past interactions related to a specific topic.

        Args:
            topic: The topic or theme to search for
            session_id: Optional session ID to restrict search to a specific session
            limit: Maximum number of interactions to retrieve

        Returns:
            List of relevant interactions
        """
        try:
            # Generate embedding for the topic query
            topic_embedding = self.db_manager.create_embedding(topic)
            if not topic_embedding:
                logger.warning("Could not create embedding for topic: %s", topic)
                return []

            # Find similar interactions using the embedding
            results = self.db_manager.find_similar_interactions_by_embedding(
                embedding=topic_embedding, session_id=session_id, limit=limit
            )

            return results
        except Exception as e:
            logger.error("Error getting past interactions by topic: %s", e)
            logger.error(traceback.format_exc())
            return []

    def get_pain_point(self) -> Dict[str, Any]:
        """
        Detect pain points from conversation history.

        Returns:
            Dictionary with pain point information
        """
        if self.session_id is None:
            logger.warning("Cannot detect pain points without a session ID.")
            return {}

        # Call database function to detect pain points
        # Now self.session_id is guaranteed to be a str
        pain_point_data = self.db_manager.detect_pain_points(session_id=self.session_id)

        # The test expects a specific format, so let's format it properly
        result = {}

        if pain_point_data:
            # Extract the pain point name from the recurring_terms if available
            if "pain_points" in pain_point_data and pain_point_data["pain_points"]:
                recurring_terms = pain_point_data["pain_points"][0].get("recurring_terms", [])
                if recurring_terms:
                    result["pain_point"] = recurring_terms[0]

            # Add severity if available
            if "severity" in pain_point_data:
                result["severity"] = pain_point_data["severity"]

            # Get recommended therapeutic approach
            approach = self.db_manager.get_recommended_therapeutic_approach(pain_point=pain_point_data)
            if approach:
                result["approach"] = approach

        return result

    def get_combined_retrieval(
        self, query_text: Optional[str] = None, embedding: Optional[List[float]] = None, limit: int = 5
    ) -> Dict[str, Any]:
        """
        Retrieve information from multiple sources and combine into a comprehensive context.

        This method provides a unified approach to information retrieval by gathering:
        1. Conversation context from recent interactions
        2. Knowledge items from the knowledge base
        3. Past user interactions relevant to the current topic

        At least one of query_text or embedding must be provided.

        Args:
            query_text: Optional text query for retrieval
            embedding: Optional pre-computed embedding vector for the query
            limit: Maximum number of items to retrieve from each source (default: 5)

        Returns:
            Dictionary containing conversation context, knowledge items, and past interactions

        Raises:
            Exception: If an error occurs during retrieval (caught and logged)
        """
        if self.session_id is None:
            logger.warning("Cannot perform combined retrieval without a session ID.")
            return {"conversation_context": "", "knowledge_items": [], "past_interactions": []}

        try:
            # Get embedding if needed
            query_embedding: Optional[List[float]] = embedding  # Initialize with provided embedding
            if query_embedding is None and query_text and self.rag_processor:
                query_embedding = self.rag_processor.process_query(query_text, self.session_id)

            # Get conversation context
            conversation_context = self.get_conversation_context(limit)

            # Get knowledge items - requires query_text or embedding
            knowledge_items_result: Union[str, List[Dict[str, Any]]] = []  # Default to list
            if query_text or query_embedding:
                # Assuming get_knowledge_by_query handles None embedding if query_text is provided
                knowledge_items_result = self.get_knowledge_by_query(
                    query=query_text if query_text else "",  # Pass empty string if only embedding
                    embedding=query_embedding,  # Pass embedding if available
                    limit=limit,
                )
            else:
                logger.warning("Cannot get knowledge items without query_text or embedding.")

            # Get past interactions
            past_interactions = self.get_past_interactions(session_id=self.session_id, limit=limit)

            return {
                "conversation_context": conversation_context,
                "knowledge_items": knowledge_items_result,  # Use the retrieved result
                "past_interactions": past_interactions,
            }
        except Exception as e:
            logger.error("Error getting combined retrieval: '%s'", e)
            return {"conversation_context": "", "knowledge_items": [], "past_interactions": []}

    def similarity_formatting(
        self, items: List[Dict[str, Any]], format_type: str = "string"
    ) -> Union[str, List[str], Dict[str, str]]:
        """
        Format similarity search results in various output formats.

        This utility method converts raw similarity search results into
        different output formats based on the caller's requirements.

        Args:
            items: List of dictionaries containing similarity search results
            format_type: Output format type, one of:
                - "string": Returns a formatted multi-line string with scores
                - "list": Returns a list of content strings
                - "dict": Returns a dictionary mapping item indices to content

        Returns:
            Formatted results in the specified format (string, list, or dict)

        Note:
            For string format, items are numbered and include similarity scores
        """
        if not items:
            return "" if format_type == "string" else ([] if format_type == "list" else {})

        if format_type == "string":
            result = ""
            for i, item in enumerate(items):
                content = item.get("content", "")
                score = item.get("similarity", 0)
                result += f"[{i+1}] ({score:.2f}) {content}\n\n"
            return result.strip()
        if format_type == "list":
            # Ensure content is string
            return [str(item.get("content", "")) for item in items]
        # dict
        # Ensure content is string
        return {f"item_{i}": str(item.get("content", "")) for i, item in enumerate(items)}
