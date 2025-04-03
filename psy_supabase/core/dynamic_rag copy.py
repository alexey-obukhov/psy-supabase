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
- school_logging.log.ColoredLogger: Provides enhanced logging capabilities for debugging and monitoring.

Usage:
    db_manager = DatabaseManager(...)
    retriever = DynamicRAGRetriever(db_manager, session_id="user_session_123")
    knowledge = retriever.get_knowledge_by_query("anxiety management")
    past_interactions = retriever.get_past_interactions(topic="anxiety")
    emotion_analysis = retriever.analyze_emotion("I'm feeling very stressed lately.")
    pain_point = retriever.get_pain_point()
"""
import json
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from school_logging.log import ColoredLogger

# Set up logging
logger = ColoredLogger(__name__)

# Use TYPE_CHECKING for type hints without runtime dependency
if TYPE_CHECKING:
    from psy_supabase.core.database import DatabaseManager

class DynamicRAGRetriever:
    """
    Handles on-demand retrieval from database during model generation.
    This reduces the context size by only fetching information when needed.
    """

    def __init__(self, db_manager: 'DatabaseManager', session_id: str, allow_dynamic_queries: bool = True):
        """
        Initialize the dynamic retriever with database connection and session info.

        Args:
            db_manager: Instance of DatabaseManager for database access
            session_id: Current user session ID
            allow_dynamic_queries: Whether to allow dynamic queries (can be disabled for testing)
        """
        self.db_manager = db_manager
        self.session_id = session_id
        self.allow_dynamic_queries = allow_dynamic_queries
        self.query_cache = {}  # Cache to avoid repeated identical queries

    def _standardize_cache_key(self, text):
        """Standardize text for consistent cache keys."""
        if not text:
            return "none"
        # Replace spaces with underscores, lowercase everything
        return text.lower().replace(' ', '_')

    def get_knowledge_by_query(self, query: str, associative_memory: bool = False,
                              min_similarity: float = 0.1, **kwargs) -> str:
        """
        Retrieve knowledge based on query with similarity filtering.

        Args:
            query: The query text
            associative_memory: Whether to use associative memory
            min_similarity: Minimum similarity threshold to include results (0.0 to 1.0)

        Returns:
            A formatted string with knowledge matching the query
        """
        try:
            # Input validation - handle empty or invalid queries
            if not query or not isinstance(query, str):
                return "No valid query provided."

            # Special character handling - if query is mostly special characters, handle gracefully
            import re
            alphanumeric_count = len(re.sub(r'[^a-zA-Z0-9]', '', query))
            if len(query) > 0 and alphanumeric_count / len(query) < 0.5:
                logger.warning(f"Query contains mostly special characters: {query}")
                return "No relevant information found."

            # Check cache
            cache_key = f"{query}_{associative_memory}_{min_similarity}"
            if cache_key in self.query_cache:
                logger.info(f"Cache hit for query: {query[:30]}...")
                return self.query_cache[cache_key]

            # Create embedding for query
            query_embedding = self.db_manager.create_embedding(query)
            if not query_embedding:
                return "Unable to process query. Please try a different question."

            # Get similar documents from DB
            try:
                docs = self.db_manager.find_similar_documents_via_rpc(
                    embedding=query_embedding,
                    session_id=self.session_id,
                    min_similarity=min_similarity,  # Pass the threshold parameter
                    **kwargs
                )
            except Exception as db_error:
                logger.error(f"Database error retrieving documents: {db_error}")
                return f"Error retrieving knowledge: {str(db_error)}"

            if not docs:
                return "No relevant information found."

            # Process initial results
            result_parts = []
            topics_to_explore = set()

            for doc in docs:
                content = doc.get('content', '')
                similarity = doc.get('similarity', 0)

                # Apply similarity filter - only include results above threshold
                if similarity >= min_similarity:
                    # Format with similarity score
                    result_parts.append(f"{content} (Relevance: {similarity:.2f})")

                    # Collect topics for associative memory
                    if associative_memory and 'metadata' in doc:
                        metadata = doc.get('metadata', {})
                        if isinstance(metadata, str):
                            try:
                                metadata = json.loads(metadata)
                            except:
                                metadata = {}

                        # Extract related topics
                        related_topics = metadata.get('related_topics', [])
                        if related_topics:
                            if isinstance(related_topics, str):
                                topics_list = [t.strip() for t in related_topics.split(',')]
                                topics_to_explore.update(topics_list)
                            elif isinstance(related_topics, list):
                                topics_to_explore.update(related_topics)

            # If using associative memory, look up related topics
            if associative_memory and topics_to_explore:
                for topic in topics_to_explore:
                    # Get documents related to this topic
                    try:
                        topic_embedding = self.db_manager.create_embedding(topic)
                        if not topic_embedding:
                            continue

                        topic_docs = self.db_manager.find_similar_documents_via_rpc(
                            embedding=topic_embedding,
                            session_id=self.session_id,
                            limit=3,
                            min_similarity=min_similarity  # Apply same threshold to associated docs
                        )
                    except Exception as topic_error:
                        logger.error(f"Error retrieving topic documents for {topic}: {topic_error}")
                        continue

                    # Add associated results with proper formatting, respecting threshold
                    for doc in topic_docs:
                        content = doc.get('content', '')
                        similarity = doc.get('similarity', 0)

                        # Only include if above threshold
                        if similarity >= min_similarity:
                            result_parts.append(f"{content} (Associated Memory, Relevance: {similarity:.2f})")

            # Combine results
            combined_results = "\n".join(result_parts)

            # Cache the result
            self.query_cache[cache_key] = combined_results

            return combined_results

        except Exception as e:
            logger.error(f"Error in get_knowledge_by_query: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"Error retrieving knowledge: {str(e)}"

    def get_past_interactions(self, topic: Optional[str] = None, limit: int = 3) -> str:
        """
        Dynamically retrieve past conversation interactions.

        Args:
            topic: Optional topic to filter by
            limit: Maximum number of interactions to return

        Returns:
            str: Formatted interaction history
        """
        if not self.allow_dynamic_queries:
            return "Dynamic querying is disabled."

        # Check cache first
        cache_key = f"interactions_{self._standardize_cache_key(topic)}_{limit}"
        if cache_key in self.query_cache:
            logger.info(f"Using cached interactions for topic: {topic}")
            return self.query_cache[cache_key]

        try:
            # Query recent interactions, optionally filtered by topic
            if topic:
                # Create embedding for the topic query
                try:
                    # Generate embedding using the DatabaseManager's method
                    embedding = self.db_manager.create_embedding(topic)

                    if embedding:
                        interactions = self.db_manager.find_similar_interactions_by_embedding(
                            embedding=embedding,
                            session_id=self.session_id,
                            limit=limit,
                            threshold=0.65
                        )
                    else:
                        # If embedding creation fails, fall back to most recent
                        interactions = self.db_manager.get_conversation_history(self.session_id)
                        if interactions:
                            interactions = interactions[-limit:] if len(interactions) > limit else interactions
                except Exception as e:
                    logger.error(f"Error creating embedding for topic: {e}")
                    # Fall back to most recent interactions
                    interactions = self.db_manager.get_conversation_history(self.session_id)
                    if interactions:
                        interactions = interactions[-limit:] if len(interactions) > limit else interactions
            else:
                # Get recent interactions
                interactions = self.db_manager.get_conversation_history(self.session_id)
                if interactions:
                    interactions = interactions[-limit:] if len(interactions) > limit else interactions

            if not interactions:
                return f"No past interactions found{' related to ' + topic if topic else ''}."

            # Format the interactions
            formatted_interactions = ""
            for interaction in interactions:
                if isinstance(interaction, dict):
                    question = interaction.get('question', interaction.get('question', ''))
                    answer = interaction.get('answer', interaction.get('answer', ''))
                    if question and answer:
                        formatted_interactions += f"User: {question}\nAssistant: {answer}\n\n"

            # Cache the results
            self.query_cache[cache_key] = formatted_interactions
            return formatted_interactions

        except Exception as e:
            logger.error(f"Error retrieving past interactions: {e}")
            return f"Error retrieving past interactions: {str(e)}"

    def reset_cache(self):
        """Clear the query cache."""
        self.query_cache = {}

    def get_related_concepts(self, concept: str, limit: int = 3) -> str:
        """
        Find related psychological concepts using pgvector similarity.

        Args:
            concept: The concept to find relationships for
            limit: Maximum number of related concepts to return

        Returns:
            str: Formatted string of related concepts
        """
        if not self.allow_dynamic_queries:
            return "Dynamic querying is disabled."

        cache_key = f"concepts_{self._standardize_cache_key(concept)}_{limit}"
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]

        try:
            # Generate embedding for proper vector search
            concept_embedding = self.db_manager.create_embedding(concept)
            if not concept_embedding:
                logger.error("Failed to create embedding for concept")
                return f"Error processing concept '{concept}'."

            # Use find_similar_documents with the embedding
            related_docs = self.db_manager.find_similar_documents(
                query_text=concept,  # For logging purposes
                embedding=concept_embedding,  # The actual embedding for search
                limit=limit*2,  # Get more initially to filter
                min_similarity=0.7  # Same threshold as the original SQL
            )

            if not related_docs:
                return f"No related concepts found for {concept}."

            # Filter for psychology-related content
            psychology_keywords = [
                'concept', 'therapy', 'psychology', 'mental health', 'counseling', 'psychiatry',
                'behavioral', 'cognitive', 'emotional', 'disorder', 'treatment', 'psychotherapy',
                'therapeutic', 'wellbeing', 'psychological', 'mindfulness', 'trauma', 'anxiety',
                'depression', 'stress', 'coping', 'mechanism', 'intervention', 'development',
                'psychoanalysis', 'psychodynamic', 'psychosocial', 'neuropsychology'
            ]
            filtered_docs = []

            for doc in related_docs:
                content = doc.get('content', '')
                if isinstance(content, str):  # Ensure content is a string before checking
                    if any(keyword in content.lower() for keyword in psychology_keywords):
                        filtered_docs.append(doc)

            # If no documents match the psychology filter, use the original results
            if not filtered_docs:
                filtered_docs = related_docs

            # Take only the top results up to the limit
            filtered_docs = filtered_docs[:limit]

            # Format the results
            formatted_results = f"Related concepts to '{concept}':\n\n"
            for i, doc in enumerate(filtered_docs):
                content = doc.get('content', '')
                similarity = doc.get('similarity', 0)
                formatted_results += f"[{i+1}] {content} [relevance: {similarity:.2f}]\n\n"

            # Cache the results
            self.query_cache[cache_key] = formatted_results
            return formatted_results

        except Exception as e:
            logger.error(f"Error finding related concepts: {e}")
            return f"Error finding related concepts: {str(e)}"

    def analyze_emotion(self, text: str) -> Dict[str, Any]:
        """
        Dynamically analyze the emotion in a piece of text.

        Args:
            text: The text to analyze for emotional content

        Returns:
            Dict: Dictionary containing emotional analysis
        """
        if not self.allow_dynamic_queries:
            return {"analysis": "Dynamic querying is disabled."}

        cache_key = f"emotion_{self._standardize_cache_key(text[:50])}"
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]

        try:
            # Create embedding for the text
            embedding = self.db_manager.create_embedding(text)
            if not embedding:
                return {"error": "Unable to create embedding for emotion analysis."}

            # Use predefined emotional anchors to analyze where this text falls
            emotions = self.db_manager.analyze_text_emotional_spectrum(
                text_embedding=embedding,
                session_id=self.session_id
            )

            if not emotions:
                return {"primary_emotion": "neutral", "intensity": 0.0, "spectrum": []}

            # Cache the results
            self.query_cache[cache_key] = emotions
            return emotions

        except Exception as e:
            logger.error(f"Error analyzing emotion: {e}")
            return {"error": f"Error analyzing emotion: {str(e)}"}

    def analyze_topics(self) -> List[Dict]:
        """
        Analyze common topics in user interactions using pgvector clustering.

        Returns:
            List[Dict]: Top topics with their frequency
        """
        if not self.allow_dynamic_queries:
            return [{"topic": "Dynamic querying is disabled.", "frequency": 0}]

        cache_key = f"topics_analysis_{self.session_id}"
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]

        try:
            # Call the RPC function directly
            response = self.db_manager.supabase.rpc(
                'analyze_conversation_topics',
                {
                    'p_schema_name': self.db_manager.schema_name,
                    'p_min_count': 1
                }
            ).execute()

            if response.data:
                topics = []
                for item in response.data:
                    topics.append({
                        "topic": item.get('topic', 'Unknown topic'),
                        "frequency": item.get('frequency', 0)
                    })

                # Cache the results
                self.query_cache[cache_key] = topics
                return topics
            else:
                return [{"topic": "No significant topics identified", "frequency": 0}]

        except Exception as e:
            logger.error(f"Error analyzing topics: {e}")
            return [{"topic": f"Error: {str(e)}", "frequency": 0}]

    def get_pain_point(self):
        """
        Check for recurring patterns in user questions and return detected pain points.

        Returns:
            Dict: Pain point information if detected, otherwise None
        """
        try:
            if not self.session_id:
                logger.warning("Cannot check for pain points without session_id")
                return None

            # Check cache first
            cache_key = f"pain_point_{self._standardize_cache_key(self.session_id)}"
            if cache_key in self.query_cache:
                return self.query_cache[cache_key]

            # Use the new method from database.py
            pain_points = self.db_manager.detect_pain_points(
                self.session_id,
                threshold=0.7,
                min_occurrences=2
            )

            # If no pain points detected, return None
            if not pain_points or not pain_points.get('pain_points'):
                return None

            # Get the most significant pain point (first detected)
            primary_pain_point = pain_points['pain_points'][0]

            # Get recommended therapeutic approach
            approach = self.db_manager.get_recommended_therapeutic_approach(primary_pain_point)

            # Combine pain point info with approach
            result = {
                'pain_point': primary_pain_point.get('recurring_terms', ['unclear theme'])[0],
                'recurring_terms': primary_pain_point.get('recurring_terms', []),
                'count': primary_pain_point.get('count', 0),
                'severity': pain_points['severity'],
                'first_detected_at': pain_points['first_detected_at'],
                'approach': approach
            }

            # Cache the result
            self.query_cache[cache_key] = result

            logger.info(f"Detected pain point: {result['pain_point']} (severity: {result['severity']})")
            return result
        except Exception as e:
            logger.error(f"Error getting pain point: {e}")
            return None
