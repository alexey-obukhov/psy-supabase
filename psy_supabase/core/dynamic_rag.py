from school_logging.log import ColoredLogger
from typing import Dict, List, Any, Optional, TYPE_CHECKING

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

    # For knowledge_by_query - use dynamic limits based on query complexity
    def get_knowledge_by_query(self, query: str, limit: Optional[int] = None) -> str:
        """
        Get knowledge based on a query by retrieving similar documents from the database.

        Args:
            query: The search query (topic) to look for
            limit: Maximum number of documents to retrieve. If None, dynamically determined.

        Returns:
            str: Combined content from retrieved documents
        """
        try:
            # Calculate appropriate limit based on query complexity if not specified
            if limit is None:
                # Simple queries need fewer results
                if len(query.split()) <= 3:
                    limit = 2
                # Complex queries might need more comprehensive information
                elif len(query.split()) >= 8:
                    limit = 5
                # Default for medium complexity
                else:
                    limit = 3

            if not self.allow_dynamic_queries:
                return "Dynamic querying is disabled."

            # Check cache first - use clean query for cache key
            cache_key = f"knowledge_{self._standardize_cache_key(query)}_{limit}"
            if cache_key in self.query_cache:
                logger.info(f"Using cached knowledge for query: {query}")
                return self.query_cache[cache_key]

            # Get embeddings for the query
            embedding = self.db_manager.create_embedding(query)
            if not embedding:
                logger.error(f"Failed to generate embedding for knowledge query: {query}")
                return ""

            # Use schema-specific knowledge base search for better performance
            # This uses the specific schema's knowledge_base table with specialized pgvector indexes
            results = self.db_manager.find_similar_documents_via_rpc(
                session_id=self.session_id,
                embedding=embedding,
                similarity_threshold=0.65,
                limit=limit
            )

            if not results:
                # Fallback to regular search if RPC fails
                results = self.db_manager.find_similar_documents_by_embedding(
                    embedding=embedding,
                    threshold=0.5,
                    limit=limit
                )

            if not results:
                logger.warning(f"No similar documents found for query: {query}")
                return f"No knowledge found for: {query}"

            # Combine the content - choose the format based on the first result's structure
            if isinstance(results[0], dict) and "content" in results[0]:
                # Simple format - just combine content fields
                combined_content = "\n\n".join([doc.get("content", "") for doc in results])
            else:
                # Format the results with more details
                combined_content = ""
                for i, doc in enumerate(results):
                    # Handle both dictionary and string formats
                    if isinstance(doc, dict):
                        content = doc.get('content', '')
                        metadata = doc.get('metadata', {})
                        similarity = doc.get('similarity', 0)

                        # Include metadata if available
                        meta_str = ""
                        if metadata and isinstance(metadata, dict):
                            if 'source' in metadata:
                                meta_str = f" (Source: {metadata['source']})"
                            elif 'category' in metadata:
                                meta_str = f" (Category: {metadata['category']})"

                        combined_content += f"{content}{meta_str}\n\n"
                    else:
                        content = str(doc)
                        combined_content += f"{content}\n\n"

            # Cache the result
            self.query_cache[cache_key] = combined_content

            logger.info(f"Retrieved knowledge for '{query}': {len(combined_content)} chars from {len(results)} docs")
            return combined_content

        except Exception as e:
            logger.error(f"Error retrieving knowledge for query '{query}': {e}")
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
                    question = interaction.get('question', interaction.get('questionText', ''))
                    answer = interaction.get('answer', interaction.get('answerText', ''))
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
            # Create embedding for the concept
            embedding = self.db_manager.create_embedding(concept)
            if not embedding:
                return f"No related concepts found for {concept}."

            # Find conceptually similar knowledge entries using pgvector
            schema_name = self.db_manager.schema_name

            # Use an SQL query that specifically targets psychological concepts
            query = f"""
            WITH concept_embedding AS (
                SELECT '{str(embedding).replace(' ', '')}'::vector as embedding
            )
            SELECT
                content,
                1 - (embedding <=> (SELECT embedding FROM concept_embedding)) as similarity
            FROM
                {schema_name}.knowledge_base
            WHERE
                1 - (embedding <=> (SELECT embedding FROM concept_embedding)) > 0.7
                AND (
                    content ILIKE '%concept%' OR
                    content ILIKE '%therapy%' OR
                    content ILIKE '%psychology%' OR
                    content ILIKE '%mental health%'
                )
            ORDER BY
                similarity DESC
            LIMIT {limit};
            """

            response = self.db_manager.supabase.rpc('sql', {'command': query}).execute()

            if not response.data or len(response.data) == 0:
                return f"No related concepts found for {concept}."

            # Format the results
            formatted_results = f"Related concepts to '{concept}':\n\n"
            for i, item in enumerate(response.data):
                if isinstance(item, dict):
                    content = item.get('content', '')
                    similarity = item.get('similarity', 0)
                    formatted_results += f"[{i+1}] {content} [relevance: {similarity:.2f}]\n\n"
                elif isinstance(item, str):
                    parts = item.split(',', 1)
                    if len(parts) >= 2:
                        content = parts[0]
                        formatted_results += f"[{i+1}] {content}\n\n"

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
            # Call the RPC function directly - consistent with your codebase
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
