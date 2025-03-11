import logging
from typing import Dict, List, Any
from psy_supabase.core.database import DatabaseManager

logger = logging.getLogger(__name__)

class DynamicRAGRetriever:
    """
    Handles on-demand retrieval from database during model generation.
    This reduces the context size by only fetching information when needed.
    """
    
    def __init__(self, db_manager: DatabaseManager, session_id: str, allow_dynamic_queries: bool = True):
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
        
    # For knowledge_by_query - use dynamic limits based on query complexity
    def get_knowledge_by_query(self, query: str, limit: int = None) -> str:
        # Calculate appropriate limit based on query complexity
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
            
        # Check cache first
        cache_key = f"knowledge_{query}_{limit}"
        if cache_key in self.query_cache:
            logger.info(f"Using cached knowledge for query: {query}")
            return self.query_cache[cache_key]
            
        try:
            # Convert query to embedding
            embedding = self.db_manager.create_embedding(query)
            if not embedding:
                return "Unable to create embedding for query."
                
            # Use schema-specific knowledge base search for better performance
            # This uses the specific schema's knowledge_base table with specialized pgvector indexes
            results = self.db_manager.find_similar_documents_via_rpc(
                session_id=self.session_id,
                embedding=embedding,
                similarity_threshold=0.65,
                limit=limit
            )
            
            if not results:
                return f"No knowledge found for: {query}"
                
            # Format the results
            formatted_results = ""
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
                            
                    formatted_results += f"[{i+1}] {content}{meta_str} [relevance: {similarity:.2f}]\n\n"
                else:
                    content = str(doc)
                    formatted_results += f"[{i+1}] {content}\n\n"
                    
            # Cache the results
            self.query_cache[cache_key] = formatted_results
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error retrieving knowledge: {e}")
            return f"Error retrieving knowledge: {str(e)}"
    
    def get_past_interactions(self, topic: str = None, limit: int = 3) -> str:
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
        cache_key = f"interactions_{topic}_{limit}"
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
    
    def get_pain_point(self) -> Dict:
        """
        Retrieve current pain point information for the user.
        
        Returns:
            Dict: Pain point information
        """
        if not self.allow_dynamic_queries:
            return {"pain_point": "Dynamic querying is disabled."}
            
        cache_key = "pain_point"
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
            
        try:
            # First, create an embedding of the recent conversation to analyze
            # Get the most recent user question from the conversation history
            recent_history = self.db_manager.get_conversation_history(self.session_id)
            
            if not recent_history or len(recent_history) == 0:
                return {"pain_point": None}
                
            # Get the most recent question
            recent_question = recent_history[-1].get('questionText', '') if len(recent_history) > 0 else ""
            
            if not recent_question:
                return {"pain_point": None}
                
            # Create embedding for the question
            question_embedding = self.db_manager.create_embedding(recent_question)
            
            if not question_embedding:
                return {"pain_point": None}
                
            # Now properly call identify_potential_pain_points with all required arguments
            pain_points = self.db_manager.identify_potential_pain_points(
                question_text=recent_question,
                question_embedding=question_embedding,
                session_id=self.session_id
            )
            
            if pain_points and len(pain_points) > 0:
                self.query_cache[cache_key] = pain_points
                return pain_points
            else:
                return {"pain_point": None}
                
        except Exception as e:
            logger.error(f"Error retrieving pain point: {e}")
            return {"pain_point": f"Error: {str(e)}"}
            
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
            
        cache_key = f"concepts_{concept}_{limit}"
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
            
        try:
            # Create embedding for the concept
            embedding = self.db_manager.create_embedding(concept)
            if not embedding:
                return f"No related concepts found for {concept}."
                
            # Find conceptually similar knowledge entries using pgvector
            schema_name = self._sanitize_schema_name(self.session_id)
            
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
            
    def _sanitize_schema_name(self, schema_name: str) -> str:
        """Sanitize schema name by replacing dashes with underscores."""
        if schema_name:
            return schema_name.replace('-', '_')
        return 'default'

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
            
        cache_key = f"emotion_{text[:50]}"  # Use first 50 chars as key to avoid huge cache keys
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
            
        cache_key = "topics_analysis"
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
            
        try:
            # Call the pgvector-powered topic analysis function
            topics = self.db_manager.analyze_conversation_topics(
                session_id=self.session_id, 
                min_count=1
            )
            
            if not topics:
                return [{"topic": "No significant topics identified", "frequency": 0}]
                
            # Cache the results
            self.query_cache[cache_key] = topics
            return topics
            
        except Exception as e:
            logger.error(f"Error analyzing topics: {e}")
            return [{"topic": f"Error: {str(e)}", "frequency": 0}]
