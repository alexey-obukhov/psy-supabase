import logging
from supabase import create_client
from typing import List, Dict, Any, Optional
import json
import re
import traceback
from datetime import datetime

from psy_supabase.utilities.text_utils import clean_text
from psy_supabase.core.model_manager import get_embedding_provider

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self, supabase_url: str, supabase_key: str, user_id: str):
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.user_id = user_id
        
        # Special case for default schema
        if user_id == "default":
            self.schema_name = "default"
        else:
            self.schema_name = self._sanitize_schema_name(user_id)
            
        self.supabase = create_client(self.supabase_url, self.supabase_key)
        
        # Create default schema if needed
        if user_id == "default":
            self.create_default_schema_sync()

    def create_user_schema(self):
        """Creates a user-specific schema and tables if they don't exist."""
        try:
            # Call the stored procedure to create the schema and tables
            response = self.supabase.rpc('create_user_schema_and_tables', {'schema_name': self.schema_name}).execute()
            
            # Check for errors
            if response.data is None:
                logger.error("Schema creation failed for user %s", self.user_id)
                return False
                
            # If response.data is False, it means the function failed
            if response.data is False:
                logger.error(f"Schema creation failed for user {self.user_id}")
                return False
                
            # If we get here, the function returned TRUE (success)
            logger.info(f"Schema '{self.schema_name}' created successfully.")
            return True

        except Exception as e:
            logger.error(f"Error creating schema for user {self.user_id}: {e}")
            return False

    def create_default_schema_sync(self):
        """Creates a default schema if it doesn't exist."""
        try:
            response = self.supabase.rpc('create_default_schema_and_tables').execute()
            logger.info(f"Default schema and tables created successfully.")
            return True
        except Exception as e:
            logger.error(f"Error creating default schema: {e}")
            logger.error(traceback.format_exc())
            return False

    def get_conversation_history(self, session_id: str):
        """Retrieves conversation history from the user's session."""
        try:
            # Sanitize session ID for safety
            sanitized_session_id = self._sanitize_schema_name(session_id)
            
            # Call the stored procedure to retrieve conversation history
            response = self.supabase.rpc('get_conversation_history', {'schema_name': sanitized_session_id}).execute()

            # Check if the response contains data
            if response.data:
                # Transform the data to have the expected field names
                transformed_data = []
                for item in response.data:
                    transformed_item = {
                        'interactionID': item.get('interactionid'),
                        'questionText': self.clean_db_text(item.get('question')),  # in db it is question
                        'answerText': self.clean_db_text(item.get('answer')),      # in db it is answer
                        'context': self.clean_db_text(item.get('context')),
                        'metadata': item.get('metadata'),
                        'created_at': item.get('created_at')
                    }
                    transformed_data.append(transformed_item)
                return transformed_data
            else:
                logger.warning(f"No conversation history found for session: {session_id}")
                return []
        except Exception as e:
            # Log the error with additional context
            logger.error(f"Error fetching conversation history for session {session_id}: {e}")
            return []

    def add_interaction(self, data_point, session_id: str = None):
        """Adds an interaction to the database."""
        try:
            # Use schema_name if session_id is not provided
            schema_name = self._sanitize_schema_name(session_id) if session_id else self.schema_name
            
            # Make sure schema exists first
            self.create_user_schema_sync()
            
            # Handle metadata properly
            if isinstance(data_point.get('metadata'), dict):
                metadata = json.dumps(data_point.get('metadata'))
            elif isinstance(data_point.get('metadata'), str):
                metadata = data_point.get('metadata')
            else:
                metadata = '{}'
                
            # Clean and escape the values
            context = clean_text(data_point.get('context', ''))
            question = clean_text(data_point.get('question', ''))
            answer = clean_text(data_point.get('answer', ''))

            # Log what we're trying to do
            logger.debug(f"Adding interaction to schema: {schema_name}")
            
            try:
                # Try using RPC function first - this is more reliable
                response = self.supabase.rpc('add_interaction', {
                    'p_schema_name': schema_name,
                    'p_context': context,
                    'p_question': question,
                    'p_answer': answer,
                    'p_metadata': metadata
                }).execute()
                
                if response.data is None:
                    logger.error("RPC add_interaction failed")
                    raise Exception("RPC call failed")
                    
                logger.info(f"Successfully added interaction via RPC")
                return True
                
            except Exception as inner_e:
                logger.error(f"Error in RPC call: {str(inner_e)}")
                # Fall back to direct table insert
                try:
                    table_name = f"{schema_name}.interactions"
                    response = self.supabase.table(table_name).insert({
                        'context': context,
                        'question': question,
                        'answer': answer,
                        'metadata': metadata
                    }).execute()
                    
                    # Check for error using the new pattern
                    if response.data is None:
                        logger.error("Table insert failed - no data in response")
                        return False
                        
                    logger.info(f"Successfully added interaction via table insert")
                    return True
                except Exception as table_e:
                    logger.error(f"Table insert also failed: {str(table_e)}")
                    raise table_e
                
        except Exception as e:
            logger.error(f"Error adding interaction: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def add_interaction_rpc(self, data_point, session_id: str = None):
        """Adds an interaction to the database using an RPC call."""
        try:
            # Use schema_name if session_id is not provided
            schema_name = self._sanitize_schema_name(session_id) if session_id else self.schema_name
            
            # Handle metadata properly
            if isinstance(data_point.get('metadata'), dict):
                metadata = json.dumps(data_point.get('metadata'))
            elif isinstance(data_point.get('metadata'), str):
                metadata = data_point.get('metadata')  # Already a JSON string
            else:
                metadata = '{}'  # Default empty JSON
                
            # Clean and escape the values
            context = clean_text(data_point['context'])
            question = clean_text(data_point['question'])
            answer = clean_text(data_point['answer'])
            
            # Call the RPC function to add the interaction
            response = self.supabase.rpc('add_interaction', {
                'p_schema_name': schema_name,
                'p_context': context,
                'p_question': question,
                'p_answer': answer,
                'p_metadata': metadata
            }).execute()
            
            if response.data is None:
                logger.error("Error adding interaction via RPC")
                return False
                
            logger.info(f"Interaction added successfully with ID: {response.data}")
            return True
        except Exception as e:
            logger.error(f"Exception adding interaction: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def create_user_schema_sync(self):
        """Creates a user-specific schema and tables if they don't exist (synchronous version)."""
        try:
            # Call the stored procedure to create the schema and tables
            response = self.supabase.rpc('create_user_schema_and_tables', {'schema_name': self.schema_name}).execute()
            
            # Check the response using the new pattern
            if response.data is None:
                logger.error(f"Schema creation failed for user {self.user_id} - no data in response")
                return False
            elif response.data is False:
                logger.error(f"Schema creation function returned FALSE for user {self.user_id}")
                return False
            else:
                logger.info(f"Schema '{self.schema_name}' and tables created successfully.")
                return True

        except Exception as e:
            logger.error(f"Error creating schema for user {self.user_id}: {e}")
            logger.error(traceback.format_exc())
            return False

    def get_interaction_history(self, user_id: str):
        """ Get interaction history from the user's schema """
        schema_name = self._sanitize_schema_name(user_id)
        logger.info(f"Retrieving interaction history for user: {user_id} with schema {schema_name}")

        # Call SQL function to retrieve interaction history
        sql_query = f"SELECT * FROM get_interaction_history('{schema_name}')"
        response = self.supabase.rpc('sql', {'command': sql_query}).execute()

        if response.data is None:
            logger.error(f"Error retrieving interaction history for user {user_id}")
            return None
            
        history = response.model_dump_json()
        logger.info(f"Retrieved interaction history for user {user_id}: {history}")
        return history

    def ensure_user_schema_view(self, user_id: str):
        """ Ensure the view for the user schema exists in the public schema """
        schema_name = self._sanitize_schema_name(user_id)
        logger.info(f"Ensuring view exists for user: {user_id} with schema {schema_name}")

        # Call SQL function to ensure the view exists
        sql_query = f"SELECT ensure_user_schema_view('{schema_name}')"
        response = self.supabase.rpc('sql', {'command': sql_query}).execute()

        if response.data is None:
            logger.error(f"Error confirming view for user {user_id}")
            return False
            
        logger.info(f"View for user {user_id} confirmed.")
        return True

    def _sanitize_schema_name(self, user_id: str) -> str:
        """Sanitizes the user ID to be a valid PostgreSQL schema name (private method)."""
        if not user_id:
            return "default"
            
        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", user_id)
        if not (safe_name[0].isalpha() or safe_name[0] == '_'):
            safe_name = "_" + safe_name
        return safe_name[:63]

    def _get_table_name(self, table_name: str) -> str:
        """
        Returns the fully qualified table name with the correct schema.
        Uses the provided `user_id` or falls back to `self.user_id`.
        """
        schema = self._sanitize_schema_name(self.user_id)
        return f'"{schema}"."{table_name}"'

    def get_all_documents_and_embeddings(self, table_name: str = "knowledge_base") -> List[Dict]:
        """Retrieves all documents and their embeddings from the knowledge base."""
        try:
            # Use the dedicated function to get knowledge base documents
            response = self.supabase.rpc('get_knowledge_base_documents', 
                                        {'schema_name': self.schema_name}).execute()

            if response.data:
                # Process vector data - convert to list format for Python
                processed_results = []
                for doc in response.data:
                    # Handle vector embedding - convert to list
                    if 'embedding' in doc:
                        # Handle PostgreSQL vector format '[0.1,0.2,...]'
                        if isinstance(doc['embedding'], str):
                            # Parse the vector string into a list of floats
                            embedding_str = doc['embedding'].strip('[]')
                            if embedding_str:
                                embedding_list = [float(x) for x in embedding_str.split(',')]
                                doc['embedding'] = embedding_list
                        # If it's already in a usable format, keep it
                    
                    processed_results.append(doc)
                    
                logger.info(f"Retrieved {len(processed_results)} documents from knowledge base")
                return processed_results
            else:
                logger.warning("No documents found in knowledge base")
                return []
        except Exception as e:
            logger.error(f"Error retrieving documents and embeddings: {e}")
            traceback.print_exc()  
            return []

    def get_topic_interactions(self, session_id: str, topic: str, limit: int = 3):
        """Retrieves interactions related to a specific topic."""
        try:
            # Get all interactions from the session
            all_interactions = self.get_conversation_history(session_id)
            
            # Filter interactions by topic
            topic_interactions = []
            for interaction in all_interactions:
                # Extract metadata
                metadata = interaction.get('metadata', {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        continue
                    
                # Check if the interaction is related to the topic
                interaction_topic = metadata.get('topic', '').lower()
                if interaction_topic == topic.lower():
                    topic_interactions.append(interaction)
            
            # Return the most recent interactions up to the limit
            return topic_interactions[-limit:] if topic_interactions else []
        except Exception as e:
            logger.error(f"Error getting topic interactions: {e}")
            return []

    def get_high_quality_interactions(self, topic_filter=None, min_effectiveness=0.7, limit=100):
        """
        Retrieves high-quality interactions suitable for training data.
        
        Args:
            topic_filter: Optional topic to filter by
            min_effectiveness: Minimum effectiveness score threshold
            limit: Maximum number of examples to retrieve
            
        Returns:
            List of high-quality interactions
        """
        try:
            # Build the WHERE clause
            where_clauses = []
            if topic_filter:
                where_clauses.append(f"metadata->>'topic' = '{topic_filter}'")
                
            # Add effectiveness filter - parse the nested JSON structure
            # This looks for term_overlap or other metrics in the effectiveness object
            where_clauses.append(f"(CAST(metadata->'effectiveness'->>'term_overlap' AS FLOAT) >= {min_effectiveness} OR metadata->'effectiveness'->>'template_adherence' = 'high')")
            
            # Combine WHERE clauses
            where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
            
            # Execute query
            query = f"""
            SELECT * FROM {self.schema_name}.interactions
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT {limit};
            """
            
            response = self.supabase.rpc('sql', {'command': query}).execute()
            
            if not response.data:
                return []
                
            return response.data
            
        except Exception as e:
            logger.error(f"Error retrieving high-quality interactions: {e}")
            return []

    def add_document_to_knowledge_base(self, content, embedding):
        """Adds a document to the knowledge base with vector embedding."""
        try:
            # Make sure schema exists
            self.create_user_schema_sync()
            
            # Convert numpy array or list to proper format
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
                
            # Format as PostgreSQL vector string
            vector_str = str(embedding).replace(' ', '')
            
            # Log what we're doing
            logger.debug(f"Adding document to knowledge base in schema: {self.schema_name}")
            logger.debug(f"Document content (truncated): {content[:100]}...")
            
            # Insert using the table API
            response = self.supabase.table(f"{self.schema_name}.knowledge_base").insert({
                'content': content,
                'embedding': vector_str
            }).execute()
            
            if response.data is None:
                logger.error("Error adding document to knowledge base")
                return False
                
            logger.info(f"Successfully added document to knowledge base: {response.data}")
            return True
        except Exception as e:
            logger.error(f"Error adding document to knowledge base: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def find_similar_documents(self, embedding: List[float], limit: int = 5, min_similarity: float = 0.7) -> List[Dict]:
        """Find similar documents based on vector similarity."""
        import numpy as np
        try:
            # Format embedding for PostgreSQL pgvector format
            # --- Convert NumPy arrays to lists (Consistent Handling) ---
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()

            # --- Construct the SQL query using Python's f-string ---
            vector_str = str(embedding)  #  [1.0, 2.0, 3.0]

            query = f"""
                SELECT 
                    id, 
                    content, 
                    embedding::text,  -- Cast to text for consistent return type
                    1 - (embedding <=> '{vector_str}'::vector) as similarity
                FROM {self.schema_name}.knowledge_base
                WHERE 1 - (embedding <=> '{vector_str}'::vector) > {min_similarity}
                ORDER BY similarity DESC
                LIMIT {limit};
            """

            logger.info(f"Finding similar documents in schema: {self.schema_name}")
            logger.debug(f"Executing SQL query:\n{query}") # Log the *entire* query

            # --- Execute the raw SQL query using supabase.rpc('sql') ---
            response = self.supabase.rpc('sql', {'command': query}).execute()
            
            if response.data:
                logger.info(f"Found {len(response.data)} similar documents")
                return response.data
            else:
                # Handle case where no documents are found
                logger.warning(f"No similar documents found in knowledge base for schema {self.schema_name}")
                
                # Check if knowledge base exists
                try:
                    # Use SQL query to check if table exists and has entries
                    check_query = f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = '{self.schema_name}'
                        AND table_name = 'knowledge_base'
                    );
                    """
                    exists_response = self.supabase.rpc('sql', {'command': check_query}).execute()
                    table_exists = exists_response.data and exists_response.data[0] == 't'
                    
                    if not table_exists:
                        logger.warning(f"Knowledge base table doesn't exist for schema {self.schema_name}")
                        # Try to initialize it
                        if self.initialize_knowledge_base(self.schema_name):
                            logger.info("Knowledge base initialized, retrying query")
                            # Try the query again
                            response = self.supabase.rpc('sql', {'command': query}).execute()
                            # response = self.supabase.rpc('find_similar_documents', {
                            #     'p_schema_name': self.schema_name,
                            #     'p_embedding': vector_str,
                            #     'p_limit': limit
                            # }).execute()
                            if response.data:
                                return response.data
                    else:
                        # Check if table is empty
                        count_query = f"""
                        SELECT COUNT(*) FROM "{self.schema_name}".knowledge_base;
                        """
                        count_response = self.supabase.rpc('sql', {'command': count_query}).execute()
                        if count_response.data and count_response.data[0] == '0':
                            logger.info(f"Knowledge base exists but is empty, initializing...")
                            if self.initialize_knowledge_base(self.schema_name):
                                # Retry query after initialization
                                response = self.supabase.rpc('sql', {'command': query}).execute()
                                # response = self.supabase.rpc('find_similar_documents', {
                                #     'p_schema_name': self.schema_name,
                                #     'p_embedding': vector_str,
                                #     'p_limit': limit
                                # }).execute()
                                if response.data:
                                    return response.data
                except Exception as check_e:
                    logger.error(f"Error checking knowledge base table: {check_e}")
                    # Try to initialize anyway as a fallback
                    if self.initialize_knowledge_base(self.schema_name):
                        logger.info("Knowledge base initialized, retrying query")
                        response = self.supabase.rpc('sql', {'command': query}).execute()
                        # response = self.supabase.rpc('find_similar_documents', {
                        #     'p_schema_name': self.schema_name,
                        #     'p_embedding': vector_str,
                        #     'p_limit': limit
                        # }).execute()
                        if response.data:
                            return response.data
                
                # If we got here, return empty list
                return []
                
        except Exception as e:
            logger.error(f"Error finding similar documents: {e}")
            logger.error(traceback.format_exc())
            return []

    def add_vector_index_to_knowledge_base(self):
        """
        Adds a vector index to the knowledge base table for the current user schema.
        This improves performance of vector similarity searches.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # First ensure schema exists
            self.create_user_schema_sync()
            
            # Call the function with the current schema name
            response = self.supabase.rpc('add_vector_index_to_knowledge_base', {
                'schema_name': self.schema_name
            }).execute()
            
            if response.data:
                logger.info(f"Successfully added vector index to knowledge_base table for {self.schema_name}")
                return True
            else:
                logger.warning(f"No confirmation received for index creation on {self.schema_name}")
                return False
        except Exception as e:
            logger.error(f"Exception adding vector index: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def clean_db_text(self, text):
        """Clean text retrieved from the database to normalize quotes."""
        if text is None:
            return text
        return re.sub(r"'{2,}", "'", text)

    def connect_psychological_concepts(self, source_id: int, target_id: int, relationship_type: str = "related", strength: float = 1.0):
        """
        Create an explicit connection between two psychological concepts or memories.
        """
        try:
            # Call the database function directly
            response = self.supabase.rpc('connect_psychological_concepts', {
                'p_schema_name': self.schema_name,
                'p_source_id': source_id,
                'p_target_id': target_id,
                'p_relationship_type': relationship_type,
                'p_strength': strength
            }).execute()
            
            if response.data is None:
                logger.error("Error creating psychological connection")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error creating psychological connection: {e}")
            return False

    def extract_psychological_themes(self, session_id: str, min_occurrences: int = 3):
        """
        Identifies recurring psychological themes in a user's conversation history.
        
        Args:
            session_id: Session identifier
            min_occurrences: Minimum number of occurrences to consider a theme significant
            
        Returns:
            dict: Mapping of themes to their occurrence counts
        """
        try:
            # Get conversation history
            history = self.get_conversation_history(session_id)
            
            # Extract themes from metadata and content
            themes = {}
            for interaction in history:
                # Check metadata for explicit themes
                metadata = interaction.get('metadata', '{}')
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {}
                
                # Extract topic from metadata
                topic = metadata.get('topic', '')
                if topic:
                    themes[topic] = themes.get(topic, 0) + 1
                
                # Extract themes from question and answer content
                question = interaction.get('questionText', '')
                answer = interaction.get('answerText', '')
                
                # List of common psychological themes to look for in content
                theme_keywords = {
                    'Abandonment': ['abandon', 'left me', 'alone', 'desert'],
                    'Rejection': ['reject', 'unwanted', 'excluded', 'cast aside'],
                    'Self-Worth': ['worthless', 'undeserving', 'inadequate', 'unworthy', 'failure'],
                    'Trust': ['trust', 'betrayal', 'suspicious', 'faith', 'rely'],
                    'Control': ['control', 'helpless', 'powerless', 'manipulation'],
                    'Emotional Regulation': ['overwhelm', 'regulate', 'emotion', 'feelings', 'manage']
                }
                
                # Check content for theme keywords
                combined_text = (question + " " + answer).lower()
                for theme, keywords in theme_keywords.items():
                    if any(keyword in combined_text for keyword in keywords):
                        themes[theme] = themes.get(theme, 0) + 1
                
            # Return themes that occur at least min_occurrences times
            return {theme: count for theme, count in themes.items() if count >= min_occurrences}
        except Exception as e:
            logger.error(f"Error extracting psychological themes: {e}")
            return {}

    def start_therapy_session(self, session_id: str, session_metadata: dict = None):
        """
        Marks the start of a new therapy session.
        
        Args:
            session_id: Session identifier
            session_metadata: Optional metadata for the session
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            metadata = session_metadata or {}
            metadata['session_start'] = True
            metadata['session_timestamp'] = datetime.now().isoformat()
            
            data_point = {
                'context': 'Session Start',
                'question': 'Beginning of therapy session',
                'answer': '',
                'metadata': metadata
            }
            
            return self.add_interaction(data_point, session_id)
        except Exception as e:
            logger.error(f"Error starting therapy session: {e}")
            return False

    def mark_therapeutic_insight(self, interaction_id: int, insight_level: str, session_id: str = None):
        """
        Marks an interaction as containing a significant therapeutic insight.
        """
        try:
            schema_name = session_id if session_id else self.schema_name
            
            response = self.supabase.rpc('mark_therapeutic_insight', {
                'p_schema_name': schema_name,
                'p_interaction_id': interaction_id,
                'p_insight_level': insight_level
            }).execute()
            
            return response.data is not None
        except Exception as e:
            logger.error(f"Error marking therapeutic insight: {e}")
            return False
            
    def get_psychological_connections(self, concept_id: int, relationship_type: str = None):
        """
        Retrieves psychological connections for a given concept.
        """
        try:
            params = {
                'p_schema_name': self.schema_name,
                'p_concept_id': concept_id
            }
            
            if relationship_type:
                params['p_relationship_type'] = relationship_type
                
            response = self.supabase.rpc('get_psychological_connections', params).execute()
            
            if response.data is None:
                logger.error("Error retrieving psychological connections")
                return []
                
            # Process results to identify the connected concept
            connections = []
            for row in response.data:
                # Determine if the concept is the source or target
                is_source = row.get('source_id') == concept_id
                connected_id = row.get('target_id') if is_source else row.get('source_id')
                
                connections.append({
                    'connection_id': row.get('id'),
                    'concept_id': concept_id,
                    'connected_id': connected_id,
                    'relationship_type': row.get('relationship_type'),
                    'strength': row.get('strength'),
                    'direction': 'outgoing' if is_source else 'incoming',
                    'created_at': row.get('created_at')
                })
                
            return connections
        except Exception as e:
            logger.error(f"Error retrieving psychological connections: {e}")
            return []
            
    def get_session_summary(self, session_id: str):
        """
        Creates a summary of a therapy session with key insights and themes.
        
        Args:
            session_id: Session identifier
            
        Returns:
            dict: Session summary with themes, emotional changes, and insights
        """
        try:
            # Get raw conversation history
            history = self.get_conversation_history(session_id)
            
            if not history:
                return {
                    'session_id': session_id,
                    'error': 'No conversation history found'
                }
            
            # Extract key data points
            themes = self.extract_psychological_themes(session_id, min_occurrences=1)
            emotions = self.analyze_emotional_trajectory(session_id)
            
            # Extract insights if marked in metadata
            insights = []
            for interaction in history:
                metadata = {}
                if isinstance(interaction.get('metadata'), str):
                    try:
                        metadata = json.loads(interaction.get('metadata', '{}'))
                    except:
                        pass
                else:
                    metadata = interaction.get('metadata', {})
                    
                if metadata and 'insight_level' in metadata:
                    insights.append({
                        'interactionID': interaction.get('interactionID'),
                        'question': interaction.get('questionText', ''),
                        'answer': interaction.get('answerText', ''),
                        'insight_level': metadata.get('insight_level'),
                        'created_at': interaction.get('created_at')
                    })
            
            # Calculate session metrics
            start_time = min(entry.get('created_at', '') for entry in history) if history else ''
            end_time = max(entry.get('created_at', '') for entry in history) if history else ''
            
            # Count total interactions
            interaction_count = len(history)
            
            # Determine emotional shift if available
            emotional_shift = None
            if len(emotions) >= 2:
                starting_emotion = emotions[0].get('emotional_state', '')
                ending_emotion = emotions[-1].get('emotional_state', '')
                starting_intensity = emotions[0].get('intensity', 0) 
                ending_intensity = emotions[-1].get('intensity', 0)
                
                emotional_shift = {
                    'starting_state': starting_emotion,
                    'ending_state': ending_emotion,
                    'intensity_change': ending_intensity - starting_intensity
                }
            
            # Build the summary
            summary = {
                'session_id': session_id,
                'start_time': start_time,
                'end_time': end_time,
                'interaction_count': interaction_count,
                'primary_themes': sorted(themes.items(), key=lambda x: x[1], reverse=True)[:3] if themes else [],
                'emotional_shift': emotional_shift,
                'insights': insights
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating session summary: {e}")
            return {
                'session_id': session_id,
                'error': f"Failed to generate summary: {str(e)}"
            }

    def find_related_memories(self, question_text: str, embedding_vector: List[float], session_id: str = None, max_results: int = 3):
        """
        Find memories related to the current question using semantic and keyword matching.
        This mimics how human memory retrieves related experiences.
        
        Args:
            question_text: The current question text
            embedding_vector: Vector embedding of the question
            session_id: Optional session ID (defaults to user's schema)
            max_results: Maximum number of related memories to retrieve
            
        Returns:
            List: Related memories with similarity scores
        """
        try:
            schema_name = self._sanitize_schema_name(session_id) if session_id else self.schema_name
            
            # First search for similar documents using vector similarity
            similar_docs = self.find_similar_documents(embedding_vector, limit=max_results)
            
            # Then get conversation history and find related exchanges
            conversation = self.get_conversation_history(schema_name)
            if not conversation:
                return similar_docs  # Early return if no conversation history
            
            # Get key terms from the question (simple tokenization)
            key_terms = [word.lower() for word in re.findall(r'\b\w+\b', question_text) 
                        if len(word) > 3 and word.lower() not in ['what', 'when', 'where', 'which', 'that', 'this', 'with', 'have', 'from']]
            
            memory_matches = []
            for interaction in conversation:
                question = interaction.get('questionText', '')
                answer = interaction.get('answerText', '')
                
                # Skip empty interactions
                if not question and not answer:
                    continue
                
                # Calculate basic term overlap
                matched_terms = sum(1 for term in key_terms if term.lower() in question.lower() or term.lower() in answer.lower())
                if matched_terms > 0:
                    # Calculate similarity score (simple version)
                    similarity = matched_terms / max(len(key_terms), 1)
                    
                    memory_matches.append({
                        'interactionID': interaction.get('interactionID'),
                        'context': interaction.get('context'),
                        'question': question,
                        'answer': answer,
                        'similarity': min(similarity * 2, 0.9),  # Scale but cap at 0.9
                        'created_at': interaction.get('created_at')
                    })
            
            # Sort by similarity score and limit results
            memory_matches = sorted(memory_matches, key=lambda x: x['similarity'], reverse=True)[:max_results]
            
            # Combine results (vector similarity and term matching)
            combined_results = similar_docs + memory_matches
            
            # Sort by similarity and return unique items (based on content)
            seen_content = set()
            unique_results = []
            
            for item in sorted(combined_results, key=lambda x: x.get('similarity', 0), reverse=True):
                content_key = item.get('content', item.get('question', ''))[:100]
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    unique_results.append(item)
                    if len(unique_results) >= max_results:
                        break
            
            return unique_results
                
        except Exception as e:
            logger.error(f"Error finding related memories: {e}")
            logger.error(traceback.format_exc())
            return []

    def find_similar_memories(self, embedding: list, session_id: str, limit: int = 5, threshold: float = 0.6):
        """
        Find psychologically similar memories using pgvector's optimized similarity search.
        
        Args:
            embedding: Vector embedding to compare against
            session_id: Session identifier
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold (0-1)
            
        Returns:
            List of similar interactions
        """
        try:
            # Convert embedding to string format for Postgres
            embedding_str = str(embedding).replace('[', '{').replace(']', '}')
            
            # Use optimized pgvector function
            response = self.supabase.rpc('find_psychological_memories', {
                'p_schema_name': session_id,
                'p_embedding': embedding_str,
                'p_limit': limit,
                'p_threshold': threshold
            }).execute()
            
            if response.data is None:
                logger.error("Error finding similar memories")
                return []
            
            # Process results
            results = []
            for item in response.data:
                # Parse metadata back from jsonb
                metadata = item.get('metadata', {})
                
                results.append({
                    'interactionid': item.get('interactionid'),
                    'question': item.get('question', ''),
                    'answer': item.get('answer', ''),
                    'context': item.get('context', ''),
                    'metadata': metadata,
                    'similarity': item.get('similarity', 0)
                })
            
            return results
        except Exception as e:
            logger.error(f"Error finding similar memories: {e}")
            return []

    def analyze_theme_clusters(self, session_id: str, min_similarity: float = 0.7, max_clusters: int = 5):
        """
        Analyze psychological theme clusters in vector space.
        
        Args:
            session_id: Session identifier
            min_similarity: Minimum similarity to consider a relationship
            max_clusters: Maximum number of clusters to identify
            
        Returns:
            List of theme clusters with example questions
        """
        try:
            # Ensure we're using a sanitized schema name
            schema_name = self._sanitize_schema_name(session_id)
            
            # Call the pgvector clustering function
            response = self.supabase.rpc('analyze_theme_clusters', {
                'p_schema_name': schema_name,
                'p_min_similarity': min_similarity,
                'p_max_clusters': max_clusters
            }).execute()
            
            if response.data is None:
                logger.error("Error analyzing theme clusters")
                return []
                
            return response.data
        except Exception as e:
            logger.error(f"Error analyzing theme clusters: {e}")
            return []

    def analyze_emotional_vector_trajectory(self, session_id: str):
        """
        Analyze the emotional trajectory in vector space.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of emotional trajectory segments with vector analysis
        """
        try:
            # Ensure we're using a sanitized schema name
            schema_name = self._sanitize_schema_name(session_id)
            
            # Call the pgvector emotional trajectory function
            response = self.supabase.rpc('analyze_emotional_vector_trajectory', {
                'p_schema_name': schema_name
            }).execute()
            
            if response.data is None:
                logger.error("Error analyzing emotional vector trajectory")
                return []
                
            return response.data
        except Exception as e:
            logger.error(f"Error analyzing emotional vector trajectory: {e}")
            return []

    def find_concept_connections(self, concept_id: int, session_id: str = None, threshold: float = 0.7):
        """
        Find connections between psychological concepts in vector space.
        
        Args:
            concept_id: ID of the concept to find connections for
            session_id: Optional session ID (defaults to user schema)
            threshold: Minimum similarity threshold
            
        Returns:
            List of connected concepts with relationship data
        """
        try:
            schema_name = session_id if session_id else self.schema_name
            schema_name = self._sanitize_schema_name(schema_name)
            
            # Call the pgvector concept connections function
            response = self.supabase.rpc('find_concept_connections', {
                'p_schema_name': schema_name,
                'p_concept_id': concept_id,
                'p_threshold': threshold
            }).execute()
            
            if response.data is None:
                logger.error("Error finding concept connections")
                return []
                
            return response.data
        except Exception as e:
            logger.error(f"Error finding concept connections: {e}")
            return []

    def find_cross_session_patterns(self, session_ids: List[str]):
        """
        Find psychological patterns that appear across multiple therapy sessions.
        
        Args:
            session_ids: List of session IDs to analyze
            
        Returns:
            List of patterns with occurrence data
        """
        try:
            # Call the pgvector cross-session pattern function
            response = self.supabase.rpc('find_cross_session_patterns', {
                'p_user_schema': self.schema_name,
                'p_session_ids': session_ids
            }).execute()
            
            if response.data is None:
                logger.error("Error finding cross-session patterns")
                return []
                
            return response.data
        except Exception as e:
            logger.error(f"Error finding cross-session patterns: {e}")
            return []

    def ensure_vector_indexes(self, session_id: str = None):
        """
        Ensure vector indexes exist for efficient similarity searches.
        
        Args:
            session_id: Optional session ID (defaults to user schema)
            
        Returns:
            Boolean indicating success
        """
        try:
            schema_name = session_id if session_id else self.schema_name
            schema_name = self._sanitize_schema_name(schema_name)
            
            # Call the index creation function
            response = self.supabase.rpc('ensure_vector_indexes', {
                'p_schema_name': schema_name
            }).execute()
            
            return response.data is not None
        except Exception as e:
            logger.error(f"Error ensuring vector indexes: {e}")
            return False

    def add_embedding_to_interaction(self, interaction_id: int, embedding: List[float], session_id: str = None):
        """
        Add embedding vector to an existing interaction for improved vector search.
        
        Args:
            interaction_id: ID of the interaction
            embedding: Vector embedding to add
            session_id: Optional session ID (defaults to user schema)
            
        Returns:
            bool: Success status
        """
        try:
            schema_name = session_id if session_id else self.schema_name
            schema_name = self._sanitize_schema_name(schema_name)
            
            # Format embedding for PostgreSQL - USING SQUARE BRACKETS for pgvector
            if isinstance(embedding, list):
                vector_str = f"[{','.join(str(x) for x in embedding)}]"
            else:
                vector_str = f"[{','.join(str(x) for x in embedding.tolist())}]"
            
            # Call the function to add embedding
            response = self.supabase.rpc('add_embedding_to_interaction', {
                'p_schema_name': schema_name,
                'p_interaction_id': interaction_id,
                'p_embedding': vector_str
            }).execute()
            
            return response.data is not None
        except Exception as e:
            logger.error(f"Error adding embedding to interaction: {e}")
            return False

    def add_embedding_to_interactions(self, session_id: str = None):
        """
        Add embedding column to interactions table if it doesn't exist.
        This is typically called once during setup.
        
        Args:
            session_id: Optional session ID (defaults to user schema)
            
        Returns:
            bool: Success status
        """
        try:
            schema_name = session_id if session_id else self.schema_name
            schema_name = self._sanitize_schema_name(schema_name)
            
            # Call the function to add embedding column
            response = self.supabase.rpc('add_embedding_to_interactions', {
                'p_schema_name': schema_name
            }).execute()
            
            return response.data is not None
        except Exception as e:
            logger.error(f"Error adding embedding column: {e}")
            return False

    def get_interactions_without_embeddings(self, session_id: str = None, limit: int = 50):
        """
        Get interactions that don't have embeddings, so they can be enriched.
        
        Args:
            session_id: Optional session ID (defaults to user schema)
            limit: Maximum number of interactions to retrieve
            
        Returns:
            list: Interactions without embeddings
        """
        try:
            schema_name = session_id if session_id else self.schema_name
            schema_name = self._sanitize_schema_name(schema_name)
            
            # Call the function to get interactions without embeddings
            response = self.supabase.rpc('get_interactions_without_embeddings', {
                'p_schema_name': schema_name,
                'p_limit': limit
            }).execute()
            
            if response.data is None:
                logger.error("Error getting interactions without embeddings")
                return []
                
            return response.data
        except Exception as e:
            logger.error(f"Error getting interactions without embeddings: {e}")
            return []

    def enrich_interactions_with_embeddings(self, session_id: str = None, model_name: str = "microsoft/phi-1_5"):
        """
        Enrich interactions that don't have embeddings by generating and adding them.
        This improves vector search capabilities.
        """
        try:
            # Get compatible embedding provider 
            embedding_provider = get_embedding_provider(model_name)
            
            schema_name = session_id if session_id else self.schema_name
            schema_name = self._sanitize_schema_name(schema_name)
            
            # First ensure the embedding column exists
            self.add_embedding_to_interactions(schema_name)
            
            # Get interactions without embeddings
            interactions = self.get_interactions_without_embeddings(schema_name, 50)
            
            # No interactions to process
            if not interactions:
                return 0
                
            # Process each interaction
            enriched_count = 0
            for interaction in interactions:
                try:
                    interaction_id = interaction.get('interactionid')
                    question = interaction.get('question', '')
                    answer = interaction.get('answer', '')
                    
                    # Generate embedding from combined text
                    combined_text = f"Question: {question}\nAnswer: {answer}"
                    embedding = embedding_provider.generate_embedding(combined_text)
                    
                    if embedding:
                        # Add embedding to interaction
                        if self.add_embedding_to_interaction(interaction_id, embedding, schema_name):
                            enriched_count += 1
                except Exception as e:
                    logger.error(f"Error enriching interaction {interaction.get('interactionid')}: {e}")
                    continue
                    
            return enriched_count
        except Exception as e:
            logger.error(f"Error enriching interactions: {e}")
            return 0

    def update_table_statistics(self, session_id: str = None):
        """
        Update table statistics for better query planning.
        This helps the database query planner make better decisions.
        
        Args:
            session_id: Optional session ID (defaults to user schema)
            
        Returns:
            bool: Success status
        """
        try:
            schema_name = session_id if session_id else self.schema_name
            schema_name = self._sanitize_schema_name(schema_name)
            
            # Call the function to update table statistics
            response = self.supabase.rpc('update_table_statistics', {
                'p_schema_name': schema_name
            }).execute()
            
            return response.data is not None
        except Exception as e:
            logger.error(f"Error updating table statistics: {e}")
            return False

    def optimize_vector_operations(self, session_id: str = None):
        """
        Perform a series of optimizations for vector operations:
        1. Ensure vector indexes exist
        2. Add embedding column if needed 
        3. Enrich interactions with embeddings
        4. Update statistics for query planner
        
        Args:
            session_id: Optional session ID (defaults to user schema)
            
        Returns:
            dict: Status report of operations performed
        """
        try:
            schema_name = session_id if session_id else self.schema_name
            schema_name = self._sanitize_schema_name(schema_name)
            
            # Ensure embedding column exists
            column_added = self.add_embedding_to_interactions(schema_name)
            
            # Ensure vector indexes exist
            indexes_created = self.ensure_vector_indexes(schema_name)
            
            # Enrich interactions with embeddings
            enriched_count = self.enrich_interactions_with_embeddings(schema_name)
            
            # Update table statistics for query planner
            stats_updated = self.update_table_statistics(schema_name)
            
            return {
                'column_added': column_added,
                'indexes_created': indexes_created,
                'interactions_enriched': enriched_count,
                'statistics_updated': stats_updated
            }
        except Exception as e:
            logger.error(f"Error optimizing vector operations: {e}")
            return {
                'column_added': False,
                'indexes_created': False,
                'interactions_enriched': 0, 
                'statistics_updated': False,
                'error': str(e)
            }

    def initialize_knowledge_base(self, session_id: str = None) -> bool:
        """
        Initialize knowledge base with foundational therapeutic concepts.
        
        Args:
            session_id: Optional session ID (defaults to user schema)
            
        Returns:
            bool: Success status
        """
        try:
            schema_name = session_id if session_id else self.schema_name
            schema_name = self._sanitize_schema_name(schema_name)
            
            # Create schema and tables if they don't exist
            self.create_user_schema_sync()
            
            # Basic therapeutic knowledge entries
            knowledge_entries = [
                "Depression is characterized by persistent sadness, loss of interest in activities, changes in sleep patterns, low energy, and feelings of worthlessness.",
                "Anxiety disorders involve excessive worry and fear that interferes with daily activities. Physical symptoms include rapid heart rate, sweating, and difficulty breathing.",
                "Cognitive Behavioral Therapy (CBT) focuses on identifying negative thought patterns and changing them to improve emotional regulation and develop coping strategies.",
                "Mindfulness involves focusing on the present moment without judgment, which can reduce stress, anxiety, and improve emotional well-being.",
                "Self-compassion means treating yourself with kindness when facing difficulties, rather than harsh self-criticism.",
                "Grief is a natural response to loss and can involve shock, denial, anger, depression, and acceptance.",
                "Trauma can have lasting effects on mental health, including flashbacks, nightmares, and hypervigilance.",
                "Social support from friends, family, and community is crucial for maintaining mental health during difficult times.",
                "Self-care activities like exercise, proper sleep, nutrition, and relaxation are important for maintaining mental health.",
                "Resilience is the ability to adapt and recover from adversity, which can be developed through building coping skills and support networks."
            ]
            
            # Get embedding provider
            embedding_provider = get_embedding_provider()
            
            # Add entries to knowledge base with embeddings
            success_count = 0
            for i, content in enumerate(knowledge_entries):
                try:
                    logger.debug(f"Generating embedding for entry {i+1}")
                    # Generate embedding
                    embedding = embedding_provider.generate_embedding(content)
                    
                    if embedding is None:
                        logger.error(f"Embedding generation returned None for entry {i+1}")
                        continue
                        
                    logger.debug(f"Successfully generated embedding with length: {len(embedding)}")
                    
                    # Format for PostgreSQL vector - IMPORTANT: Use square brackets format!
                    try:
                        if isinstance(embedding, list):
                            vector_str = f"[{','.join(str(x) for x in embedding)}]"
                        else:
                            vector_str = f"[{','.join(str(x) for x in embedding.tolist())}]"
                        
                        logger.debug(f"Formatted vector string (first 20 chars): {vector_str[:20]}...")
                    except Exception as format_e:
                        logger.error(f"Error formatting vector string: {format_e}")
                        logger.error(f"Embedding type: {type(embedding)}")
                        continue
                    
                    # Direct SQL approach to avoid possible table API issues
                    try:
                        # Escape content for SQL
                        safe_content = content.replace("'", "''")
                        
                        # Use SQL to insert directly
                        query = f"""
                        INSERT INTO "{schema_name}".knowledge_base (content, embedding)
                        VALUES ('{safe_content}', '{vector_str}')
                        RETURNING id;
                        """
                        
                        insert_response = self.supabase.rpc('sql', {'command': query}).execute()
                        
                        if insert_response.data:
                            success_count += 1
                            logger.info(f"Added knowledge base entry {i+1}: {success_count}")
                        else:
                            logger.warning(f"No data returned when inserting entry {i+1}")
                    except Exception as insert_e:
                        logger.error(f"SQL insert error for entry {i+1}: {insert_e}")
                        logger.error(traceback.format_exc())
                        
                        # Try the direct table API as a fallback
                        try:
                            logger.info(f"Trying direct table API as fallback for entry {i+1}")
                            insert_result = self.supabase.table(f"{schema_name}.knowledge_base").insert({
                                "content": content,
                                "embedding": vector_str
                            }).execute()
                            
                            if insert_result.data:
                                success_count += 1
                                logger.info(f"Added knowledge base entry via fallback: {success_count}")
                        except Exception as fallback_e:
                            logger.error(f"Fallback insert also failed: {fallback_e}")
                            logger.error(traceback.format_exc())
                    
                except Exception as e:
                    logger.error(f"Complete error processing entry {i+1}: {e}")
                    logger.error(traceback.format_exc())
                    continue
            
            # Make sure we have at least one success
            if success_count > 0:
                logger.info(f"Successfully initialized knowledge base with {success_count} entries")
                
                # Create vector index for better performance
                self.ensure_vector_indexes(schema_name)
                return True
            else:
                logger.error("Failed to add any knowledge base entries")
                return False
        except Exception as e:
            logger.error(f"Error initializing knowledge base: {e}")
            logger.error(traceback.format_exc())
            return False

    def find_similar_question_embedding(self, question_text: str, session_id: str, similarity_threshold: float = 0.92) -> Optional[List[float]]:
        """
        Find embedding of a similar previous question to avoid regenerating embeddings.
        Uses pgvector similarity search to find cached embeddings of similar questions.
        
        Args:
            question_text: Current question text
            session_id: Session identifier
            similarity_threshold: Minimum text similarity to reuse an embedding
            
        Returns:
            Optional[List[float]]: Embedding vector if a similar question was found, None otherwise
        """
        try:
            # Ensure we're using a sanitized schema name
            schema_name = self._sanitize_schema_name(session_id)
            
            # Use simple preprocessing to normalize the question
            normalized_question = question_text.lower().strip()
            # Escape single quotes for SQL
            normalized_question = normalized_question.replace("'", "''")
            
            # Query for similar questions in previous interactions
            query = f"""
            WITH question_interactions AS (
                SELECT 
                    i.interaction_id,
                    i.question,
                    ie.embedding
                FROM 
                    {schema_name}.interactions i
                JOIN 
                    {schema_name}.interaction_embeddings ie
                ON 
                    i.interaction_id = ie.interaction_id
                WHERE
                    similarity(lower(i.question), '{normalized_question}') > {similarity_threshold}
                ORDER BY
                    similarity(lower(i.question), '{normalized_question}') DESC
                LIMIT 1
            )
            SELECT 
                interaction_id, 
                question,
                embedding::text as embedding_text
            FROM 
                question_interactions;
            """
            
            response = self.supabase.rpc('sql', {'command': query}).execute()
            
            # Debug the response type
            logger.debug(f"Response data type: {type(response.data)}")
            if response.data:
                logger.debug(f"First element type: {type(response.data[0])}")
            
            if response.data and len(response.data) > 0:
                # PostgreSQL might be returning rows as strings, not dictionaries
                # Handle both cases
                
                # Case 1: If response.data[0] is a dictionary (normal case)
                if isinstance(response.data[0], dict):
                    embedding_text = response.data[0].get('embedding_text', '')
                    
                # Case 2: If response.data[0] is a string (your current issue)
                else:
                    # Try to parse the response string
                    # The row is likely a string representation with column values
                    row_values = response.data[0].split(',')
                    if len(row_values) >= 3:  # Make sure we have at least 3 columns
                        # The embedding should be the third element
                        embedding_text = row_values[2]
                    else:
                        logger.error(f"Unexpected response format: {response.data[0]}")
                        return None
                
                if embedding_text:
                    try:
                        # Clean up the embedding text and parse it
                        # Remove any non-numeric characters except periods, commas, minus signs
                        embedding_text = embedding_text.strip('{}[]"\'')
                        
                        # Split by comma and convert to float
                        embedding_values = embedding_text.split(',')
                        embedding = [float(x) for x in embedding_values if x.strip()]
                        
                        # Log a hit in the embedding cache
                        logger.info(f"Found similar question embedding for: '{question_text[:30]}...'")
                        return embedding
                    except Exception as parse_error:
                        logger.error(f"Error parsing embedding text: {parse_error}")
                        logger.error(f"Raw embedding text: {embedding_text}")
                        return None
                    
            # No similar question found above threshold
            logger.debug(f"No similar question embedding found above threshold {similarity_threshold}")
            return None
            
        except Exception as e:
            logger.error(f"Error finding similar question embedding: {e}")
            logger.error(traceback.format_exc())
            return None
