from supabase import create_client
from typing import List, Dict, Any, Optional
import json
import re
import traceback
import numpy as np
from typeguard import typechecked
from datetime import datetime

from school_logging.log import ColoredLogger
from psy_supabase.utilities.utils import clean_text
from psy_supabase.core.model_manager import get_embedding_provider
from psy_supabase.utilities.embedding_utils import format_embedding_for_db
from psy_supabase.utilities.utils_mapping import map_theme_to_approach_type, map_approach_name

# Set up logging
logger = ColoredLogger(__name__)


class DatabaseManager:
    def __init__(self, supabase_url: str, supabase_key: str, user_id: str = 'default'):
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.user_id = user_id
        
        # Special case for default schema
        if user_id == "default":
            self.schema_name = "default"
            self.create_default_schema_sync()
        else:
            self.schema_name = self._sanitize_schema_name(user_id)
            
        self.supabase = create_client(self.supabase_url, self.supabase_key)

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
            if response.data:
                logger.info(f"Default schema and tables created successfully.")
                return True
            return False
        except Exception as e:
            logger.error(f"Error creating default schema: {e}")
            logger.error(traceback.format_exc())
            return False

    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Retrieves conversation history for a specific session."""
        try:
            # Use SQL WHERE clause to filter by session_id in metadatas
            query = f"""
            SELECT * FROM {self.schema_name}.interactions i
            WHERE i.metadata->>'session_id' = '{session_id}'
            ORDER BY created_at ASC;
            """
            
            response = self.supabase.rpc('sql', {'command': query}).execute()

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

    def add_interaction(self, data_point, session_id: Optional[str] = None):
        """Adds an interaction to the database."""
        try:
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
            logger.debug(f"Adding interaction to schema: {self.schema_name}")
            
            try:
                # Try using RPC function first - this is more reliable
                response = self.supabase.rpc('add_interaction', {
                    'p_schema_name': self.schema_name,
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
                    table_name = f"{self.schema_name}.interactions"
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

    def add_interaction_rpc(self, data_point, session_id: Optional[str] = None):
        """Adds an interaction to the database using an RPC call."""
        try:
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
                'p_schema_name': self.schema_name,
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
            
            # Check schema creation response
            if response.data is None:
                logger.error(f"Schema creation failed for user {self.user_id} - no data in response")
                return False
            elif response.data is False:
                logger.error(f"Schema creation function returned FALSE for user {self.user_id}")
                return False
                
            # Now optimize vector queries
            p_response = self.supabase.rpc('optimize_vector_queries', {'p_schema_name': self.schema_name}).execute()
            if p_response.data is None or p_response.data is False:
                logger.warning(f"Vector optimization failed for schema {self.schema_name}")
                # Continue anyway since basic schema creation worked
            else:
                logger.info(f"Vector statistics optimized for schema {self.schema_name}")
                
            logger.info(f"Schema '{self.schema_name}' and tables created successfully.")
            return True

        except Exception as e:
            logger.error(f"Error creating schema for user {self.user_id}: {e}")
            logger.error(traceback.format_exc())
            return False

    def get_interaction_history(self, user_id: str):
        """ Get interaction history from the user's schema """
        logger.info(f"Retrieving interaction history for user: {user_id} with schema {self.schema_name}")

        # Call SQL function to retrieve interaction history
        sql_query = f"SELECT * FROM get_interaction_history('{self.schema_name}')"
        response = self.supabase.rpc('sql', {'command': sql_query}).execute()

        if response.data is None:
            logger.error(f"Error retrieving interaction history for user {user_id}")
            return None
            
        history = response.model_dump_json()
        logger.info(f"Retrieved interaction history for user {user_id}: {history}")
        return history

    def ensure_user_schema_view(self, user_id: str):
        """ Ensure the view for the user schema exists in the public schema """
        logger.info(f"Ensuring view exists for user: {user_id} with schema {self.schema_name}")

        # Call SQL function to ensure the view exists
        sql_query = f"SELECT ensure_user_schema_view('{self.schema_name}')"
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
        return f'"{self.schema_name}"."{table_name}"'

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

    @typechecked
    def find_similar_documents(self,
                               embedding: List[float],
                               table_name: str = "knowledge_base",
                               limit: int = 5,
                               min_similarity: float = 0.7
                               ) -> List[Dict]:
        """
        Find documents similar to the given embedding using pgvector.
        
        Args:
            embedding: Vector embedding to compare against
            table_name: Name of the table to search (e.g., "knowledge_base")
            limit: Maximum number of documents to return
            min_similarity: Minimum cosine similarity threshold
            
        Returns:
            List of similar documents with similarity scores
        """
        try:
            # Format embedding for PostgreSQL pgvector format
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()

            vector_str = format_embedding_for_db(embedding)

            query = f"""
            SELECT 
                id, 
                content, 
                1 - (embedding <=> '{vector_str}'::vector) as similarity
            FROM 
                {self.schema_name}.{table_name}
            WHERE 
                1 - (embedding <=> '{vector_str}'::vector) > {min_similarity}
            ORDER BY 
                similarity DESC
            LIMIT {limit};
            """

            logger.info("Finding similar documents in schema: %s", self.schema_name)
            
            response = self.supabase.rpc('sql', {'command': query}).execute()
            if response.data:
                logger.info(f"Found {len(response.data)} similar documents")
                return response.data
            else:
                logger.warning("No similar documents found in knowledge base for schema %s", self.schema_name)
                return []
                
        except Exception as e:
            logger.error(f"Error finding similar documents: {e}")
            logger.error(traceback.format_exc())
            return []

    def find_similar_documents_via_rpc(self, session_id: str, embedding: List[float], 
                              similarity_threshold: float = 0.7, limit: int = 3):
        """
        Find documents similar to the provided embedding using pgvector directly in PostgreSQL.
        This offloads computation from Python/GPU to the database.
        
        Args:
            session_id: The session ID
            embedding: Vector embedding as a list of floats
            similarity_threshold: Minimum similarity threshold
            limit: Maximum number of results to return
            
        Returns:
            List of documents with similarity scores
        """
        try:
            # Format embedding for PostgreSQL pgvector format
            vector_str = format_embedding_for_db(embedding)
            
            # Create an optimized dynamic SQL query to leverage pgvector within the specific schema
            # This performs the similarity search entirely within PostgreSQL
            query = f"""
            SELECT 
                id, 
                content, 
                metadata,
                1 - (embedding <=> '{vector_str}'::vector) as similarity
            FROM 
                {self.schema_name}.knowledge_base
            WHERE 
                1 - (embedding <=> '{vector_str}'::vector) > {similarity_threshold}
            ORDER BY 
                similarity DESC
            LIMIT {limit};
            """
            
            logger.info(f"Finding similar documents in schema: {self.schema_name}")
            response = self.supabase.rpc('sql', {'command': query}).execute()
            
            if response.data:
                logger.info(f"Found {len(response.data)} similar documents")
                return response.data
            else:
                logger.warning("No similar documents found")
                return []
                
        except Exception as e:
            logger.error(f"Error finding similar documents via RPC: {e}")
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

    def start_therapy_session(self, session_id: str, session_metadata: Optional[dict] = None):
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

    def mark_therapeutic_insight(self, interaction_id: int, insight_level: str, session_id: Optional[str] = None):
        """
        Marks an interaction as containing a significant therapeutic insight.
        """
        try:
            response = self.supabase.rpc('mark_therapeutic_insight', {
                'p_schema_name': self.schema_name,
                'p_interaction_id': interaction_id,
                'p_insight_level': insight_level
            }).execute()
            
            return response.data is not None
        except Exception as e:
            logger.error(f"Error marking therapeutic insight: {e}")
            return False
            
    def get_psychological_connections(self, concept_id: int, relationship_type: Optional[str] = None, session_id: Optional[str] = None):
        """
        Retrieves psychological connections for a given concept.
        
        Args:
            concept_id: ID of the concept to find connections for
            relationship_type: Optional type of relationship to filter by
            session_id: Optional session ID to filter by (stored in metadata)
            
        Returns:
            List of connections for the concept
        """
        try:
            params = {
                'p_schema_name': self.schema_name,
                'p_concept_id': concept_id
            }
            
            if relationship_type:
                params['p_relationship_type'] = relationship_type
                
            if session_id:
                params['p_session_id'] = session_id
                
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

    def find_related_memories(self, question_text: str, embedding_vector: List[float], session_id: Optional[str] = None, max_results: int = 3):
        """
        Find memories related to the current question using semantic and keyword matching.
        This mimics how human memory retrieves related experiences.
        
        Args:
            question_text: The current question text
            embedding_vector: Vector embedding of the question
            session_id: Optional session ID (stored in metadata)
            max_results: Maximum number of related memories to retrieve
            
        Returns:
            List: Related memories with similarity scores
        """
        try:
            # First search for similar documents using vector similarity with session filtering
            similar_docs = self.find_similar_documents_by_embedding(
                embedding=embedding_vector, 
                threshold=0.6,
                limit=max_results
            )
            
            # If no session_id provided, we can only return knowledge base results
            if not session_id:
                return similar_docs
            
            # Then get conversation history filtered by session_id
            conversation = self.get_conversation_history(session_id)
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
        try:
            # Convert embedding to string format for Postgres
            embedding_str = str(embedding).replace('[', '{').replace(']', '}')
            
            # Use optimized pgvector function with session_id as a parameter 
            # (not as schema name)
            response = self.supabase.rpc('find_psychological_memories', {
                'p_schema_name': self.schema_name,  # Use schema_name consistently 
                'p_embedding': embedding_str,
                'p_session_id': session_id,
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
            # Call the pgvector clustering function
            response = self.supabase.rpc('analyze_theme_clusters', {
                'p_schema_name': self.schema_name,
                'p_session_id': session_id, 
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
            session_id: Session identifier to filter interactions
            
        Returns:
            List of emotional trajectory segments with vector analysis
        """
        try:
            # Call the pgvector emotional trajectory function with session_id parameter
            response = self.supabase.rpc('analyze_emotional_vector_trajectory', {
                'p_schema_name': self.schema_name,
                'p_session_id': session_id  # Add this parameter
            }).execute()

            if response.data is None:
                logger.error("Error analyzing emotional vector trajectory")
                return []

            return response.data
        except Exception as e:
            logger.error("Error analyzing emotional vector trajectory %s", e)
            return []

    def find_concept_connections(self, concept_id: int, session_id: Optional[str] = None, threshold: float = 0.7):
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
            # Call the pgvector concept connections function
            response = self.supabase.rpc('find_concept_connections', {
                'p_schema_name': self.schema_name,
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

    def ensure_vector_indexes(self, session_id: Optional[str] = None):
        """
        Ensure vector indexes exist for efficient similarity searches.
        
        Args:
            session_id: Optional session ID (defaults to user schema)
            
        Returns:
            Boolean indicating success
        """
        try:
            # Call the index creation function
            response = self.supabase.rpc('ensure_vector_indexes', {
                'p_schema_name': self.schema_name
            }).execute()
            
            return response.data is not None
        except Exception as e:
            logger.error(f"Error ensuring vector indexes: {e}")
            return False

    def add_embedding_to_interaction(self, interaction_id: int, embedding: List[float], session_id: Optional[str] = None):
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
            # Format embedding for PostgreSQL - USING SQUARE BRACKETS for pgvector
            if isinstance(embedding, list):
                vector_str = f"[{','.join(str(x) for x in embedding)}]"
            else:
                vector_str = f"[{','.join(str(x) for x in embedding.tolist())}]"
            
            # Call the function to add embedding
            response = self.supabase.rpc('add_embedding_to_interaction', {
                'p_schema_name': self.schema_name,
                'p_interaction_id': interaction_id,
                'p_embedding': vector_str
            }).execute()
            
            return response.data is not None
        except Exception as e:
            logger.error(f"Error adding embedding to interaction: {e}")
            return False

    def add_embedding_to_interactions(self, session_id: Optional[str] = None):
        """
        Add embedding column to interactions table if it doesn't exist.
        This is typically called once during setup.
        
        Args:
            session_id: Optional session ID (defaults to user schema)
            
        Returns:
            bool: Success status
        """
        try:
            # Call the function to add embedding column
            response = self.supabase.rpc('add_embedding_to_interactions', {
                'p_schema_name': self.schema_name
            }).execute()

            return response.data is not None
        except Exception as e:
            logger.error(f"Error adding embedding column: {e}")
            return False

    def get_interactions_without_embeddings(self, session_id: Optional[str] = None, limit: int = 50):
        """
        Get interactions that don't have embeddings, so they can be enriched.
        
        Args:
            session_id: Optional session ID (defaults to user schema)
            limit: Maximum number of interactions to retrieve
            
        Returns:
            list: Interactions without embeddings
        """
        try:
            # Call the function to get interactions without embeddings
            response = self.supabase.rpc('get_interactions_without_embeddings', {
                'p_schema_name': self.schema_name,
                'p_limit': limit
            }).execute()
            
            if response.data is None:
                logger.error("Error getting interactions without embeddings")
                return []
                
            return response.data
        except Exception as e:
            logger.error(f"Error getting interactions without embeddings: {e}")
            return []

    def enrich_interactions_with_embeddings(self, session_id: Optional[str] = None, model_name: str = "microsoft/phi-1_5"):
        """
        Enrich interactions that don't have embeddings by generating and adding them.
        This improves vector search capabilities.
        """
        try:
            # Get compatible embedding provider 
            embedding_provider = get_embedding_provider(model_name)

            # First ensure the embedding column exists
            self.add_embedding_to_interactions(self.schema_name)
            
            # Get interactions without embeddings
            interactions = self.get_interactions_without_embeddings(self.schema_name, 50)
            
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
                        if self.add_embedding_to_interaction(interaction_id, embedding, self.schema_name):
                            enriched_count += 1
                except Exception as e:
                    logger.error(f"Error enriching interaction {interaction.get('interactionid')}: {e}")
                    continue
                    
            return enriched_count
        except Exception as e:
            logger.error(f"Error enriching interactions: {e}")
            return 0

    def update_table_statistics(self, session_id: Optional[str] = None):
        """
        Update table statistics for better query planning.
        This helps the database query planner make better decisions.
        
        Args:
            session_id: Optional session ID (defaults to user schema)
            
        Returns:
            bool: Success status
        """
        try:
            # Call the function to update table statistics
            response = self.supabase.rpc('update_table_statistics', {
                'p_schema_name': self.schema_name
            }).execute()
            
            return response.data is not None
        except Exception as e:
            logger.error(f"Error updating table statistics: {e}")
            return False

    def optimize_vector_operations(self, session_id: Optional[str] = None):
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
            # Ensure embedding column exists
            column_added = self.add_embedding_to_interactions(self.schema_name)

            # Ensure vector indexes exist
            indexes_created = self.ensure_vector_indexes(self.schema_name)

            # Enrich interactions with embeddings
            enriched_count = self.enrich_interactions_with_embeddings(self.schema_name)

            # Update table statistics for query planner
            stats_updated = self.update_table_statistics(self.schema_name)

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

    def initialize_knowledge_base(self, session_id: Optional[str] = None) -> bool:
        """
        Initialize knowledge base with foundational therapeutic concepts.
        
        Args:
            session_id: Optional session ID (defaults to user schema)
            
        Returns:
            bool: Success status
        """
        try:
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
                        INSERT INTO "{self.schema_name}".knowledge_base (content, embedding)
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
                            insert_result = self.supabase.table(f"{self.schema_name}.knowledge_base").insert({
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
                self.ensure_vector_indexes(self.schema_name)
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
            # Use simple preprocessing to normalize the question
            normalized_question = question_text.lower().strip()
            # Escape single quotes for SQL
            normalized_question = normalized_question.replace("'", "''")
            
            # Query for similar questions in previous interactions
            query = f"""
            WITH question_interactions AS (
                SELECT 
                    i.interactionID as interaction_id,
                    i.question,
                    ie.embedding
                FROM 
                    {self.schema_name}.interactions i
                JOIN 
                    {self.schema_name}.interaction_embeddings ie
                ON 
                    i.interactionID = ie.interaction_id
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
                    embedding_str = response.data[0].get('embedding_text', None)
                    
                    if embedding_str:
                        # Convert the string to a list of floats
                        try:
                            # Remove brackets and split by commas
                            embedding_list = [float(val) for val in embedding_str.strip('[]').split(',')]
                            logger.info(f"Found similar question with embedding (length: {len(embedding_list)})")
                            return embedding_list
                        except Exception as e:
                            logger.error(f"Error converting embedding string to list: {e}")
                            return None
                            
                # Case 2: If response.data[0] is a string (CSV-like format)
                elif isinstance(response.data[0], str):
                    try:
                        # Format could be something like: "123,What is your name?,{0.1,0.2,...}"
                        parts = response.data[0].split(',', 2)  # Split into 3 parts
                        if len(parts) >= 3:
                            embedding_str = parts[2]  # Get the embedding part
                            
                            # Now handle the embedding text format
                            # Remove any extra brackets and split
                            embedding_str = embedding_str.strip('{}[]').replace('{', '').replace('}', '')
                            embedding_list = [float(val) for val in embedding_str.split(',')]
                            
                            logger.info(f"Found similar question with embedding from string format (length: {len(embedding_list)})")
                            return embedding_list
                    except Exception as e:
                        logger.error(f"Error parsing string result: {e}")
                        return None
            
            # No similar question found
            return None
                
        except Exception as e:
            logger.error(f"Error finding similar question embedding: {e}")
            logger.error(traceback.format_exc())
            return None

    def find_similar_interactions_by_embedding(self, embedding: List[float], session_id: str, 
                                               limit: int = 5, threshold: float = 0.7) -> List[Dict]:
        """
        Find interactions with similar embeddings using pgvector.
        This is useful for identifying pain points in conversation patterns.

        Args:
            embedding: Query embedding vector
            session_id: Session identifier
            limit: Maximum results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of similar interactions with similarity scores
        
        Args:
            embedding: Query embedding vector
            session_id: Session identifier
            limit: Maximum results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of similar interactions with similarity scores
        """
        try:
            # First check if the interaction_embeddings table has any rows
            count_query = f"""
            SELECT COUNT(*) FROM {self.schema_name}.interaction_embeddings;
            """
            
            count_response = self.supabase.rpc('sql', {'command': count_query}).execute()
            
            # If no rows or count is 0, migrate embeddings
            if not count_response.data or count_response.data == 'r' or count_response.data == '0' or count_response.data == 0:
                logger.info(f"interaction_embeddings table is empty, migrating embeddings")
                self.migrate_embeddings_to_interaction_embeddings_table(session_id)
                
            # Format embedding as PostgreSQL vector format
            if isinstance(embedding, list):
                vector_str = str(embedding).replace(' ', '')
            else:
                vector_str = str(embedding.tolist()).replace(' ', '')
            
            # CHANGE HERE: Use "interactionID" instead of "interaction_id" to match DB schema
            query = f"""
            SELECT 
                i.interactionID as interaction_id,  
                i.question,
                i.answer,
                i.created_at,
                1 - (ie.embedding <=> '{vector_str}'::vector) as similarity
            FROM 
                {self.schema_name}.interactions i
            JOIN 
                {self.schema_name}.interaction_embeddings ie
            ON 
                i.interactionID = ie.interaction_id
            WHERE 
                1 - (ie.embedding <=> '{vector_str}'::vector) > {threshold}
            ORDER BY 
                similarity DESC
            LIMIT {limit};
            """
            
            response = self.supabase.rpc('sql', {'command': query}).execute()
            
            if not response.data:
                return []
                
            # Process and return results
            results = []
            for item in response.data:
                # Handle various response formats
                if isinstance(item, str):
                    # Parse CSV-like string format
                    values = item.split(',')
                    if len(values) >= 5:
                        results.append({
                            'interaction_id': int(values[0]) if values[0].isdigit() else 0,
                            'question': values[1],
                            'answer': values[2],
                            'created_at': values[3],
                            'similarity': float(values[4]) if values[4] and values[4].replace('.','').isdigit() else 0
                        })
                elif isinstance(item, dict):
                    # Dictionary format
                    results.append({
                        'interaction_id': item.get('interaction_id'),
                        'question': item.get('question', ''),
                        'answer': item.get('answer', ''),
                        'created_at': item.get('created_at', ''),
                        'similarity': item.get('similarity', 0)
                    })
                    
            return results
        except Exception as e:
            logger.error(f"Error finding similar interactions by embedding: {e}")
            logger.error(traceback.format_exc())
            return []

    def analyze_emotional_response_to_interaction(self, interaction_id: int, session_id: str) -> List[Dict]:
        """
        Analyzes emotional responses to a specific interaction to understand its psychological impact.
        
        Args:
            interaction_id: ID of the interaction to analyze
            session_id: Session identifier
            
        Returns:
            List of emotional responses with metadata
        """
        try:
            # Get conversation history
            history = self.get_conversation_history(self.schema_name)
            
            # Find the interaction and subsequent responses
            found_interaction = False
            emotional_responses = []
            
            for i, interaction in enumerate(history):
                # Check if this is the target interaction
                if str(interaction.get('interactionID')) == str(interaction_id) or \
                   str(interaction.get('interaction_id')) == str(interaction_id):
                    found_interaction = True
                    
                    # Get the next 3 interactions after this one (if available)
                    for j in range(i+1, min(i+4, len(history))):
                        follow_up = history[j]
                        
                        # Extract emotional state from metadata
                        metadata = follow_up.get('metadata', {})
                        if isinstance(metadata, str):
                            try:
                                metadata = json.loads(metadata)
                            except:
                                metadata = {}
                        
                        emotional_state = metadata.get('emotional_state', '')
                        
                        if emotional_state:
                            emotional_responses.append({
                                'interaction_id': follow_up.get('interactionID'),
                                'emotional_state': emotional_state,
                                'question': follow_up.get('questionText', '')[:100],
                                'created_at': follow_up.get('created_at')
                            })
                    break

            if not found_interaction:
                return []

            return emotional_responses
        except Exception as e:
            logger.error(f"Error analyzing emotional response to interaction: {e}")
            return []

    def get_therapeutic_insights_for_interaction(self, interaction_id: int, session_id: str) -> List[Dict]:
        """
        Retrieves therapeutic insights related to a specific interaction.
        
        Args:
            interaction_id: ID of the interaction
            session_id: Session identifier
            
        Returns:
            List of therapeutic insights
        """
        try:            
            # Query for therapeutic insights related to this interaction
            query = f"""
            SELECT 
                i.interaction_id,
                i.question,
                i.answer,
                i.metadata->>'insight_level' as insight_level,
                i.created_at
            FROM 
                {self.schema_name}.interactions i
            WHERE 
                i.interaction_id = {interaction_id}
                OR i.metadata->>'related_to_interaction' = '{interaction_id}'
            ORDER BY 
                i.created_at ASC;
            """
            
            response = self.supabase.rpc('sql', {'command': query}).execute()
            
            if not response.data:
                return []
                
            # Process and return results
            insights = []
            for item in response.data:
                if isinstance(item, str):
                    # Parse CSV-like string
                    values = item.split(',')
                    if len(values) >= 5:
                        insights.append({
                            'interaction_id': values[0],
                            'question': values[1],
                            'answer': values[2],
                            'insight_level': values[3],
                            'created_at': values[4]
                        })
                elif isinstance(item, dict):
                    insights.append({
                        'interaction_id': item.get('interaction_id'),
                        'question': item.get('question', ''),
                        'answer': item.get('answer', ''),
                        'insight_level': item.get('insight_level', ''),
                        'created_at': item.get('created_at', '')
                    })
                    
            return insights
        except Exception as e:
            logger.error(f"Error getting therapeutic insights: {e}")
            return []

    def identify_potential_pain_points(self,
                                       question_text: str,
                                       question_embedding: List[float], 
                                       session_id: str,
                                       pain_threshold: float = 0.85
                                       ) -> Dict:
        """
        Identifies potential psychological pain points by analyzing the current question
        against past user messages using advanced vector similarity.
        
        Args:
            question_text: The current question from the user
            question_embedding: Vector embedding of the question
            session_id: Session identifier
            pain_threshold: Threshold above which we consider a high similarity pain point
            
        Returns:
            Dict containing pain point data if found, empty dict otherwise
        """
        try:
            # Get conversation history with existing embeddings
            history = self.get_conversation_history(session_id)
            
            if not history:
                logger.debug(f"No conversation history found for session {session_id}")
                return {}
            
            # Skip if less than 3 interactions (not enough history to identify patterns)
            if len(history) < 3:
                logger.debug(f"Not enough history to identify pain points ({len(history)} interactions)")
                return {}
                
            # Get embeddings for past questions using optimized pgvector search
            past_questions = []
            for item in history:
                interaction_id = item.get('interactionID')
                past_question = item.get('questionText', '')
                if past_question and interaction_id and past_question != question_text:
                    past_questions.append({
                        'id': interaction_id,
                        'text': past_question,
                        'created_at': item.get('created_at')
                    })
            
            # No past questions to analyze
            if not past_questions:
                return {}
                
            # Find similar questions from history with pgvector
            similar_questions = []
            
            # 1. First method: Use existing embeddings through pgvector
            vector_similar_questions = self.find_similar_interactions_by_embedding(
                question_embedding,
                session_id,
                limit=3,
                threshold=0.7  # Lower threshold to get candidates
            )
            
            # 2. Second method: Check for linguistic/semantic similarity using text
            # This adds a different dimension to similarity detection
            for past_q in past_questions:
                # Simple text similarity check (ratio of common words)
                past_words = set(re.findall(r'\b\w+\b', past_q['text'].lower()))
                current_words = set(re.findall(r'\b\w+\b', question_text.lower()))
                
                if past_words and current_words:
                    # Calculate Jaccard similarity
                    intersection = past_words.intersection(current_words)
                    union = past_words.union(current_words)
                    text_similarity = len(intersection) / len(union) if union else 0
                    
                    # Add if text similarity is high enough
                    if text_similarity > 0.3:  # Lower threshold for text similarity
                        vector_similar_questions.append({
                            'interaction_id': past_q['id'],
                            'question': past_q['text'],
                            'similarity': text_similarity,
                            'method': 'text'
                        })
            
            # No similar questions found with either method
            if not vector_similar_questions:
                return {}
            
            # Combine and deduplicate results
            seen_ids = set()
            for item in vector_similar_questions:
                interaction_id = item.get('interaction_id') or item.get('interactionid')
                if interaction_id and interaction_id not in seen_ids:
                    seen_ids.add(interaction_id)
                    similar_questions.append({
                        'id': interaction_id,
                        'text': item.get('question'),
                        'similarity': item.get('similarity', 0),
                    })
            
            # Sort by similarity (highest first)
            similar_questions = sorted(similar_questions, key=lambda x: x['similarity'], reverse=True)
            
            # Find the highest similarity score
            highest_similarity = similar_questions[0]['similarity'] if similar_questions else 0
            
            # If we have a very high similarity, we've identified a potential pain point
            if highest_similarity >= pain_threshold:
                most_similar = similar_questions[0]
                
                # Extract emotional patterns related to this interaction
                emotions = self.analyze_emotional_response_to_interaction(
                    most_similar['id'], 
                    session_id
                )
                
                # Extract therapeutic insights if available
                insights = self.get_therapeutic_insights_for_interaction(
                    most_similar['id'],
                    session_id
                )
                
                # Find how many times similar questions have been asked
                repetition_pattern = self.detect_repetition_pattern(
                    most_similar['text'],
                    question_text,
                    similar_questions
                )
                
                # Look for recurring terms in the repetition pattern
                primary_theme = 'unknown'
                if repetition_pattern and 'recurring_terms' in repetition_pattern:
                    recurring_terms = repetition_pattern.get('recurring_terms', [])
                    if recurring_terms and len(recurring_terms) > 0:
                        primary_theme = recurring_terms[0]  # Use the first recurring term
                        logger.info(f"Extracted primary theme '{primary_theme}' from repetition pattern")
                
                # If we couldn't get a theme from repetition pattern, extract from the question text
                if primary_theme == 'unknown':
                    # Use simple keyword matching to extract a theme
                    theme_keywords = {
                        'anxiety': ['anxiety', 'anxious', 'worry', 'panic', 'fear'],
                        'depression': ['depression', 'depressed', 'sad', 'unmotivated', 'hopeless'],
                        'relationship': ['relationship', 'partner', 'husband', 'wife', 'girlfriend', 'boyfriend'],
                        'loneliness': ['lonely', 'alone', 'isolated', 'connection'],
                        'self-esteem': ['confidence', 'self-esteem', 'worth', 'failure'],
                        'trauma': ['trauma', 'ptsd', 'abuse', 'assault'],
                        'grief': ['grief', 'loss', 'death', 'died']
                    }
                    
                    # Check for themes in the question text
                    for theme, keywords in theme_keywords.items():
                        if any(keyword in question_text.lower() for keyword in keywords):
                            primary_theme = theme
                            logger.info(f"Extracted primary theme '{primary_theme}' from keywords")
                            break
                            
                    # If still unknown, check the original question text too
                    if primary_theme == 'unknown':
                        for theme, keywords in theme_keywords.items():
                            if any(keyword in most_similar['text'].lower() for keyword in keywords):
                                primary_theme = theme
                                logger.info(f"Extracted primary theme '{primary_theme}' from original question")
                                break
                
                return {
                    'detected': True,
                    'similarity': highest_similarity,
                    'original_question': most_similar['text'],
                    'current_question': question_text,
                    'interaction_id': most_similar['id'],
                    'emotions': emotions,
                    'insights': insights,
                    'repetition_pattern': repetition_pattern,
                    'suggested_approach': {
                        'name': map_approach_name(primary_theme),
                        'approach_type': map_theme_to_approach_type(primary_theme)
                    }
                }
            
            # No pain point detected
            return {}
            
        except Exception as e:
            logger.error(f"Error identifying potential pain points: {e}")
            logger.error(traceback.format_exc())
            return {}

    def migrate_embeddings_to_interaction_embeddings_table(self, session_id: Optional[str] = None) -> int:
        """
        Populates the interaction_embeddings table from existing embeddings in the interactions table.
        
        Args:
            session_id: Optional session ID (defaults to user schema)
            
        Returns:
            int: Number of embeddings migrated
        """
        try:
            
            # SQL that copies embeddings from interactions table to interaction_embeddings table
            # Only copies embeddings that don't already exist in interaction_embeddings
            query = f"""
            INSERT INTO {self.schema_name}.interaction_embeddings (interaction_id, embedding)
            SELECT 
                i.interactionID, 
                i.embedding
            FROM 
                {self.schema_name}.interactions i
            LEFT JOIN 
                {self.schema_name}.interaction_embeddings ie 
            ON 
                i.interactionID = ie.interaction_id
            WHERE 
                i.embedding IS NOT NULL 
                AND ie.interaction_id IS NULL
            RETURNING id;
            """

            response = self.supabase.rpc('sql', {'command': query}).execute()

            if response.data:
                if isinstance(response.data, list):
                    count = len(response.data)
                else:
                    count = 1

                logger.info(f"Migrated {count} embeddings to interaction_embeddings table")
                return count
            else:
                logger.info("No embeddings to migrate")
                return 0

        except Exception as e:
            logger.error(f"Error migrating embeddings to interaction_embeddings table: {e}")
            logger.error(traceback.format_exc())
            return 0

    def create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding for the given text using the appropriate embedding provider.
        
        Args:
            text: Text to create embedding for
            
        Returns:
            List[float]: Embedding vector
        """
        try:
            from psy_supabase.core.model_manager import get_embedding_provider
            
            # Get the embedding provider 
            embedding_provider = get_embedding_provider()
            
            # Generate embedding
            embedding = embedding_provider.generate_embedding(text)
            
            # Convert to proper format if needed
            if hasattr(embedding, 'tolist') and callable(getattr(embedding, 'tolist')):
                embedding = embedding.tolist()
                
            return embedding
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            logger.error(traceback.format_exc())
            return None

    def save_interaction(self,
                         context: str,
                         question: str,
                         answer: str, 
                         metadata: Optional[Dict] = None,
                         session_id: Optional[str] = None
                         ) -> bool:
        """Save an interaction to the database with proper metadata handling."""
        try:
            # Prepare metadata as JSON string
            if metadata is None:
                metadata = {}
                
            # Add session_id to metadata if provided
            if session_id is not None:
                metadata['session_id'] = session_id
                
            # Convert metadata to string if it's a dict
            if isinstance(metadata, dict):
                metadata_str = json.dumps(metadata)
            else:
                metadata_str = str(metadata)
                
            # Clean the text data
            clean_context = self._clean_text_for_db(context)
            clean_question = self._clean_text_for_db(question)
            clean_answer = self._clean_text_for_db(answer)

            # Add the interaction using our RPC function
            response = self.supabase.rpc('add_interaction', {
                'p_schema_name': self.schema_name,
                'p_context': clean_context,
                'p_question': clean_question,
                'p_answer': clean_answer,
                'p_metadata': metadata_str
            }).execute()

            if response.data is None:
                logger.error("Error saving interaction via RPC")
                return False
                
            # Extract interaction ID from response
            interaction_id = int(response.data)
            logger.info(f"Interaction saved successfully with ID: {interaction_id}")

            # Extract interaction ID from response
            interaction_id = int(response.data)
            logger.info(f"Interaction saved successfully with ID: {interaction_id}")

            # If we have an embedding for the question, store it in interaction_embeddings
            try:
                # Generate embedding for the question
                question_embedding = self.create_embedding(clean_question)
                logger.debug(f"Generated embedding: type={type(question_embedding)}, length={len(question_embedding) if question_embedding else 'None'}")
                
                if question_embedding:
                    # Format the embedding for PostgreSQL
                    from psy_supabase.utilities.embedding_utils import format_embedding_for_db
                    embedding_str = format_embedding_for_db(question_embedding)
                    logger.debug(f"Formatted embedding (first 50 chars): {embedding_str[:50]}...")

                    # First check if interaction_embeddings table exists
                    check_table_query = f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = '{self.schema_name}' AND table_name = 'interaction_embeddings'
                    );
                    """
                    check_result = self.supabase.rpc('sql', {'command': check_table_query}).execute()
                    
                    if not check_result.data:
                        logger.warning(f"interaction_embeddings table not found in schema {self.schema_name}, creating it")
                        create_table_result = self.supabase.rpc('create_user_schema_and_tables', {'schema_name': self.schema_name}).execute()
                        logger.debug(f"Table creation result: {create_table_result.data}")
                    
                    # Store in interaction_embeddings table
                    logger.info(f"Storing embedding for interaction {interaction_id}")
                    
                    # METHOD 1: Store via dedicated function (preferred)
                    embed_response = self.supabase.rpc('add_embedding_to_interaction', {
                        'p_schema_name': self.schema_name,
                        'p_interaction_id': interaction_id,
                        'p_embedding': embedding_str
                    }).execute()
                    
                    if embed_response.data is True:
                        logger.info(f"Embedding stored successfully for interaction {interaction_id}")
                        return True
                    else:
                        logger.warning(f"Embedding storage function returned: {embed_response.data}")
                        
                        # METHOD 2: Fall back to direct SQL if the function fails
                        logger.info("Trying direct SQL insertion...")
                        embed_query = f"""
                        INSERT INTO {self.schema_name}.interaction_embeddings 
                        (interaction_id, embedding) 
                        VALUES ({interaction_id}, '{embedding_str}'::vector(2048))
                        ON CONFLICT (interaction_id) DO UPDATE 
                        SET embedding = '{embedding_str}'::vector(2048);
                        """
                        
                        direct_result = self.supabase.rpc('sql', {'command': embed_query}).execute()
                        logger.debug(f"Direct SQL result: {direct_result.data}")
                        
                        # Also update the interactions table for backward compatibility
                        update_query = f"""
                        UPDATE {self.schema_name}.interactions
                        SET embedding = '{embedding_str}'::vector(2048)
                        WHERE interactionID = {interaction_id};
                        """
                        update_result = self.supabase.rpc('sql', {'command': update_query}).execute()
                        logger.debug(f"Update interactions result: {update_result.data}")
                        
                else:
                    logger.warning(f"No embedding generated for question: {clean_question[:100]}...")
                    
            except Exception as embed_error:
                logger.error(f"Error storing embedding: {embed_error}")
                logger.error(traceback.format_exc())
                # Continue anyway since the basic interaction was saved
                
            return True

        except Exception as e:
            logger.error(f"Error saving interaction: {e}")
            logger.error(traceback.format_exc())
            return False

    def _clean_text_for_db(self, text: str) -> str:
        """
        Clean and escape text for database storage.
        
        Args:
            text: Text to clean
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
            
        # PostgreSQL escaping - replace single quotes with two single quotes
        text = text.replace("'", "''")
        
        # Remove any null bytes which can cause issues in PostgreSQL
        text = text.replace('\0', '')
        
        # Optional: Truncate extremely long texts
        max_length = 65000  # PostgreSQL TEXT can handle much more, but this is safer
        if len(text) > max_length:
            text = text[:max_length]
            
        return text

    def detect_pain_points(self, session_id: str, threshold: float = 0.7, min_occurrences: int = 2) -> Dict:
        """
        Detect pain points from conversation history using vector similarity.
        
        Args:
            session_id: The session ID to analyze
            threshold: Similarity threshold for clustering (0.0-1.0)
            min_occurrences: Minimum number of occurrences to consider a pain point
            
        Returns:
            Dict with pain point information: {
                'pain_points': List of pain point objects,
                'severity': Overall severity assessment,
                'first_detected_at': Index of first detection
            }
        """
        try:
            # Get conversation history
            history = self.get_conversation_history(session_id)
            
            if not history or len(history) < min_occurrences:
                return {'pain_points': [], 'severity': 'none', 'first_detected_at': None}
            
            # Extract questions and convert to vectors
            questions = [item.get('questionText', '') for item in history]
            
            # Track clusters of similar questions
            question_clusters = []
            
            # For each question, check if it forms a cluster with others
            for i, question in enumerate(questions):
                # Skip empty questions
                if not question.strip():
                    continue
                    
                # Search for similar questions in the conversation
                similar_indices = []
                
                # Get the vector for this question (using standard embedding function)
                # In a real implementation, you'd use your actual embedding function
                for j, other_question in enumerate(questions):
                    if i == j:  # Skip comparing to self
                        continue
                        
                    # In production, you'd use vector similarity here
                    # For testing, we'll use simple text matching as a proxy
                    similarity = self._text_similarity(question, other_question)
                    
                    if similarity > threshold:
                        similar_indices.append(j)
                
                # If we found enough similar questions, we have a cluster
                if len(similar_indices) + 1 >= min_occurrences:  # +1 to include the current question
                    # Create a pain point cluster
                    cluster = {
                        'indices': [i] + similar_indices,
                        'questions': [questions[i]] + [questions[j] for j in similar_indices],
                        'recurring_terms': self._extract_recurring_terms(
                            [questions[i]] + [questions[j] for j in similar_indices]
                        ),
                        'first_occurrence': min([i] + similar_indices),
                        'count': len(similar_indices) + 1
                    }
                    
                    # Check if this cluster overlaps significantly with an existing one
                    is_new_cluster = True
                    for existing in question_clusters:
                        overlap = len(set(cluster['indices']).intersection(set(existing['indices'])))
                        # If more than 50% overlap, consider it the same cluster
                        if overlap > len(cluster['indices']) / 2:
                            is_new_cluster = False
                            break
                            
                    if is_new_cluster:
                        question_clusters.append(cluster)
            
            # Calculate overall pain point metrics
            pain_points = sorted(question_clusters, key=lambda x: x['first_occurrence'])
            
            # Determine severity based on cluster counts and sizes
            total_questions = len(questions)
            if not pain_points:
                severity = 'none'
            elif sum(p['count'] for p in pain_points) > total_questions * 0.7:
                severity = 'high'
            elif sum(p['count'] for p in pain_points) > total_questions * 0.4:
                severity = 'medium'
            else:
                severity = 'low'
                
            # Find the first detected pain point
            first_detected_at = min([p['first_occurrence'] for p in pain_points]) if pain_points else None
                
            return {
                'pain_points': pain_points,
                'severity': severity,
                'first_detected_at': first_detected_at
            }
            
        except Exception as e:
            logger.error(f"Error detecting pain points: {e}")
            return {'pain_points': [], 'severity': 'none', 'first_detected_at': None}

    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple text similarity for testing purposes.
        In production, use actual vector embeddings.
        """
        # Simple word overlap calculation for testing
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
            
        overlap = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return overlap / union if union > 0 else 0.0

    def _extract_recurring_terms(self, texts: List[str]) -> List[str]:
        """
        Extract common terms from a set of texts that might indicate pain points.
        """
        # Combine all texts
        combined = " ".join(texts).lower()
        
        # Extract words and count frequencies
        words = re.findall(r'\b\w+\b', combined)
        word_counts = {}
        
        for word in words:
            # Skip stop words and very short words
            if len(word) <= 2 or word in ['the', 'and', 'for', 'that', 'this', 'with', 'you']:
                continue
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Find words that appear in multiple texts
        recurring_words = []
        for word, count in word_counts.items():
            # Word must appear multiple times and in multiple texts
            if count >= 2 and sum(1 for text in texts if word in text.lower()) >= 2:
                recurring_words.append(word)
        
        # Sort by frequency
        recurring_words.sort(key=lambda w: word_counts[w], reverse=True)
        
        # Return top terms
        return recurring_words[:5]  # Limit to top 5 terms

    def get_recommended_therapeutic_approach(self, pain_point: Dict) -> Dict:
        """
        Get a recommended therapeutic approach for a detected pain point.
        
        Args:
            pain_point: Information about the detected pain point
            
        Returns:
            Dict with approach information
        """
        try:
            # Determine the pattern type based on recurring terms
            recurring_terms = pain_point.get('recurring_terms', [])
            term_set = set(t.lower() for t in recurring_terms)
            
            approaches = {
                'anxiety': {
                    'name': 'Anxiety Management',
                    'primary_technique': 'CBT',
                    'redirection_strategy': 'Encourage exploration of anxiety triggers and develop coping mechanisms',
                    'exploration_questions': 'What physical sensations do you notice when anxious? What thoughts come to mind?'
                },
                'worried': {
                    'name': 'Worry Management',
                    'primary_technique': 'CBT',
                    'redirection_strategy': 'Examine evidence for and against worries, develop realistic assessments',
                    'exploration_questions': 'How likely is this worry to come true? What would happen if it did?'
                },
                'relationship': {
                    'name': 'Relationship Pattern Exploration',
                    'primary_technique': 'Interpersonal Therapy',
                    'redirection_strategy': 'Explore recurring relationship patterns and attachment style',
                    'exploration_questions': 'Have you noticed this pattern in other relationships? How does this relate to early experiences?'
                },
                'alone': {
                    'name': 'Loneliness & Attachment',
                    'primary_technique': 'Attachment-Based Therapy',
                    'redirection_strategy': 'Explore fear of abandonment and connection needs',
                    'exploration_questions': 'What does being alone mean to you? What feelings come up when you think about it?'
                },
                'sad': {
                    'name': 'Mood Exploration',
                    'primary_technique': 'Behavioral Activation',
                    'redirection_strategy': 'Focus on activities that may improve mood and energy',
                    'exploration_questions': 'What activities used to bring you joy? What small step might feel manageable?'
                },
                'grief': {
                    'name': 'Grief Processing',
                    'primary_technique': 'Grief Therapy',
                    'redirection_strategy': 'Create space for grief expression and meaning-making',
                    'exploration_questions': 'What does this loss mean to you? What memories are most significant?'
                },
                'failure': {
                    'name': 'Self-Criticism Pattern',
                    'primary_technique': 'Compassion-Focused Therapy',
                    'redirection_strategy': 'Develop self-compassion practice and examine inner critic',
                    'exploration_questions': 'How would you respond to a friend who felt this way? What might self-compassion look like here?'
                },
                'trauma': {
                    'name': 'Trauma Processing',
                    'primary_technique': 'Trauma-Informed Care',
                    'redirection_strategy': 'Focus on safety and grounding before processing traumatic content',
                    'exploration_questions': 'What helps you feel safe in the present moment? How can we work on grounding techniques?'
                }
            }
            
            # Find matching approach
            for key, approach in approaches.items():
                if key in term_set or any(key in term.lower() for term in recurring_terms):
                    return approach
            
            # Default approach if no specific match
            return {
                'name': 'General Exploration',
                'primary_technique': 'Person-Centered',
                'redirection_strategy': 'Reflect recurring theme and invite deeper exploration',
                'exploration_questions': 'I notice this theme comes up frequently. Could you share more about what this means to you?'
            }
            
        except Exception as e:
            logger.error(f"Error getting therapeutic approach: {e}")
            return {
                'name': 'Supportive Listening',
                'primary_technique': 'Person-Centered',
                'redirection_strategy': 'Provide empathetic reflection',
                'exploration_questions': 'Can you tell me more about your experience?'
            }

    def analyze_pain_points_over_time(self, session_id: str) -> List[Dict]:
        """
        Analyze how pain points evolve over therapy sessions.
        """
        try:
            # Get conversation history filtered by session_id in metadata
            history = self.get_conversation_history(session_id)

            # Identify session boundaries
            sessions = []
            current_session = []

            for item in history:
                metadata = {}
                if isinstance(item.get('metadata'), str):
                    try:
                        metadata = json.loads(item.get('metadata', '{}'))
                    except:
                        metadata = {}
                elif isinstance(item.get('metadata'), dict):
                    metadata = item.get('metadata')

                # Check for session start markers
                if metadata.get('session_start'):
                    if current_session:
                        sessions.append(current_session)
                    current_session = [item]
                else:
                    current_session.append(item)

            # Add the last session if not empty
            if current_session:
                sessions.append(current_session)

            # If no explicit sessions, create time-based sessions (weekly)
            if not sessions:
                # Sort by timestamp
                try:
                    history = sorted(history, key=lambda x: x.get('created_at', ''))
                    
                    # Group by approximate week
                    week_ms = 7 * 24 * 60 * 60 * 1000  # One week in milliseconds
                    current_week = None
                    current_session = []
                    
                    for item in history:
                        timestamp = item.get('created_at', '')
                        if not timestamp:
                            continue
                            
                        # Extract milliseconds from timestamp
                        import datetime
                        try:
                            dt = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            ms = int(dt.timestamp() * 1000)
                            
                            # Start a new week if needed
                            if current_week is None:
                                current_week = ms
                                current_session = [item]
                            elif ms - current_week > week_ms:
                                sessions.append(current_session)
                                current_session = [item]
                                current_week = ms
                            else:
                                current_session.append(item)
                        except:
                            current_session.append(item)
                            
                    # Add the last session if not empty
                    if current_session:
                        sessions.append(current_session)
                except:
                    # If timestamp parsing fails, fall back to simple chunking
                    chunk_size = max(len(history) // 3, 1)  # At least 3 chunks if possible
                    sessions = [history[i:i+chunk_size] for i in range(0, len(history), chunk_size)]
            
            # Analyze pain points in each session
            results = []
            
            for i, session in enumerate(sessions):
                # Skip very small sessions
                if len(session) < 2:
                    continue
                    
                # Get session date from first message
                session_date = session[0].get('created_at', '')
                
                # Detect pain points in this session
                session_pain_points = self.detect_pain_points(
                    session_id, 
                    threshold=0.6,  # Lower threshold for smaller sample
                    min_occurrences=max(min(len(session) // 3, 2), 1)  # Scale with session size
                )
                
                # Get primary themes from pain points
                primary_themes = []
                approach_types = []
                
                for pp in session_pain_points.get('pain_points', []):
                    # Get the primary theme term if available
                    primary_theme = pp.get('recurring_terms', ['unknown'])[0] if pp.get('recurring_terms') else 'unknown'
                    primary_themes.append(primary_theme)
                    
                    # Map the theme to an approach type
                    approach = map_theme_to_approach_type(primary_theme)
                    approach_types.append(approach)
                
                # Record pain point data with both themes and approach types
                results.append({
                    'session_number': i + 1,
                    'session_date': session_date,
                    'pain_point_count': len(session_pain_points.get('pain_points', [])),
                    'severity': session_pain_points.get('severity', 'none'),
                    'primary_themes': primary_themes,
                    'approach_types': approach_types,
                    'message_count': len(session)
                })
                
            return results
                
        except Exception as e:
            logger.error(f"Error analyzing pain points over time: {e}")
            return []

    def find_similar_documents_by_embedding(self, embedding, threshold=0.5, limit=5):
        """
        Find documents similar to the provided embedding vector.
        
        Args:
            embedding: The embedding vector to compare against
            threshold: Minimum similarity threshold (0-1)
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries containing similar documents
        """
        try:
            # Ensure we have a valid embedding
            if not embedding or not isinstance(embedding, list):
                logger.error("Invalid embedding provided to find_similar_documents_by_embedding")
                return []

            # Use the pgvector extension to find similar documents
            # The <=> operator is the cosine distance operator (lower is more similar)
            query = f"""
            WITH embedding_vector AS (
                SELECT '{str(embedding).replace(' ', '')}'::vector as embedding
            )
            SELECT 
                id, 
                content, 
                metadata,
                1 - (embedding <=> (SELECT embedding FROM embedding_vector)) as similarity
            FROM 
                {self.schema_name}.knowledge_base
            WHERE 
                1 - (embedding <=> (SELECT embedding FROM embedding_vector)) > {threshold}
            ORDER BY 
                similarity DESC
            LIMIT {limit};
            """
            
            # Execute the query using RPC
            response = self.supabase.rpc('sql', {'command': query}).execute()
            
            if not response.data:
                logger.warning(f"No similar documents found with threshold {threshold}")
                return []
                
            # Format the results
            results = []
            for item in response.data:
                if isinstance(item, dict):
                    # Well-formed response
                    results.append({
                        'id': item.get('id'),
                        'content': item.get('content', ''),
                        'metadata': item.get('metadata', {}),
                        'similarity': item.get('similarity', 0)
                    })
                elif isinstance(item, str):
                    # Fallback for string response format
                    parts = item.split(',', 3)
                    if len(parts) >= 3:
                        results.append({
                            'id': parts[0],
                            'content': parts[1],
                            'metadata': {},
                            'similarity': float(parts[2]) if len(parts) > 2 else 0
                        })
                        
            return results
            
        except Exception as e:
            logger.error(f"Error finding similar documents by embedding: {e}")
            return []

    def get_emotional_signals(self, session_id):
        """
        Analyze emotional signals from user interactions in the current session.
        
        Args:
            session_id: The session ID to analyze
            
        Returns:
            Dictionary with emotional signals and their frequencies
        """
        try:
            # Ensure we have a valid session
            if not session_id:
                return {"error": "No session ID provided"}

            # Query to analyze emotional content across user messages
            query = f"""
            WITH user_messages AS (
                SELECT 
                    question as text,
                    created_at
                FROM 
                    {self.schema_name}.interactions
                WHERE 
                    question IS NOT NULL AND question != ''
                    AND metadata->>'session_id' = '{session_id}'  # Add this filter
                ORDER BY 
                    created_at DESC
                LIMIT 10
            ),
            emotion_words AS (
                SELECT word, category FROM (
                    VALUES 
                    ('happy', 'joy'),
                    ('joy', 'joy'),
                    ('excited', 'joy'),
                    ('pleased', 'joy'),
                    ('sad', 'sadness'),
                    ('unhappy', 'sadness'),
                    ('depressed', 'sadness'),
                    ('miserable', 'sadness'),
                    ('angry', 'anger'),
                    ('frustrated', 'anger'),
                    ('annoyed', 'anger'),
                    ('furious', 'anger'),
                    ('anxious', 'anxiety'),
                    ('worried', 'anxiety'),
                    ('nervous', 'anxiety'),
                    ('scared', 'anxiety'),
                    ('confused', 'confusion'),
                    ('uncertain', 'confusion'),
                    ('lost', 'confusion'),
                    ('hopeful', 'hope'),
                    ('optimistic', 'hope'),
                    ('grateful', 'gratitude'),
                    ('thankful', 'gratitude'),
                    ('lonely', 'loneliness'),
                    ('alone', 'loneliness'),
                    ('isolated', 'loneliness'),
                    ('ashamed', 'shame'),
                    ('embarrassed', 'shame'),
                    ('guilty', 'guilt')
                ) AS t(word, category)
            ),
            word_matches AS (
                SELECT 
                    e.category,
                    COUNT(*) as frequency
                FROM 
                    user_messages m,
                    emotion_words e
                WHERE 
                    m.text ILIKE '%' || e.word || '%'
                GROUP BY 
                    e.category
                ORDER BY 
                    frequency DESC
            )
            SELECT 
                category,
                frequency
            FROM 
                word_matches
            ORDER BY 
                frequency DESC
            LIMIT 5;
            """
            
            # Execute the query
            response = self.supabase.rpc('sql', {'command': query}).execute()
            
            if not response.data:
                # No emotional signals detected
                return {"signals": [], "primary_emotion": "neutral"}
                
            # Format the results
            signals = []
            for item in response.data:
                if isinstance(item, dict):
                    signals.append({
                        'emotion': item.get('category', 'unknown'),
                        'frequency': item.get('frequency', 0)
                    })
                elif isinstance(item, str):
                    parts = item.split(',')
                    if len(parts) >= 2:
                        signals.append({
                            'emotion': parts[0],
                            'frequency': int(parts[1]) if parts[1].isdigit() else 0
                        })
            
            # Determine primary emotion
            primary_emotion = signals[0]['emotion'] if signals else "neutral"
            
            return {
                "signals": signals,
                "primary_emotion": primary_emotion
            }
            
        except Exception as e:
            logger.error(f"Error analyzing emotional signals: {e}")
            return {"error": str(e), "signals": [], "primary_emotion": "neutral"}
