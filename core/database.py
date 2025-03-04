import logging
from supabase import create_client, Client
from typing import List, Dict
import json
import re
import traceback
from utilities.text_utils import clean_text

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
            self.schema_name = "default"  # Don't prepend "user_" for default schema
        else:
            self.schema_name = f"user_{user_id}"
            
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
            if response.error:
                logger.error(f"Error creating schema for user {self.user_id}: {response.error}")
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
            # Call the stored procedure to retrieve conversation history
            response = self.supabase.rpc('get_conversation_history', {'schema_name': session_id}).execute()

            # Check if the response contains data
            if response.data:
                # Transform the data to have the expected field names
                transformed_data = []
                for item in response.data:
                    transformed_item = {
                        'interactionID': item.get('interactionid'),
                        'questionText': item.get('question'),  # in db it is question
                        'answerText': item.get('answer'),      # in db it is answer
                        'context': item.get('context'),
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
            schema_name = session_id if session_id else self.schema_name
            
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
                
                if hasattr(response, 'error') and response.error:
                    logger.error(f"RPC add_interaction failed: {response.error}")
                    raise Exception(str(response.error))
                    
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
            schema_name = session_id if session_id else self.schema_name
            
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
            
            if response.error:
                logger.error(f"Error adding interaction via RPC: {response.error}")
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
            
            # Check the response structure and handle it appropriately
            # Let's first debug the response to understand its structure
            logger.debug(f"Response type: {type(response)}")
            logger.debug(f"Response attributes: {dir(response)}")
            logger.debug(f"Response data: {response.data}")
            
            # Handle the response based on the data it contains
            if hasattr(response, 'data'):
                if response.data is False:
                    logger.error(f"Schema creation failed for user {self.user_id}")
                    return False
                else:
                    logger.info(f"Schema '{self.schema_name}' and tables created successfully.")
                    return True
            else:
                # Fallback check for error message in the response object
                response_dict = response.__dict__ if hasattr(response, '__dict__') else {}
                if 'error' in response_dict:
                    logger.error(f"Error creating schema for user {self.user_id}: {response_dict['error']}")
                    return False
                
                # If we can't determine the result, log a warning and assume success
                logger.warning(f"Could not determine schema creation result. Assuming success.")
                return True

        except Exception as e:
            logger.error(f"Error creating schema for user {self.user_id}: {e}")
            logger.error(traceback.format_exc())
            return False

    def get_interaction_history(self, user_id: str):
        """ Get interaction history from the user's schema """
        schema_name = f'user_{user_id}'
        logger.info(f"Retrieving interaction history for user: {user_id} with schema {schema_name}")

        # Call SQL function to retrieve interaction history
        sql_query = f"SELECT * FROM get_interaction_history('{schema_name}')"
        response = self.supabase.rpc('sql', {'command': sql_query}).execute()

        if response.data:
            history = response.model_dump_json()
            logger.info(f"Retrieved interaction history for user {user_id}: {history}")
            return history
        else:
            logger.error(f"Error retrieving interaction history for user {user_id}: {response.json()}")
            return None

    def ensure_user_schema_view(self, user_id: str):
        """ Ensure the view for the user schema exists in the public schema """
        schema_name = f'user_{user_id}'
        logger.info(f"Ensuring view exists for user: {user_id} with schema {schema_name}")

        # Call SQL function to ensure the view exists
        sql_query = f"SELECT ensure_user_schema_view('{schema_name}')"
        response = self.supabase.rpc('sql', {'command': sql_query}).execute()

        if response.data:
            logger.info(f"View for user {user_id} confirmed.")
        else:
            logger.error(f"Error confirming view for user {user_id}: {response.json()}")

    def _sanitize_schema_name(self, user_id: str) -> str:
        """Sanitizes the user ID to be a valid PostgreSQL schema name (private method)."""
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
            
            if response.error:
                logger.error(f"Error adding document: {response.error}")
                return False
                
            logger.info(f"Successfully added document to knowledge base: {response.data}")
            return True
        except Exception as e:
            logger.error(f"Error adding document to knowledge base: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def find_similar_documents(self, embedding, limit: int = 5) -> List[Dict]:
        """
        Find documents similar to the given embedding using vector similarity search.
        
        Args:
            embedding: The embedding vector to search with
            limit: Maximum number of results to return
            
        Returns:
            List of documents with their similarity scores
        """
        try:
            # Convert numpy array or list to proper format
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
                
            # Format as PostgreSQL vector string
            vector_str = str(embedding).replace(' ', '')
            
            # Use the vector similarity search function
            response = self.supabase.rpc('find_similar_documents', {
                'p_schema_name': self.schema_name,
                'p_embedding': vector_str,
                'p_limit': limit
            }).execute()
            
            if response.data:
                logger.info(f"Found {len(response.data)} similar documents")
                return response.data
            else:
                logger.warning("No similar documents found")
                return []
        except Exception as e:
            logger.error(f"Error finding similar documents: {str(e)}")
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
