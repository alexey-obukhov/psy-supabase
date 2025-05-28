"""
database.py

This module implements the `DatabaseManager` class, which provides a comprehensive interface for interacting
with a Supabase database in the context of psychological AI applications. It supports operations such as
schema management, interaction tracking, vector embedding storage, and retrieval of knowledge and conversation history.

Key Features:

- **Schema Management**:
  - Create and manage user-specific schemas for isolating data.
  - Ensure schema structure and optimize vector operations for efficient queries.

- **Interaction Management**:
  - Add, retrieve, and analyse user interactions, including questions, answers, and metadata.
  - Support for embedding generation and storage for vector similarity searches.

- **Knowledge Base Operations**:
  - Add documents to the knowledge base with vector embeddings.
  - Retrieve similar documents using pgvector for semantic similarity.

- **Psychological Analysis**:
  - Detect recurring psychological themes and pain points in user interactions.
  - Analyze emotional signals and trajectories over time.
  - Identify therapeutic insights and recommend therapeutic approaches.

- **Advanced Vector Operations**:
  - Perform vector similarity searches for documents and interactions.
  - Optimize vector indexes and enrich interactions with embeddings.

Classes:

- `DatabaseManager`: The main class that provides methods for schema management, interaction tracking,
  knowledge base operations, and psychological analysis.

Dependencies:
- `supabase.create_client`: Used for interacting with the Supabase database.
- `psy_supabase.utilities.utils`: Utility functions for cleaning and formatting text.
- `psy_supabase.core.model_manager`: Provides embedding generation for vector operations.
- `psy_supabase.utilities.embedding_utils`: Utilities for formatting embeddings for database storage.
- `prismalog.log.ColoredLogger`: Enhanced logging for debugging and monitoring.

Usage:

.. code-block:: python

    from psy_supabase.core.database import DatabaseManager

    # Initialize the database manager
    db_manager = DatabaseManager(supabase_url="https://your-supabase-url", supabase_key="your-supabase-key")

    # Add an interaction
    db_manager.add_interaction({
        "context": "Therapeutic context",
        "question": "How can I manage my anxiety?",
        "answer": "Practice mindfulness and deep breathing exercises.",
        "metadata": {"topic": "anxiety"}
    }, session_id="session_123")

    # Retrieve conversation history
    history = db_manager.get_conversation_history(session_id="session_123")

    # Find similar documents
    similar_docs = db_manager.find_similar_documents(embedding=[0.1, 0.2, 0.3], limit=5)

    # Analyze psychological themes
    themes = db_manager.extract_psychological_themes(session_id="session_123")

"""

import json
import re
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from supabase import create_client
from typeguard import typechecked

from psy_supabase import get_package_logger
from psy_supabase.config import DEFAULT_APPROACH, DEFAULT_EMOTION, DEFAULT_THEME, DEFAULT_TOPIC, TEXT_GENERATING_MODEL
from psy_supabase.core.model_manager import get_embedding_provider
from psy_supabase.utilities.embedding_utils import detect_repetition_pattern
from psy_supabase.utilities.utils import clean_text, debug_errors
from psy_supabase.utilities.utils_mapping import map_approach_name, map_theme_to_approach_type
from psy_supabase.utilities.vector_utils import ensure_vector_indexes
from psy_supabase.utilities.vector_utils import optimize_vector_operations as optimize_vectors
from psy_supabase.utilities.vector_utils import update_table_statistics

# Set up logging
logger = get_package_logger(__name__)


class DatabaseManager:
    """Database manager for psychology-specific Supabase operations.

    This class provides a comprehensive interface for working with a Supabase
    database in psychological applications, including vector embeddings,
    conversation history tracking, and therapeutic data analysis.

    Attributes:
        supabase_url (str): URL to the Supabase instance
        supabase_key (str): API key for authentication
        user_id (str): Identifier used for schema isolation
        schema_name (str): PostgreSQL schema name for this user
    """

    DEFAULT_TOPIC = "supportive_listening"
    DEFAULT_EMOTION = "concern"
    DEFAULT_APPROACH = "empathy_validation"
    DEFAULT_THEME = "general_support"

    def __init__(self, supabase_url: str, supabase_key: str, user_id: str = "default"):
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

    def create_default_schema_sync(self) -> bool:
        """Creates a default schema if it doesn't exist."""
        try:
            response = self.supabase.rpc("create_default_schema_and_tables").execute()
            if response.data:
                logger.info("Default schema and tables created successfully.")
                return True
            return False
        except Exception as e:
            logger.error("Error creating default schema: %s", e)
            logger.error(traceback.format_exc())
            return False

    @property
    def theme_keywords(self) -> Dict[str, List[str]]:
        """Get keywords for therapeutic themes."""
        from psy_supabase.utilities.therapeutic_mappings import TherapeuticMappings

        return {theme: data["keywords"] for theme, data in TherapeuticMappings.THERAPEUTIC_THEMES.items()}

    @typechecked
    def get_conversation_history(self, session_id: Optional[str] = None) -> List[Dict]:
        """
        Retrieve conversation history for a specific session.

        Args:
            session_id: The session ID to fetch history for

        Returns:
            List of interaction dictionaries
        """
        try:
            if not session_id:
                logger.warning("No session_id provided to get_conversation_history")
                return []

            if not self.schema_name:
                logger.error("Schema name is not set")
                return []

            # Log for debugging
            logger.debug("Retrieving conversation history for session: %s", session_id)

            # Build parameters for RPC call
            rpc_params = {"p_schema_name": self.schema_name, "p_session_id": session_id}

            # Check if schema exists
            response = self.supabase.rpc("get_conversation_history", rpc_params).execute()

            if bool(response.data) and all(v is None for v in response.data[0].values()):
                logger.debug("No data returned from get_conversation_history")
                return []

            # Process the results - handle different response formats
            result = []
            for item in response.data:
                if isinstance(item, dict):
                    interaction_id = item.get("interaction_id")
                    question = item.get("question", "")
                    answer = item.get("answer", "")
                    context = item.get("context", "")
                    metadata = item.get("metadata", {})
                    created_at = item.get("created_at")
                    item_session_id = item.get("session_id", session_id)

                    # Create standardized response
                    transformed_item = {
                        "interaction_id": interaction_id,
                        "question": self.clean_db_text(question),
                        "answer": self.clean_db_text(answer),
                        "context": self.clean_db_text(context),
                        "metadata": metadata,
                        "session_id": item_session_id,
                        "created_at": created_at,
                    }
                    result.append(transformed_item)

            logger.debug("Found %d interactions for session %s", len(result), session_id)
            return result

        except Exception as e:
            logger.error("Error retrieving conversation history: %s", e)
            logger.error(traceback.format_exc())
            return []

    @typechecked
    def add_interaction(self, data_point: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Add an interaction to the databases interactions table.

        Args:
            data_point: Dictionary containing interaction data
            session_id: Optional session identifier

        Returns:
            Dictionary with operation result
        """
        try:
            question = data_point.get("question", "")
            answer = data_point.get("answer", "")
            context = data_point.get("context", "")
            metadata = data_point.get("metadata", {})

            # Ensure metadata is a dictionary
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {"raw_metadata": metadata}

            # Add the session_id to metadata for backward compatibility
            if session_id:
                metadata["session_id"] = session_id

            # Log the exact data being inserted for debugging
            logger.debug("Adding interaction with data: %s..., session_id: %s", question[:30], session_id)

            # Bypass RPC completely - use direct table insertion
            insert_data = {"question": question, "answer": answer, "context": context, "metadata": metadata}

            # Include session_id if provided
            if session_id:
                insert_data["session_id"] = session_id

            # Direct table insertion
            response = self.supabase.rpc(
                "add_interaction",
                {
                    "p_schema_name": self.schema_name,
                    "p_context": context,
                    "p_question": question,
                    "p_answer": answer,
                    "p_metadata": metadata,
                    "p_session_id": session_id,
                },
            ).execute()

            if not response.data:
                logger.error("No data returned from insert operation")
                return {"success": False, "error": "No data returned from insert operation"}

            # Check if we got data back and extract ID
            interaction_id = None
            if response.data > 0:
                interaction_id = response.data

            if interaction_id is None:
                logger.error("Failed to extract interaction ID from response: %s", response.data)
                return {"success": False, "error": "Failed to extract interaction ID"}

            logger.debug("Interaction added with ID %d", interaction_id)

            # Generate and store embedding if content is valid
            if question:
                question_embedding = get_embedding_provider().generate_embedding(question)
                embedding_result = self.add_embedding_to_interaction(interaction_id, question_embedding)
                if not embedding_result:
                    logger.error("Failed to add embedding for interaction %d", interaction_id)
                    return {"success": False, "error": "Failed to add embedding for interaction"}
                return {"success": True}
            return {"success": False, "error": "Question is empty"}

        except Exception as e:
            logger.error("Error adding interaction: %s", e)
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}

    def add_interaction_rpc(self, data_point: dict, session_id: Optional[str] = None) -> bool:
        """Adds an interaction to the database using an RPC call."""
        try:
            metadata_dict: Dict[str, Any] = {}
            raw_metadata = data_point.get("metadata")

            # Populate the dictionary based on the raw metadata type
            if isinstance(raw_metadata, dict):
                metadata_dict = raw_metadata.copy()  # Use a copy to avoid modifying original
            elif isinstance(raw_metadata, str):
                try:
                    # Try loading if it's a valid JSON string
                    metadata_dict = json.loads(raw_metadata)
                    if not isinstance(metadata_dict, dict):  # Ensure it loaded as a dict
                        logger.warning("Metadata string did not decode to a dictionary. Storing as raw.")
                        metadata_dict = {"raw_metadata": raw_metadata}
                except json.JSONDecodeError:
                    # If it's not valid JSON, store the raw string under a key
                    logger.warning("Metadata was a string but not valid JSON. Storing as raw.")
                    metadata_dict = {"raw_metadata": raw_metadata}

            # Add session_id to metadata if provided
            if session_id:
                metadata_dict["session_id"] = session_id

            metadata_json_str = json.dumps(metadata_dict)

            # Clean and escape the values (use .get with defaults for safety)
            context = clean_text(data_point.get("context", ""))
            question = clean_text(data_point.get("question", ""))
            answer = clean_text(data_point.get("answer", ""))

            # Call the RPC function to add the interaction
            response = self.supabase.rpc(
                "add_interaction",
                {
                    "p_schema_name": self.schema_name,
                    "p_context": context,
                    "p_question": question,
                    "p_answer": answer,
                    "p_metadata": metadata_json_str,
                    "p_session_id": session_id,
                },
            ).execute()

            if response.data is None:
                logger.error("Error adding interaction via RPC")
                return False

            logger.info("Interaction added successfully with ID: %s", response.data)
            return True
        except Exception as e:
            logger.error("Exception adding interaction: %s", str(e))
            logger.error(traceback.format_exc())
            return False

    def create_user_schema_sync(self) -> bool:
        """Creates a user-specific schema and tables if they don't exist (synchronous version)."""
        try:
            # First check if schema already exists
            schema_check = self.supabase.rpc("get_schema_exists", {"p_schema_name": self.schema_name}).execute()

            if schema_check.data:
                logger.debug("Schema '%s' already exists, skipping creation", self.schema_name)
                return True

            # Continue with schema creation for new users
            response = self.supabase.rpc("create_user_schema_and_tables", {"schema_name": self.schema_name}).execute()

            # Check schema creation response
            if response.data is None:
                logger.error("Schema creation failed for user %s - no data in response", self.user_id)
                return False
            if response.data is False:
                error_message = response.error if response.error else "Schema creation failed"
                logger.error("Error creating schema for user %s: %s", self.user_id, error_message)
                return False

            logger.info("Schema '%s' and tables created successfully.", self.schema_name)

            # Verify the schema structure after creation
            if self.verify_schema_structure():
                logger.info("Schema structure verification successful for %s", self.schema_name)
            else:
                logger.warning("Schema structure verification failed for %s", self.schema_name)

            return True

        except Exception as e:
            logger.error("Error creating schema for user %s: %s", self.user_id, e)
            logger.error(traceback.format_exc())
            return False

    def get_interaction_history(self, user_id: str) -> Optional[str]:
        """Get interaction history from the user's schema"""
        logger.info("Retrieving interaction history for user: %s with schema %s", user_id, self.schema_name)

        # Call SQL function to retrieve interaction history
        sql_query = f"SELECT * FROM get_interaction_history('{self.schema_name}')"
        response = self.supabase.rpc("sql", {"command": sql_query}).execute()

        if response.data is None:
            logger.error("Error retrieving interaction history for user %s", user_id)
            return None

        history = response.model_dump_json()
        logger.info("Retrieved interaction history for user %s: %s", user_id, history)
        return history

    def _sanitize_schema_name(self, user_id: str) -> str:
        """Sanitizes the user ID to be a valid PostgreSQL schema name (private method)."""
        if not user_id:
            return "default"

        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", user_id)
        if not (safe_name[0].isalpha() or safe_name[0] == "_"):
            safe_name = "_" + safe_name
        return safe_name[:63]

    def get_topic_interactions(self, session_id: str, topic: str, limit: int = 3) -> List[Optional[Dict]]:
        """Retrieves interactions related to a specific topic."""
        try:
            # Get all interactions from the session
            all_interactions = self.get_conversation_history(session_id)

            # Filter interactions by topic
            topic_interactions = []
            for interaction in all_interactions:
                # Extract metadata
                metadata = interaction.get("metadata", {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        continue

                # Check if the interaction is related to the topic
                interaction_topic = metadata.get("topic", DEFAULT_TOPIC).lower()
                if interaction_topic == topic.lower() and interaction not in topic_interactions:
                    topic_interactions.append(interaction)

            # Return the most recent interactions up to the limit
            return topic_interactions[-limit:] if topic_interactions else []
        except Exception as e:
            logger.error("Error getting topic interactions: %s", e)
            return []

    def get_high_quality_interactions(
        self, topic_filter: Optional[str] = None, min_effectiveness: float = 0.7, limit: int = 100
    ) -> List[Dict]:
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
            # Call the SQL function via RPC
            response = self.supabase.rpc(
                "get_high_quality_interactions",
                {
                    "p_schema_name": self.schema_name,
                    "p_topic_filter": topic_filter,
                    "p_min_effectiveness": min_effectiveness,
                    "p_limit": limit,
                },
            ).execute()

            # Check if the response contains data
            if response.data:
                return response.data
            logger.warning("No high-quality interactions found for topic: %s", topic_filter)
            return []

        except Exception as e:
            logger.error("Error retrieving high-quality interactions: %s", e)
            return []

    @typechecked
    def find_similar_documents(
        self,
        query_text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        limit: int = 5,
        min_similarity: float = 0.1,
    ) -> List[Dict]:
        """
        Find documents similar to the query text or embedding using pgvector.

        This is a high-level function that handles both text and embedding inputs.
        For embedding inputs, it calls find_similar_documents_via_rpc internally.

        Args:
            query_text: Text to find similar documents for (optional if embedding is provided)
            embedding: Vector embedding to compare against (optional if query_text is provided)
            limit: Maximum number of documents to return
            min_similarity: Minimum similarity threshold (0-1)

        Returns:
            List of documents with similarity scores

        Note:
            This is the primary entry point for content-based searches. For direct embedding
            searches, use find_similar_documents_via_rpc instead.
        """
        try:
            logger.info("Finding similar documents in schema: %s", self.schema_name)

            # Check if we have either query_text or embedding
            if embedding is None and query_text is None:
                logger.error("Must provide either query_text or embedding for similarity search")
                return []

            # If embedding is not provided but query_text is, generate embedding
            if embedding is None and query_text is not None:
                embedding = self.create_embedding(query_text)
                if not embedding:
                    logger.error("Failed to generate embedding for query text")
                    return []

            assert embedding is not None, "Embedding should not be None here after checks."

            # Use the core RPC method that does the actual work with the embedding
            documents = self.find_similar_documents_via_rpc(
                embedding=embedding,  # First parameter is embedding
                session_id=None,  # Second parameter is session_id
                similarity_threshold=min_similarity,  # Third parameter is similarity_threshold
                limit=limit,  # Fourth parameter is limit
            )

            return documents

        except Exception as e:
            logger.error("Error finding similar documents: %s", e)
            logger.error(traceback.format_exc())
            return []

    def find_similar_documents_via_rpc(
        self,
        embedding: List[float],
        session_id: Optional[str] = None,
        limit: int = 3,
        similarity_threshold: float = 0.7,
    ) -> List[Dict]:
        """
        Find documents similar to the provided embedding using pgvector directly in PostgreSQL.
        This offloads computation from Python/GPU to the database.

        This function is typically called by:
        1. The high-level find_similar_documents function
        2. DynamicRAGRetriever.get_knowledge_by_query which passes:
        - embedding: The query embedding vector
        - session_id: The current session ID
        - similarity_threshold: The min_similarity parameter (not match_threshold)
        - limit: Max number of results to return

        Args:
            embedding: Vector embedding to compare against
            session_id: Optional session ID to filter documents by session
            similarity_threshold: Minimum similarity threshold (0-1)
            limit: Maximum number of results to return

        Returns:
            List of documents with similarity scores

        Note:
            This is the core similarity search function used throughout the application.
            Parameter names must remain consistent (similarity_threshold, not match_threshold).
        """
        try:
            # Format embedding for PostgreSQL pgvector format
            from psy_supabase.utilities.embedding_utils import format_embedding_for_db

            vector_str = format_embedding_for_db(embedding)

            # Determine the schema and table to search in
            schema_name = self.schema_name

            # Query the interactions and interaction_embeddings tables
            if session_id:
                # Filter by session_id in the dedicated column if it exists
                check_column_query = f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns
                    WHERE table_schema = '{schema_name}'
                    AND table_name = 'interactions'
                    AND column_name = 'session_id'
                );
                """

                check_response = self.supabase.rpc("sql", {"command": check_column_query}).execute()

                if check_response.data and str(check_response.data).lower().strip() in ("t", "true"):
                    # If session_id column exists, use it
                    query = f"""
                    SELECT
                        i.interaction_id as id,
                        i.question as content,
                        i.metadata,
                        1 - (ie.embedding <=> '{vector_str}'::vector) as similarity
                    FROM
                        {schema_name}.interactions i
                    JOIN
                        {schema_name}.interaction_embeddings ie
                    ON
                        i.interaction_id = ie.interaction_id
                    WHERE
                        1 - (ie.embedding <=> '{vector_str}'::vector) > {similarity_threshold}
                        AND i.session_id = '{session_id}'
                    ORDER BY
                        similarity DESC
                    LIMIT {limit};
                    """
                else:
                    # Fall back to metadata if no session_id column
                    query = f"""
                    SELECT
                        i.interaction_id as id,
                        i.question as content,
                        i.metadata,
                        1 - (ie.embedding <=> '{vector_str}'::vector) as similarity
                    FROM
                        {schema_name}.interactions i
                    JOIN
                        {schema_name}.interaction_embeddings ie
                    ON
                        i.interaction_id = ie.interaction_id
                    WHERE
                        1 - (ie.embedding <=> '{vector_str}'::vector) > {similarity_threshold}
                        AND i.metadata->>'session_id' = '{session_id}'
                    ORDER BY
                        similarity DESC
                    LIMIT {limit};
                    """
            else:
                # If no session_id, search in all interactions
                query = f"""
                SELECT
                    i.interaction_id as id,
                    i.question as content,
                    i.metadata,
                    1 - (ie.embedding <=> '{vector_str}'::vector) as similarity
                FROM
                    {schema_name}.interactions i
                JOIN
                    {schema_name}.interaction_embeddings ie
                ON
                    i.interaction_id = ie.interaction_id
                WHERE
                    1 - (ie.embedding <=> '{vector_str}'::vector) > {similarity_threshold}
                ORDER BY
                    similarity DESC
                LIMIT {limit};
                """

            logger.info("Finding similar documents in schema: %s", schema_name)
            response = self.supabase.rpc("sql", {"command": query}).execute()

            # Process the response to ensure proper formatting
            if response.data:
                results: List[Dict] = []
                for doc in response.data:
                    if isinstance(doc, dict):
                        # Parse metadata if it's a string
                        if "metadata" in doc and isinstance(doc["metadata"], str):
                            try:
                                doc["metadata"] = json.loads(doc["metadata"])
                            except:
                                doc["metadata"] = {}
                        results.append(doc)

                logger.info("Found %d similar documents", len(results))
                return results
            logger.warning("No similar documents found")
            return []

        except Exception as e:
            logger.error("Error finding similar documents via RPC: %s", e)
            logger.error(traceback.format_exc())
            return []

    def clean_db_text(self, text: Optional[str]) -> Optional[str]:
        """Clean text retrieved from the database to normalize quotes."""
        if text is None:
            return text
        return re.sub(r"'{2,}", "'", text)

    def connect_psychological_concepts(
        self, source_id: int, target_id: int, relationship_type: str = "related", strength: float = 1.0
    ) -> bool:
        """
        Create an explicit connection between two psychological concepts or memories.
        """
        try:
            # Call the database function directly
            response = self.supabase.rpc(
                "connect_psychological_concepts",
                {
                    "p_schema_name": self.schema_name,
                    "p_source_id": source_id,
                    "p_target_id": target_id,
                    "p_relationship_type": relationship_type,
                    "p_strength": strength,
                },
            ).execute()

            # Check if the response contains data
            if response.data is None or response.data == -1:
                logger.error("Error creating psychological connection")
                return False

            return True
        except Exception as e:
            logger.error("Error creating psychological connection: %s", e)
            return False

    def extract_psychological_themes(self, session_id: str, min_occurrences: int = 3) -> Dict[str, int]:
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
            themes: Dict[str, int] = {}
            for interaction in history:
                # Check metadata for explicit themes
                metadata = interaction.get("metadata", "{}")
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {}

                # Extract topic from metadata
                topic = metadata.get("topic", "")
                if topic:
                    themes[topic] = themes.get(topic, 0) + 1

                # Extract themes from question and answer content
                question = interaction.get("question", "")
                answer = interaction.get("answer", "")

                # Check content for theme keywords
                combined_text = (question + " " + answer).lower()
                for theme, keywords in self.theme_keywords.items():
                    if any(keyword in combined_text for keyword in keywords):
                        themes[theme] = themes.get(theme, 0) + 1

            # Return themes that occur at least min_occurrences times
            return {theme: count for theme, count in themes.items() if count >= min_occurrences}
        except Exception as e:
            logger.error("Error extracting psychological themes: %s", e)
            return {}

    def mark_therapeutic_insight(self, interaction_id: int, insight_level: str) -> bool:
        """
        Marks an interaction as containing a significant therapeutic insight.
        """
        try:
            response = self.supabase.rpc(
                "mark_therapeutic_insight",
                {
                    "p_schema_name": self.schema_name,
                    "p_interaction_id": interaction_id,
                    "p_insight_level": insight_level,
                },
            ).execute()

            return response.data is not None
        except Exception as e:
            logger.error("Error marking therapeutic insight: %s", e)
            return False

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
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
                return {"session_id": session_id, "error": "No conversation history found"}

            # Extract key data points
            themes = self.extract_psychological_themes(session_id, min_occurrences=1)
            emotions = self.analyze_emotional_vector_trajectory(session_id)

            # Extract insights if marked in metadata
            insights = []
            for interaction in history:
                metadata = {}
                if isinstance(interaction.get("metadata"), str):
                    try:
                        metadata = json.loads(interaction.get("metadata", "{}"))
                    except:
                        pass
                else:
                    metadata = interaction.get("metadata", {})

                if metadata and "insight_level" in metadata:
                    insights.append(
                        {
                            "interaction_id": interaction.get("interaction_id"),
                            "question": interaction.get("question", ""),
                            "answer": interaction.get("answer", ""),
                            "insight_level": metadata.get("insight_level"),
                            "created_at": interaction.get("created_at"),
                        }
                    )

            # Calculate session metrics
            start_time = min(entry.get("created_at", "") for entry in history) if history else ""
            end_time = max(entry.get("created_at", "") for entry in history) if history else ""

            # Count total interactions
            interaction_count = len(history)

            # Determine emotional shift if available
            emotional_shift = None
            if len(emotions) >= 2:
                starting_emotion = emotions[0].get("emotional_state", "")
                ending_emotion = emotions[-1].get("emotional_state", "")
                starting_intensity = emotions[0].get("intensity", 0)
                ending_intensity = emotions[-1].get("intensity", 0)

                emotional_shift = {
                    "starting_state": starting_emotion,
                    "ending_state": ending_emotion,
                    "intensity_change": ending_intensity - starting_intensity,
                }

            # Build the summary
            summary = {
                "session_id": session_id,
                "start_time": start_time,
                "end_time": end_time,
                "interaction_count": interaction_count,
                "primary_themes": sorted(themes.items(), key=lambda x: x[1], reverse=True)[:3] if themes else [],
                "emotional_shift": emotional_shift,
                "insights": insights,
            }

            return summary

        except Exception as e:
            logger.error("Error generating session summary: %s", e)
            return {"session_id": session_id, "error": f"Failed to generate summary: {str(e)}"}

    def analyze_theme_clusters(self, session_id: str, min_similarity: float = 0.7, max_clusters: int = 5) -> List[Dict]:
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
            response = self.supabase.rpc(
                "analyze_theme_clusters",
                {
                    "p_schema_name": self.schema_name,
                    "p_session_id": session_id,
                    "p_min_similarity": min_similarity,
                    "p_max_clusters": max_clusters,
                },
            ).execute()

            if response.data is None:
                logger.error("Error analysing theme clusters")
                return []

            return response.data
        except Exception as e:
            logger.error("Error analysing theme clusters: %s", e)
            return []

    def analyze_emotional_vector_trajectory(self, session_id: str) -> List[Dict]:
        """
        Analyze the emotional trajectory in vector space.

        Args:
            session_id: Session identifier to filter interactions

        Returns:
            List of emotional trajectory segments with vector analysis
        """
        try:
            # Call the pgvector emotional trajectory function with session_id parameter
            response = self.supabase.rpc(
                "analyze_emotional_vector_trajectory", {"p_schema_name": self.schema_name, "p_session_id": session_id}
            ).execute()

            # Check if the response contains data
            if response.data is None or len(response.data) == 0:
                logger.info("No emotional trajectory segments found for session: %s", session_id)
                return []

            return response.data
        except Exception as e:
            logger.error("Error analysing emotional vector trajectory for session %s: %s", session_id, e)
            return []

    def find_cross_session_patterns(self, session_ids: List[str]) -> List[Dict]:
        """
        Find psychological patterns that appear across multiple therapy sessions.

        Args:
            session_ids: List of session IDs to analyse

        Returns:
            List of patterns with occurrence data
        """
        try:
            # Call the pgvector cross-session pattern function
            response = self.supabase.rpc(
                "find_cross_session_patterns", {"p_user_schema": self.schema_name, "p_session_ids": session_ids}
            ).execute()

            if response.data is None:
                logger.error("Error finding cross-session patterns")
                return []

            return response.data
        except Exception as e:
            logger.error("Error finding cross-session patterns: %s", e)
            return []

    def ensure_vector_indexes(self, session_id: Optional[str] = None) -> bool:
        """
        Ensure vector indexes exist for efficient similarity searches.

        Args:
            session_id: Optional session ID (defaults to user schema)

        Returns:
            Boolean indicating success
        """
        try:
            return ensure_vector_indexes(self, self.schema_name)
        except Exception as e:
            logger.error("Error ensuring vector indexes: %s", e)
            return False

    def add_embedding_to_interaction(
        self, interaction_id: int, embedding: List[float], session_id: Optional[str] = None
    ) -> bool:
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
                # Use str(embedding) which preserves spaces after commas
                vector_str = str(embedding)
            elif hasattr(embedding, "tolist"):
                # Convert numpy array to list, then to string
                vector_str = str(embedding.tolist())
            else:
                # Already a string
                vector_str = embedding

            # Call the function to add embedding
            response = self.supabase.rpc(
                "add_embedding_to_interaction",
                {"p_schema_name": self.schema_name, "p_interaction_id": interaction_id, "p_embedding": vector_str},
            ).execute()

            return response.data is not None
        except Exception as e:
            logger.error("Error adding embedding to interaction: %s", e)
            return False

    def add_embedding_to_interactions(self, session_id: Optional[str] = None) -> bool:
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
            response = self.supabase.rpc("add_embedding_to_interactions", {"p_schema_name": self.schema_name}).execute()

            return response.data is not None
        except Exception as e:
            logger.error("Error adding embedding column: %s", e)
            return False

    def get_interactions_without_embeddings(self, session_id: Optional[str] = None, limit: int = 50) -> List[Dict]:
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
            response = self.supabase.rpc(
                "get_interactions_without_embeddings", {"p_schema_name": self.schema_name, "p_limit": limit}
            ).execute()

            if response.data is None:
                logger.error("Error getting interactions without embeddings")
                return []

            return response.data
        except Exception as e:
            logger.error("Error getting interactions without embeddings: %s", e)
            return []

    def enrich_interactions_with_embeddings(
        self, session_id: Optional[str] = None, model_name: str = TEXT_GENERATING_MODEL
    ) -> int:
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
                    interaction_id = interaction.get("interaction_id")
                    if interaction_id is None:
                        logger.warning("Skipping interaction with missing ID.")
                        continue
                    try:
                        interaction_id_int = int(interaction_id)
                    except (ValueError, TypeError):
                        logger.warning(f"Skipping interaction with invalid ID: {interaction_id}")
                        continue

                    question = interaction.get("question", "")
                    answer = interaction.get("answer", "")

                    # Generate embedding from combined text
                    combined_text = f"Question: {question}\nAnswer: {answer}"
                    embedding = embedding_provider.generate_embedding(combined_text)

                    if embedding:
                        # Add embedding to interaction using the validated int ID
                        # Pass schema_name explicitly if the method expects it
                        if self.add_embedding_to_interaction(
                            interaction_id_int, embedding
                        ):  # Removed schema_name if not needed by method def
                            enriched_count += 1
                except Exception as e:
                    # Use the validated int ID in the error message if available
                    log_id = interaction.get("interaction_id", "unknown")
                    logger.error("Error enriching interaction %s: %s", log_id, e)
                    continue

            return enriched_count
        except Exception as e:
            logger.error("Error enriching interactions: %s", e)
            return 0

    def update_table_statistics(self, session_id: Optional[str] = None) -> bool:
        """
        Update table statistics for better query planning.
        This helps the database query planner make better decisions.

        Args:
            session_id: Optional session ID (defaults to user schema)

        Returns:
            bool: Success status
        """
        try:
            return update_table_statistics(self, self.schema_name)
        except Exception as e:
            logger.error("Error updating table statistics: %s", e)
            return False

    def optimize_vector_operations(self, session_id: Optional[str] = None) -> Dict[str, Any]:
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
            # Use the optimized vector utility function
            return optimize_vectors(self, self.schema_name)
        except Exception as e:
            logger.error("Error optimizing vector operations: %s", e)
            return {
                "column_added": False,
                "indexes_created": False,
                "interactions_enriched": 0,
                "statistics_updated": False,
                "error": str(e),
            }

    def find_similar_question_embedding(
        self, question_text: str, session_id: str, similarity_threshold: float = 0.92
    ) -> Optional[List[float]]:
        """Find embedding of a similar previous question to avoid regenerating embeddings."""
        try:
            # Generate embedding for the current question
            question_embedding = self.create_embedding(question_text)
            if not question_embedding:
                logger.warning("Failed to create embedding for question text")
                return None

            # Format the embedding for PostgreSQL
            from psy_supabase.utilities.embedding_utils import format_embedding_for_db

            vector_str = format_embedding_for_db(question_embedding)

            # Call the find_similar_interactions function with ALL required parameters
            response = self.supabase.rpc(
                "find_similar_interactions",
                {
                    "p_schema_name": self.schema_name,
                    "p_embedding": vector_str,
                    "p_session_id": session_id,
                    "p_threshold": similarity_threshold,
                    "p_limit": 1,
                },
            ).execute()

            if response.data and len(response.data) > 0:
                # Process the response
                if isinstance(response.data[0], dict):
                    interaction_id = response.data[0].get("interaction_id")
                    embedding_str = response.data[0].get("embedding")
                    if embedding_str is None:
                        logger.warning("No embedding string found for interaction %s", interaction_id)
                        return None
                    if interaction_id:
                        # Convert the string to a list of floats
                        try:
                            embedding_list = json.loads(embedding_str.strip())
                            logger.info("Found similar question with embedding (length: %d)", len(embedding_list))
                            return embedding_list
                        except Exception as e:
                            logger.error("Error converting embedding string to list: %s", e)
                            return None

            # No similar question found
            logger.info("No similar question found with threshold %.2f", similarity_threshold)
            return None

        except Exception as e:
            logger.error("Error finding similar question embedding: %s", e)
            logger.error(traceback.format_exc())
            return None

    def find_similar_question_embedding_(
        self, question_text: str, session_id: str, similarity_threshold: float = 0.92
    ) -> Optional[List[float]]:
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
                    i.interaction_id as interaction_id,
                    i.question,
                    ie.embedding
                FROM
                    {self.schema_name}.interactions i
                JOIN
                    {self.schema_name}.interaction_embeddings ie
                ON
                    i.interaction_id = ie.interaction_id
                WHERE
                    calculate_vector_similarity(lower(i.question), '{normalized_question}') > {similarity_threshold}
                ORDER BY
                    calculate_vector_similarity(lower(i.question), '{normalized_question}') DESC
                LIMIT 1
            )
            SELECT
                interaction_id,
                question,
                embedding::text as embedding_text
            FROM
                question_interactions;
            """

            response = self.supabase.rpc("sql", {"command": query}).execute()

            if response.data and len(response.data) > 0:
                # PostgreSQL might be returning rows as strings, not dictionaries
                # Handle both cases

                # Case 1: If response.data[0] is a dictionary (normal case)
                if isinstance(response.data[0], dict):
                    embedding_str = response.data[0].get("embedding_text", None)

                    if embedding_str:
                        # Convert the string to a list of floats
                        try:
                            # Remove brackets and split by commas
                            embedding_list = [float(val) for val in embedding_str.strip("[]").split(",")]
                            logger.info("Found similar question with embedding (length: %d)", len(embedding_list))
                            return embedding_list
                        except Exception as e:
                            logger.error("Error converting embedding string to list: %s", e)
                            return None

                # Case 2: If response.data[0] is a string (CSV-like format)
                elif isinstance(response.data[0], str):
                    try:
                        # Format could be something like: "123,What is your name?,{0.1,0.2,...}"
                        parts = response.data[0].split(",", 2)  # Split into 3 parts
                        if len(parts) >= 3:
                            embedding_str = parts[2]  # Get the embedding part

                            # Now handle the embedding text format
                            # Remove any extra brackets and split
                            embedding_str = embedding_str.strip("{}[]").replace("{", "").replace("}", "")
                            embedding_list = [float(val) for val in embedding_str.split(",")]

                            logger.info(
                                "Found similar question with embedding from string format (length: %d)",
                                len(embedding_list),
                            )
                            return embedding_list
                    except Exception as e:
                        logger.error("Error parsing string result: %s", e)
                        return None

            # No similar question found
            return None

        except Exception as e:
            logger.error("Error finding similar question embedding: %s", e)
            logger.error(traceback.format_exc())
            return None

    @typechecked
    def find_similar_interactions_by_embedding(
        self, embedding: List[float], session_id: Optional[str] = None, limit: int = 5, threshold: float = 0.7
    ) -> List[Dict]:
        """
        Find interactions with similar embeddings using pgvector.

        Args:
            embedding: Query embedding vector
            session_id: Optional session ID to restrict search
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of similar interactions with similarity scores
        """
        try:
            # Use direct SQL execution
            response = self.supabase.rpc(
                "find_similar_interactions",
                {
                    "p_schema_name": self.schema_name,
                    "p_embedding": embedding,
                    "p_session_id": session_id,
                    "p_threshold": threshold,
                    "p_limit": limit,
                },
            ).execute()

            if not response.data:
                return []

            # Process results
            results = []
            for item in response.data:
                if isinstance(item, dict):
                    question = item.get("question", "")
                    answer = item.get("answer", "")

                    results.append(
                        {
                            "interaction_id": item.get("interaction_id", 0),
                            "question": self.clean_db_text(question),
                            "answer": self.clean_db_text(answer),
                            "created_at": item.get("created_at", ""),
                            "metadata": item.get("metadata", {}),
                            "session_id": item.get("session_id"),
                            "similarity": item.get("similarity", 0),
                        }
                    )
                elif isinstance(item, str):
                    # Try to handle string format (very rare)
                    parts = item.split(",")
                    if len(parts) >= 6:
                        results.append(
                            {
                                "interaction_id": int(parts[0]) if parts[0].strip().isdigit() else 0,
                                "question": parts[1].strip() if len(parts) > 1 else "",
                                "answer": parts[2].strip() if len(parts) > 2 else "",
                                "created_at": parts[3].strip() if len(parts) > 3 else "",
                                "metadata": parts[4].strip() if len(parts) > 4 else {},
                                "session_id": parts[5].strip() if len(parts) > 5 else None,
                                "similarity": (
                                    float(parts[6])
                                    if len(parts) > 6 and parts[6].strip().replace(".", "").isdigit()
                                    else 0
                                ),
                            }
                        )

            return results
        except Exception as e:
            logger.error("Error finding similar interactions by embedding: %s", e)
            logger.error(traceback.format_exc())
            return []

    @debug_errors(logger=logger)
    def analyze_emotional_response_to_interaction(self, session_id: str, interaction_id: int) -> Dict[str, Any]:
        """
        Analyze emotional response in interactions with robust type handling.

        Safely analyzes emotional content in user interactions, handling various input types
        including lists, dictionaries, and custom objects. Implements defensive programming
        techniques to prevent common errors like "'list' object has no attribute 'get'".

        The method:
        1. Performs type checking on input data
        2. Handles both single interactions and lists of interactions
        3. Safely extracts question and answer text using appropriate access methods
        4. Delegates detailed emotion analysis to the prompt_selector
        5. Provides graceful fallbacks and detailed error logging

        Args:
            interaction_data: Data from the interaction (can be dict, list, or custom object)

        Returns:
            Dict with emotional analysis containing:
            - emotion: Detected emotion (string)
            - intensity: Emotion intensity score (float 0.0-1.0)
        """
        try:
            # Get the interaction data
            history = self.get_conversation_history(session_id)

            # Return default values if history is empty or invalid
            if not history:
                logger.warning(f"Empty history for session {session_id}")
                return {"emotion": DEFAULT_EMOTION, "quality": 0.5}

            logger.debug(f"Found {len(history)} interactions for session {session_id}")

            # Find the specific interaction
            target_interaction = None
            response_text = ""

            for interaction in history:
                interaction_id_value = None

                # Check what format we're dealing with
                if isinstance(interaction, dict):
                    # Dictionary format
                    interaction_id_value = interaction.get("id")
                    if interaction_id_value == interaction_id:
                        response_text = interaction.get("response", "")
                        target_interaction = interaction
                        break
                elif isinstance(interaction, list) and len(interaction) > 0:
                    # List format
                    interaction_id_value = interaction[0]
                    if interaction_id_value == interaction_id and len(interaction) > 2:
                        response_text = interaction[2]
                        target_interaction = {
                            "id": interaction[0],
                            "question": interaction[1] if len(interaction) > 1 else "",
                            "response": interaction[2] if len(interaction) > 2 else "",
                            "timestamp": interaction[3] if len(interaction) > 3 else "",
                        }
                        break

            if not target_interaction:
                logger.warning(f"Interaction {interaction_id} not found in history for session {session_id}")
                return {"emotion": DEFAULT_EMOTION, "quality": 0.5}

            # Perform analysis on response_text
            # For demonstration, return simple metrics
            word_count = len(response_text.split())
            quality = min(0.9, max(0.1, word_count / 50))  # Simple metric based on length

            return {
                "emotion": "empathetic",  # Could be replaced with sentiment analysis
                "quality": quality,
                "length": len(response_text),
            }

        except Exception as e:
            logger.error(f"Error analysing emotional response to interaction: {str(e)}")
            # Return default values in case of error
            return {"emotion": DEFAULT_EMOTION, "quality": 0.5}

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

            response = self.supabase.rpc("sql", {"command": query}).execute()

            if not response.data:
                return []

            # Process and return results
            insights: List[Dict[str, Any]] = []
            for item in response.data:
                if isinstance(item, str):
                    # Parse CSV-like string
                    values = item.split(",")
                    if len(values) >= 5:
                        insights.append(
                            {
                                "interaction_id": values[0],
                                "question": values[1],
                                "answer": values[2],
                                "insight_level": values[3],
                                "created_at": values[4],
                            }
                        )
                elif isinstance(item, dict):
                    insights.append(
                        {
                            "interaction_id": item.get("interaction_id"),
                            "question": item.get("question", ""),
                            "answer": item.get("answer", ""),
                            "insight_level": item.get("insight_level", ""),
                            "created_at": item.get("created_at", ""),
                        }
                    )

            return insights
        except Exception as e:
            logger.error("Error getting therapeutic insights: %s", e)
            return []

    @typechecked
    def identify_potential_pain_points(
        self, question_text: str, question_embedding: List[float], session_id: str, pain_threshold: float = 0.85
    ) -> Dict:
        """
        Identifies potential psychological pain points by analysing the current question
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
                return {"session_id": session_id, "error": "No conversation history found"}

            # Skip if less than 2 interactions (not enough history to identify patterns)
            if len(history) < 2:
                logger.debug("Not enough history to identify pain points (%d interactions)", len(history))
                return {}

            # Get embeddings for past questions using optimized pgvector search
            past_questions = []
            for item in history:
                interaction_id = item.get("interaction_id")
                past_question = item.get("question", "")
                if past_question and interaction_id and past_question != question_text:
                    past_questions.append(
                        {"id": interaction_id, "text": past_question, "created_at": item.get("created_at")}
                    )

            # No past questions to analyse
            if not past_questions:
                return {}

            # Find similar questions from history with pgvector
            similar_questions = []

            # 1. First method: Use existing embeddings through pgvector
            vector_similar_questions = self.find_similar_interactions_by_embedding(
                embedding=question_embedding,
                session_id=session_id,
                limit=3,
                threshold=0.7,  # Lower threshold to get candidates
            )

            # 2. Second method: Check for linguistic/semantic similarity using text
            # This adds a different dimension to similarity detection
            for past_q in past_questions:
                # Simple text similarity check (ratio of common words)
                past_words = set(re.findall(r"\b\w+\b", past_q["text"].lower()))
                current_words = set(re.findall(r"\b\w+\b", question_text.lower()))

                if past_words and current_words:
                    # Calculate Jaccard similarity
                    intersection = past_words.intersection(current_words)
                    union = past_words.union(current_words)
                    text_similarity = len(intersection) / len(union) if union else 0

                    # Add if text similarity is high enough
                    if text_similarity > 0.3:  # Lower threshold for text similarity
                        vector_similar_questions.append(
                            {
                                "interaction_id": past_q["id"],
                                "question": past_q["text"],
                                "similarity": text_similarity,
                                "method": "text",
                            }
                        )

            # No similar questions found with either method
            if not vector_similar_questions:
                return {}

            # Combine and deduplicate results
            seen_ids = set()
            for item in vector_similar_questions:
                interaction_id = item.get("interaction_id") or item.get("interaction_id")
                if interaction_id and interaction_id not in seen_ids:
                    seen_ids.add(interaction_id)
                    similar_questions.append(
                        {
                            "id": interaction_id,
                            "text": item.get("question"),
                            "similarity": item.get("similarity", 0),
                        }
                    )

            # Sort by similarity (highest first)
            similar_questions = sorted(similar_questions, key=lambda x: x["similarity"], reverse=True)

            # Find the highest similarity score
            highest_similarity = similar_questions[0]["similarity"] if similar_questions else 0

            # If we have a very high similarity, we've identified a potential pain point
            if highest_similarity >= pain_threshold:
                most_similar = similar_questions[0]

                # Extract emotional patterns related to this interaction
                emotions = self.analyze_emotional_response_to_interaction(
                    session_id=session_id, interaction_id=most_similar["id"]
                )

                # Extract therapeutic insights if available
                insights = self.get_therapeutic_insights_for_interaction(
                    interaction_id=most_similar["id"], session_id=session_id
                )

                # Find how many times similar questions have been asked
                repetition_pattern = detect_repetition_pattern(most_similar["text"], question_text, similar_questions)

                # Look for recurring terms in the repetition pattern
                primary_theme = DEFAULT_THEME
                if repetition_pattern and "recurring_terms" in repetition_pattern:
                    recurring_terms = repetition_pattern.get("recurring_terms", [])
                    if recurring_terms and isinstance(recurring_terms, list) and len(recurring_terms) > 0:
                        # Ensure the first element is a string before assigning
                        first_term = recurring_terms[0]
                        if isinstance(first_term, str):
                            primary_theme = first_term
                            logger.info("Extracted primary theme '%s' from repetition pattern", primary_theme)
                        else:
                            logger.warning("First recurring term is not a string: %s", first_term)

                # If we couldn't get a theme from repetition pattern, extract from the question text
                if primary_theme == DEFAULT_THEME:
                    # Check for themes in the question text
                    for theme, keywords in self.theme_keywords.items():
                        if any(keyword in question_text.lower() for keyword in keywords):
                            primary_theme = theme
                            logger.info("Extracted primary theme '%s' from keywords", primary_theme)
                            break

                    # If still using default, check the original question text too
                    if primary_theme == DEFAULT_THEME:
                        for theme, keywords in self.theme_keywords.items():
                            if any(keyword in most_similar["text"].lower() for keyword in keywords):
                                primary_theme = theme
                                logger.info("Extracted primary theme '%s' from original question", primary_theme)
                                break

                logger.info(
                    "Identified potential pain point with primary theme '%s' and similarity score %.2f",
                    primary_theme,
                    highest_similarity,
                )

                return {
                    "detected": True,
                    "similarity": highest_similarity,
                    "original_question": most_similar["text"],
                    "current_question": question_text,
                    "interaction_id": most_similar["id"],
                    "emotions": emotions,
                    "insights": insights,
                    "repetition_pattern": repetition_pattern,
                    "suggested_approach": {
                        "name": map_approach_name(primary_theme),
                        "approach_type": map_theme_to_approach_type(primary_theme),
                    },
                }

            # No pain point detected
            return {}

        except Exception as e:
            logger.error("Error identifying potential pain points: %s", e)
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
                i.interaction_id,
                i.embedding
            FROM
                {self.schema_name}.interactions i
            LEFT JOIN
                {self.schema_name}.interaction_embeddings ie
            ON
                i.interaction_id = ie.interaction_id
            WHERE
                i.embedding IS NOT NULL
                AND ie.interaction_id IS NULL
            RETURNING id;
            """

            response = self.supabase.rpc("sql", {"command": query}).execute()

            if response.data:
                if isinstance(response.data, list):
                    count = len(response.data)
                count = 1

                logger.info("Migrated %d embeddings to interaction_embeddings table", count)
                return count
            logger.info("No embeddings to migrate")
            return 0

        except Exception as e:
            logger.error("Error migrating embeddings to interaction_embeddings table: %s", e)
            logger.error(traceback.format_exc())
            return 0

    @typechecked
    def create_embedding(self, text: str) -> Optional[List[float]]:
        """
        Create an embedding for the given text using the appropriate embedding provider.

        Args:
            text: Text to create embedding for

        Returns:
            Optional[List[float]]: Embedding vector or None if an error occurs
        """
        try:
            # Get the embedding provider
            embedding_provider = get_embedding_provider()

            # Generate embedding
            embedding = embedding_provider.generate_embedding(text)

            # Convert to proper format if needed
            if hasattr(embedding, "tolist") and callable(getattr(embedding, "tolist")):
                embedding = embedding.tolist()

            # Ensure it's a list of floats before returning
            if isinstance(embedding, list) and all(isinstance(x, (float, int)) for x in embedding):
                return [float(x) for x in embedding]
            logger.error("Generated embedding is not a valid list of floats.")
            return None

        except Exception as e:
            logger.error("Error creating embedding: %s", e)
            logger.error(traceback.format_exc())
            return None

    def save_interaction(
        self,
        context: str,
        question: str,
        answer: str,
        metadata: Optional[Dict] = None,
        session_id: Optional[str] = None,
    ) -> bool:
        """Save an interaction to the database with proper metadata handling."""
        try:
            # Ensure schema exists
            logger.info("Ensuring schema exists for user: %s", self.user_id)
            self.create_user_schema_sync()

            # Prepare metadata
            if metadata is None:
                metadata = {}

            # Always add session_id to metadata explicitly
            if session_id is not None:
                metadata["session_id"] = session_id
                logger.info("Adding session_id to metadata: %s", session_id)

            # Clean the text data
            clean_question = self._clean_text_for_db(question)
            clean_answer = self._clean_text_for_db(answer)

            # Convert metadata to JSON string
            if isinstance(metadata, dict):
                metadata_str = json.dumps(metadata)
            else:
                metadata_str = str(metadata)

            logger.info("Calling add_interaction RPC with schema: %s, session_id: %s", self.schema_name, session_id)

            # Add the interaction using our updated RPC function
            response = self.supabase.rpc(
                "add_interaction",
                {
                    "p_schema_name": self.schema_name,
                    "p_context": context,
                    "p_question": clean_question,
                    "p_answer": clean_answer,
                    "p_metadata": metadata_str,
                    "p_session_id": session_id,
                },
            ).execute()

            # Check response
            logger.info("RPC response: %s", response.data)

            if response.data is not None:
                try:
                    interaction_id = int(response.data)
                    logger.info("Interaction saved successfully with ID: %d", interaction_id)
                except (ValueError, TypeError):
                    logger.error("Could not parse interaction ID from response: %s", response.data)
                    return False

            # Generate embedding for the question if needed
            if clean_question:
                logger.info("Generating embedding for question")
                question_embedding = self.create_embedding(clean_question)

                if question_embedding:
                    # Format the embedding for PostgreSQL
                    from psy_supabase.utilities.embedding_utils import format_embedding_for_db

                    embedding_str = format_embedding_for_db(question_embedding)

                    # Store in interaction_embeddings table
                    logger.info("Storing embedding for interaction %d", interaction_id)

                    embed_response = self.supabase.rpc(
                        "add_embedding_to_interaction",
                        {
                            "p_schema_name": self.schema_name,
                            "p_interaction_id": interaction_id,
                            "p_embedding": embedding_str,
                        },
                    ).execute()

                    if embed_response.data is True:
                        logger.info("Embedding stored successfully for interaction %d", interaction_id)
                    else:
                        logger.warning("Embedding storage function returned: %s", embed_response.data)
                else:
                    logger.warning("No embedding generated for question")

            return True

        except Exception as e:
            logger.error("Error saving interaction: %s", e)
            logger.error(traceback.format_exc())
            return False

    @typechecked
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
        text = text.replace("\0", "")

        # Optional: Truncate extremely long texts
        max_length = 65000  # PostgreSQL TEXT can handle much more, but this is safer
        if len(text) > max_length:
            text = text[:max_length]

        return text

    def detect_pain_points(self, session_id: str, threshold: float = 0.7, min_occurrences: int = 2) -> Dict:
        """
        Detect pain points from conversation history using vector similarity.

        Args:
            session_id: The session ID to analyse
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
                return {"pain_points": [], "severity": "none", "first_detected_at": None}

            # Extract questions and convert to vectors
            questions = [item.get("question", "") for item in history]

            # Track clusters of similar questions
            question_clusters: List[Dict[str, Any]] = []

            # For each question, check if it forms a cluster with others
            for primary_idx, question in enumerate(questions):
                # Skip empty questions
                if not question.strip():
                    continue

                # Get the vector for this question
                primary_embedding = self.create_embedding(question)
                if not primary_embedding:
                    # Skip this question if embedding generation fails
                    logger.warning(f"Failed to generate embedding for question at index {primary_idx}")
                    continue

                # Search for similar questions in the conversation
                similar_indices = []

                for compare_idx, other_question in enumerate(questions):
                    if primary_idx == compare_idx:  # Skip comparing to self
                        continue

                    if not other_question.strip():
                        continue

                    # Generate embedding for comparing question
                    compare_embedding = self.create_embedding(other_question)
                    if not compare_embedding:
                        continue

                    # Use proper vector similarity via Supabase
                    similar_items = self.find_similar_interactions_by_embedding(
                        embedding=primary_embedding, session_id=session_id, limit=1, threshold=threshold
                    )

                    # Check if the compared question is returned as similar
                    if similar_items and len(similar_items) > 0:
                        # Extract the similarity score from the result
                        similarity = similar_items[0].get("similarity", 0)

                        if similarity > threshold:
                            similar_indices.append(compare_idx)

                # If we found enough similar questions, we have a cluster
                if len(similar_indices) + 1 >= min_occurrences:  # +1 to include the current question
                    cluster: Dict[str, Any] = {
                        "indices": [primary_idx] + similar_indices,  # Type is List[int]
                        "questions": [questions[primary_idx]] + [questions[s_i] for s_i in similar_indices],
                        "recurring_terms": self._extract_recurring_terms(
                            [questions[primary_idx]] + [questions[s_i] for s_i in similar_indices]
                        ),
                        "first_occurrence": min([primary_idx] + similar_indices),
                        "count": len(similar_indices) + 1,
                    }

                    # Check if this cluster overlaps significantly with an existing one
                    is_new_cluster = True
                    for existing in question_clusters:
                        # Explicitly get the lists - mypy should infer List[int] now
                        cluster_indices: List[int] = cluster["indices"]
                        existing_indices: List[int] = existing["indices"]

                        # Create sets - mypy should be happy now
                        overlap = len(set(cluster_indices).intersection(set(existing_indices)))

                        # If more than 50% overlap, consider it the same cluster
                        if overlap > len(cluster_indices) / 2:
                            is_new_cluster = False
                            break

                    if is_new_cluster:
                        question_clusters.append(cluster)

            # Calculate overall pain point metrics
            pain_points = sorted(question_clusters, key=lambda x: x["first_occurrence"])

            # Determine severity based on cluster counts and sizes
            total_questions = len(questions)
            if not pain_points:
                severity = "none"
            elif sum(p["count"] for p in pain_points) > total_questions * 0.7:
                severity = "high"
            elif sum(p["count"] for p in pain_points) > total_questions * 0.4:
                severity = "medium"
            else:
                severity = "low"

            # Find the first detected pain point
            first_detected_at = min(p["first_occurrence"] for p in pain_points) if pain_points else None

            return {"pain_points": pain_points, "severity": severity, "first_detected_at": first_detected_at}

        except Exception as e:
            logger.error("Error detecting pain points: %s", e)
            return {"pain_points": [], "severity": "none", "first_detected_at": None}

    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple text similarity for testing purposes.
        In production, use actual vector embeddings.
        """
        # Simple word overlap calculation for testing
        words1 = set(re.findall(r"\b\w+\b", text1.lower()))
        words2 = set(re.findall(r"\b\w+\b", text2.lower()))

        if not words1 or not words2:
            return 0.0

        overlap = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return overlap / union if union > 0 else 0.0

    def _extract_recurring_terms(self, texts: List[str]) -> List[str]:
        """
        Extract common terms from a set of texts that might indicate pain points.
        """
        from psy_supabase.utilities.stop_words import stop_words

        # Combine all texts
        combined = " ".join(texts)

        # Extract words and count frequencies
        words = re.findall(r"\b\w+\b", combined.lower())  # Ensure lowercase for consistency
        word_counts: Dict[str, int] = {}

        for word in words:
            # Skip stop words and very short words
            if len(word) <= 2 or word in stop_words:
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

    def get_recommended_therapeutic_approach(self, pain_point: Dict[Any, Any]) -> Dict[str, Any]:
        """
        Get a recommended therapeutic approach for a detected pain point
        based on recurring terms and a predefined dictionary of approaches.

        Args:
            pain_point: Dictionary containing details of the detected pain point,
                        including 'recurring_terms'.

        Returns:
            Dict: Dictionary with approach information (name, technique, strategy, questions).
        """
        try:
            # Define the available approaches directly within the function
            # Keys are lowercase trigger terms.
            # 'name' should be a canonical approach name (often used by map_approach_to_template).
            # 'primary_technique' is a short identifier for the core technique.
            approaches = {
                "anxiety": {
                    "name": "cognitive_behavioral_therapy",
                    "primary_technique": "cbt",
                    "redirection_strategy": "Explore anxiety triggers, cognitive distortions, and develop coping mechanisms.",
                    "exploration_questions": "What physical sensations do you notice when anxious? What thoughts typically accompany these feelings?",
                },
                "panic": {
                    "name": "cognitive_behavioral_therapy",  # Or specific panic_control_treatment
                    "primary_technique": "cbt",
                    "redirection_strategy": "Psychoeducation about panic, identify catastrophic thoughts, and practice coping skills.",
                    "exploration_questions": "What are your main fears when you experience panic? What has helped, even a little, in the past?",
                },
                "worried": {
                    "name": "worry_management_cbt",
                    "primary_technique": "cbt",
                    "redirection_strategy": "Examine evidence for and against worries, differentiate productive vs. unproductive worry, and schedule 'worry time'.",
                    "exploration_questions": "How likely is this worry to come true? What's the worst that could happen, and how would you cope if it did?",
                },
                "fear": {  # General fear, could be phobia related
                    "name": "exposure_therapy",  # Or CBT if more general
                    "primary_technique": "exposure",
                    "redirection_strategy": "Gradual exposure to feared situations or stimuli, coupled with relaxation techniques.",
                    "exploration_questions": "What specific situations or things trigger this fear? What do you typically do to avoid it?",
                },
                "depressed": {
                    "name": "behavioral_activation_cbt",
                    "primary_technique": "behavioral_activation",
                    "redirection_strategy": "Identify and schedule pleasant or mastery-oriented activities to counteract withdrawal and improve mood.",
                    "exploration_questions": "What activities used to bring you joy or a sense of accomplishment? What's one small step you could take towards re-engaging?",
                },
                "sad": {  # General sadness, might be less clinical than "depressed"
                    "name": "supportive_listening_mood",  # Or behavioral_activation if persistent
                    "primary_technique": "supportive_listening",
                    "redirection_strategy": "Validate feelings of sadness, explore its context, and identify potential coping strategies.",
                    "exploration_questions": "Can you tell me more about what's making you feel sad? What usually helps you when you feel this way?",
                },
                "hopeless": {
                    "name": "cognitive_behavioral_therapy",  # Often linked with depression
                    "primary_technique": "cbt",
                    "redirection_strategy": "Challenge hopeless thoughts, identify exceptions, and build a sense of agency or hope.",
                    "exploration_questions": "What makes you feel hopeless right now? Have there been times when you felt differently, even slightly?",
                },
                "trauma": {
                    "name": "trauma_informed",
                    "primary_technique": "trauma_informed",
                    "redirection_strategy": "Prioritize safety, grounding, psychoeducation about trauma, and validate experiences.",
                    "exploration_questions": "What helps you feel safe and grounded in this moment? How are these past experiences affecting you today?",
                },
                "ptsd": {
                    "name": "trauma_informed",
                    "primary_technique": "trauma_informed",
                    "redirection_strategy": "Focus on managing PTSD symptoms like flashbacks and hyperarousal, using grounding and coping skills.",
                    "exploration_questions": "Are there specific triggers for your PTSD symptoms? What coping strategies have you found helpful?",
                },
                "abuse": {
                    "name": "trauma_informed",
                    "primary_technique": "trauma_informed",
                    "redirection_strategy": "Validate the experience of abuse, focus on safety, and explore its impact on current well-being.",
                    "exploration_questions": "How has this experience of abuse impacted you? What does safety mean to you now?",
                },
                "grief": {
                    "name": "grief_processing",
                    "primary_technique": "grief_processing",
                    "redirection_strategy": "Provide space for expressing grief, validate feelings, explore meaning-making and coping with the loss.",
                    "exploration_questions": "What does this loss mean to you? What are some of the most challenging aspects of this grief right now?",
                },
                "loss": {
                    "name": "grief_processing",
                    "primary_technique": "grief_processing",
                    "redirection_strategy": "Acknowledge the pain of loss, explore the different emotions involved, and support adaptation.",
                    "exploration_questions": "How are you coping with this loss day-to-day? Are there any rituals or memories you find comforting?",
                },
                "relationship": {
                    "name": "interpersonal_therapy",
                    "primary_technique": "interpersonal",
                    "redirection_strategy": "Explore patterns in relationships, communication styles, unmet needs, and role transitions.",
                    "exploration_questions": "What are the recurring themes in your relationship conflicts? What are your needs and expectations in a relationship?",
                },
                "partner": {  # More specific than "relationship"
                    "name": "interpersonal_therapy",
                    "primary_technique": "interpersonal",
                    "redirection_strategy": "Focus on the dynamics with the specific partner, communication, and shared goals or conflicts.",
                    "exploration_questions": "Can you describe a recent interaction with your partner that was challenging? What would you like to be different in this relationship?",
                },
                "conflict": {  # General conflict
                    "name": "problem_solving_therapy",  # Or interpersonal if relationship-focused
                    "primary_technique": "problem_solving",
                    "redirection_strategy": "Identify the core issues of the conflict, explore different perspectives, and brainstorm solutions.",
                    "exploration_questions": "What is the main point of disagreement in this conflict? What outcomes are you hoping for?",
                },
                "alone": {
                    "name": "attachment_based_therapy",
                    "primary_technique": "attachment",
                    "redirection_strategy": "Explore feelings of loneliness, attachment patterns, and fears of abandonment or disconnection.",
                    "exploration_questions": "What does being alone bring up for you? Can you describe your typical patterns in close relationships?",
                },
                "lonely": {
                    "name": "attachment_based_therapy",
                    "primary_technique": "attachment",
                    "redirection_strategy": "Validate feelings of loneliness and explore underlying needs for connection and belonging.",
                    "exploration_questions": "In what situations do you feel most lonely? What kind of connections are you seeking?",
                },
                "failure": {
                    "name": "compassion_focused_therapy",
                    "primary_technique": "cft",
                    "redirection_strategy": "Challenge self-critical thoughts related to perceived failure and cultivate self-compassion.",
                    "exploration_questions": "How do you typically respond to yourself when you feel you've failed? What would a compassionate perspective be?",
                },
                "self-esteem": {
                    "name": "compassion_focused_therapy",  # Or strengths_based
                    "primary_technique": "cft",
                    "redirection_strategy": "Identify and challenge negative self-beliefs, focus on strengths, and practice self-acceptance.",
                    "exploration_questions": "What are some qualities you value in yourself? When do you feel most confident or capable?",
                },
                "worthless": {
                    "name": "compassion_focused_therapy",
                    "primary_technique": "cft",
                    "redirection_strategy": "Address core beliefs of worthlessness with self-compassion and by examining evidence.",
                    "exploration_questions": "Where do you think this feeling of worthlessness comes from? Can you think of times you've acted in line with your values despite these feelings?",
                },
                "guilt": {
                    "name": "cognitive_behavioral_therapy_guilt",  # Or CFT
                    "primary_technique": "cbt",
                    "redirection_strategy": "Explore the source of guilt, differentiate appropriate vs. inappropriate guilt, and work towards self-forgiveness or making amends.",
                    "exploration_questions": "What specific actions or inactions are you feeling guilty about? What are your values related to this situation?",
                },
                "shame": {
                    "name": "compassion_focused_therapy",
                    "primary_technique": "cft",
                    "redirection_strategy": "Normalize shame as a human emotion, reduce self-criticism, and build shame resilience through self-compassion.",
                    "exploration_questions": "Shame can be a very powerful emotion. How does it show up for you? What might help you meet this feeling with kindness?",
                },
                "stress": {
                    "name": "mindfulness_based_stress_reduction",
                    "primary_technique": "mindfulness",
                    "redirection_strategy": "Identify stressors, develop mindfulness skills to manage stress, and increase awareness of present moment reactions.",
                    "exploration_questions": "What are your main stressors currently? Have you explored mindfulness or relaxation techniques before?",
                },
                "overwhelmed": {
                    "name": "problem_solving_therapy",  # Or mindfulness
                    "primary_technique": "problem_solving",
                    "redirection_strategy": "Break down overwhelming situations into manageable parts, prioritize, and brainstorm coping strategies.",
                    "exploration_questions": "What specific things are contributing to this feeling of being overwhelmed? What's one small thing you could address first?",
                },
                "work": {  # General work-related stress
                    "name": "workplace_stress_management",
                    "primary_technique": "problem_solving",  # Or CBT
                    "redirection_strategy": "Identify specific workplace challenges, explore coping strategies, problem-solving, or boundary setting.",
                    "exploration_questions": "Can you describe the specific situations at work causing distress? What aspects are within your control to change or influence?",
                },
                "job": {  # Similar to "work"
                    "name": "workplace_stress_management",
                    "primary_technique": "problem_solving",
                    "redirection_strategy": "Focus on job-specific stressors, career satisfaction, and work-life balance.",
                    "exploration_questions": "What aspects of your job are most challenging right now? What would make your work life feel more manageable or fulfilling?",
                },
                "boss": {  # Specific to authority figures at work
                    "name": "workplace_conflict_resolution",
                    "primary_technique": "communication_skills",  # Or assertiveness training
                    "redirection_strategy": "Explore dynamics with the boss, develop communication strategies, and address power imbalances if relevant.",
                    "exploration_questions": "Can you describe a recent interaction with your boss that was difficult? What are your communication goals in these situations?",
                },
                "addiction": {
                    "name": "motivational_interviewing",
                    "primary_technique": "mi",
                    "redirection_strategy": "Explore ambivalence about change, build motivation, support self-efficacy, and discuss harm reduction or recovery goals.",
                    "exploration_questions": "What are some of the reasons you're considering a change regarding this? What does a positive change look like to you?",
                },
                "substance": {
                    "name": "motivational_interviewing",
                    "primary_technique": "mi",
                    "redirection_strategy": "Non-judgmentally explore substance use patterns, motivations, and potential impacts on life goals.",
                    "exploration_questions": "How does substance use fit into your life right now? What are your thoughts about its role?",
                },
                "craving": {
                    "name": "relapse_prevention_cbt",
                    "primary_technique": "cbt",
                    "redirection_strategy": "Identify triggers for cravings, develop coping strategies, and create a relapse prevention plan.",
                    "exploration_questions": "What situations or feelings usually lead to cravings? What strategies have helped you manage them in the past, or what new ones could you try?",
                },
                # Add more specific keywords and approaches as needed
            }

            # Default approach if no specific match is found
            default_approach = {
                "name": DEFAULT_APPROACH,
                "primary_technique": "Person-Centered",
                "redirection_strategy": "Reflect recurring theme and invite deeper exploration",
                "exploration_questions": "I notice this theme comes up frequently. Could you share more about what this means to you?",
            }

            # Extract recurring terms from the pain point data
            recurring_terms = pain_point.get("recurring_terms", [])
            if not recurring_terms:
                logger.warning("No recurring terms found in pain point data.")
                return default_approach

            # Find the first matching approach based on recurring terms
            for term in recurring_terms:
                term_lower = term.lower()
                # Check for exact match or if term contains a key
                for key, approach in approaches.items():
                    if term_lower == key or key in term_lower:
                        logger.info("Found matching approach '%s' for term '%s'", approach["name"], term)
                        return approach  # Return the full approach dictionary

            # If no specific match, return the default approach
            logger.info("No specific approach found for terms: %s. Returning default.", recurring_terms)
            return default_approach

        except Exception as e:
            logger.error("Error getting recommended therapeutic approach: %s", e)
            # Return a safe default in case of any error
            return {
                "name": DEFAULT_TOPIC,
                "primary_technique": "Person-Centered",
                "redirection_strategy": "Provide empathetic reflection",
                "exploration_questions": "Can you tell me more about your experience?",
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
            current_session: List[Dict[str, Any]] = []

            for item in history:
                metadata = {}
                if isinstance(item.get("metadata"), str):
                    try:
                        metadata = json.loads(item.get("metadata", "{}"))
                    except:
                        metadata = {}
                elif isinstance(item.get("metadata"), dict):
                    metadata = item.get("metadata")

                # Check for session start markers
                if metadata.get("session_start"):
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
                    history = sorted(history, key=lambda x: x.get("created_at", ""))

                    # Group by approximate week
                    week_ms = 7 * 24 * 60 * 60 * 1000  # One week in milliseconds
                    current_week = None
                    current_session = []

                    for item in history:
                        timestamp = item.get("created_at", "")
                        if not timestamp:
                            continue

                        # Extract milliseconds from timestamp
                        try:
                            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
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
                    sessions = [history[i : i + chunk_size] for i in range(0, len(history), chunk_size)]

            # Analyze pain points in each session
            results = []

            for i, session in enumerate(sessions):
                # Skip very small sessions
                if len(session) < 2:
                    continue

                # Get session date from first message
                session_date = session[0].get("created_at", "")

                # Detect pain points in this session
                session_pain_points = self.detect_pain_points(
                    session_id,
                    threshold=0.6,  # Lower threshold for smaller sample
                    min_occurrences=max(min(len(session) // 3, 2), 1),  # Scale with session size
                )

                # Get primary themes from pain points
                primary_themes = []
                approach_types = []

                for pp in session_pain_points.get("pain_points", []):
                    # Get the primary theme term if available
                    primary_theme = (
                        pp.get("recurring_terms", ["unknown"])[0] if pp.get("recurring_terms") else "unknown"
                    )
                    primary_themes.append(primary_theme)

                    # Map the theme to an approach type
                    approach = map_theme_to_approach_type(primary_theme)
                    approach_types.append(approach)

                # Record pain point data with both themes and approach types
                results.append(
                    {
                        "session_number": i + 1,
                        "session_date": session_date,
                        "pain_point_count": len(session_pain_points.get("pain_points", [])),
                        "severity": session_pain_points.get("severity", "none"),
                        "primary_themes": primary_themes,
                        "approach_types": approach_types,
                        "message_count": len(session),
                    }
                )

            return results

        except Exception as e:
            logger.error("Error analysing pain points over time: %s", e)
            return []

    def get_emotional_signals(self, session_id: str) -> Dict[str, Any]:
        """
        Analyze emotional signals from user interactions in the current session.

        Args:
            session_id: The session ID to analyse

        Returns:
            Dictionary with emotional signals and their frequencies
        """
        try:
            # Ensure session is valid
            if not session_id:
                return {"error": "No session ID provided"}

            # Query to analyse emotional content across user messages
            query = f"""
            WITH user_messages AS (
                SELECT
                    question as text,
                    created_at
                FROM
                    {self.schema_name}.interactions
                WHERE
                    question IS NOT NULL AND question != ''
                    AND metadata->>'session_id' = '{session_id}'
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
            response = self.supabase.rpc("sql", {"command": query}).execute()

            if not response.data:
                # No emotional signals detected
                return {"signals": [], "primary_emotion": "neutral"}

            # Format the results
            signals = []
            for item in response.data:
                if isinstance(item, dict):
                    signals.append(
                        {"emotion": item.get("category", DEFAULT_EMOTION), "frequency": item.get("frequency", 0)}
                    )
                elif isinstance(item, str):
                    parts = item.split(",")
                    if len(parts) >= 2:
                        signals.append({"emotion": parts[0], "frequency": int(parts[1]) if parts[1].isdigit() else 0})

            # Determine primary emotion
            primary_emotion = signals[0]["emotion"] if signals else DEFAULT_EMOTION

            return {"signals": signals, "primary_emotion": primary_emotion}

        except Exception as e:
            logger.error("Error analysing emotional signals: %s", e)
            return {"error": str(e), "signals": [], "primary_emotion": "neutral"}

    def verify_schema_structure(self) -> bool:
        """Verifies the structure of the user schema for proper table setup."""
        try:
            # Call the function to verify schema structure
            response = self.supabase.rpc("verify_schema_structure", {"p_schema_name": self.schema_name}).execute()

            # Check if all required tables exist and have correct column counts
            if not response.data:
                logger.error("No tables found in schema: %s", self.schema_name)
                return False

            all_valid = True
            for table in response.data:
                if not table.get("table_exists"):  # Changed from 'exists' to 'table_exists'
                    logger.error("Missing table: %s", table.get("table_name"))
                    all_valid = False
                elif table.get("columns_found") != table.get("columns_expected"):
                    logger.error(
                        f"Table {table.get('table_name')} has {table.get('columns_found')} "
                        + f"columns but expected {table.get('columns_expected')}"
                    )
                    all_valid = False

            return all_valid

        except Exception as e:
            logger.error("Error verifying schema structure: %s", e)
            return False

    def ensure_schema_exists(self) -> bool:
        """

        Ensure the user schema exists and is properly set up.

        Returns:
            bool: True if the schema exists, False otherwise.

        """
        # Check if the schema already exists
        response = self.supabase.rpc("ensure_schema_exists", {"schema_name": self.schema_name}).execute()

        if response.data:
            logger.info("Schema %s already exists.", self.schema_name)
            return True
        return False
