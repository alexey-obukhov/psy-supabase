"""
RAGProcessor Module
===================

This module implements a specialized Retrieval-Augmented Generation (RAG) system for psychological
applications. It combines vector database retrieval, pain point detection, and therapeutic
response generation for psychological support conversations.

Key Components:
---------------
1. Vector Retrieval: Optimized pgvector similarity search with psychological knowledge base
2. Pain Point Detection: Identifies psychological fixations and recurring concerns
3. Therapeutic Approach Selection: Dynamically selects appropriate therapeutic approaches
4. Context Enhancement: Enriches responses with relevant psychological knowledge
5. Memory Management: Records and analyses conversation history for psychological patterns
6. Safety Monitoring: Detects and handles potentially harmful content
7. Dynamic RAG Integration: Real-time knowledge retrieval during response generation

Classes:
--------
RAGProcessor: Primary class for psychological RAG operations with specialized therapeutic features

Typical Usage:
--------------

.. code-block:: python

    from psy_supabase.core.database import DatabaseManager
    from psy_supabase.core.text_generator import TextGenerator
    from psy_supabase.core.rag_processor import RAGProcessor

    # Create dependencies
    db_manager = DatabaseManager(supabase_url, supabase_key, "user_123")
    generator = TextGenerator(model_name="mistralai/Mistral-7B-Instruct-v0.2", device="cuda")

    # Initialize RAG processor
    rag = RAGProcessor(db_manager=db_manager, generator=generator)

    # Generate therapeutic response
    response = rag.generate_response(
        user_question="I keep having the same anxious thoughts over and over",
        session_id="therapy_session_456"
    )

    # Generate response with contextual data
    context_data = rag.get_contextual_data(
        question="How can I manage my recurring panic attacks?",
        session_id="therapy_session_456"
    )

Dependencies:
psy_supabase.core.database: Vector database operations
psy_supabase.core.text_generator: LLM-based text generation
psy_supabase.core.dynamic_rag: Dynamic retrieval during generation
psy_supabase.utilities.prompt_selector: Therapeutic prompt selection
psy_supabase.utilities.embedding_utils: Vector embedding utilities
"""

import json
import traceback
from typing import Any, Dict, List, Optional, Union

from prismalog.log import get_logger
from typeguard import typechecked

from psy_supabase.config import DEFAULT_APPROACH, DEFAULT_EMOTION, DEFAULT_TOPIC
from psy_supabase.core.database import DatabaseManager
from psy_supabase.core.dynamic_rag import DynamicRAGRetriever
from psy_supabase.core.model_manager import EmbeddingProviderAdapter
from psy_supabase.core.response_generator import ResponseGenerator
from psy_supabase.core.text_generator import TextGenerator
from psy_supabase.memory.associative_memory import AssociativeMemory
from psy_supabase.rag.context_determination import create_context_from_similar_interactions, determine_context
from psy_supabase.utilities.embedding_utils import format_embedding_for_db
from psy_supabase.utilities.prompt_selector import PromptSelector
from psy_supabase.utilities.safety_handler import SafetyHandler
from psy_supabase.utilities.utils_mapping import map_approach_to_template

# Set up logging
logger = get_logger(__name__)


class RAGProcessor:
    """
    Advanced retrieval-augmented generation processor specialized for psychological applications.

    This class implements a comprehensive RAG system that integrates vector similarity search,
    pain point detection, therapeutic approach selection, and context enhancement to
    generate psychologically-informed responses for therapeutic conversations.

    Attributes:
        db_manager (DatabaseManager): Database manager for vector operations and history
        generator (TextGenerator): Text generator for response generation
        text_generator (TextGenerator): Alias for the generator
        safety_handler (SafetyHandler): Handler for content safety checks
        prompt_selector (PromptSelector): Selector for therapeutic prompts
        intelligent_processing_enabled (bool): Flag for enabling advanced processing
        embedding_provider (EmbeddingProviderAdapter): Provider for vector embeddings
        embedding_dimension (int): Dimension of embedding vectors
        SIMILARITY_THRESHOLD (float): Minimum similarity for relevant documents
        MAX_KNOWLEDGE_CHARS (int): Maximum characters for knowledge context
        MAX_CONVERSATION_EXCHANGES (int): Maximum conversation exchanges to include
        VECTOR_CACHE_ENABLED (bool): Enable vector caching for similar questions

    Pain Point Detection:
        The system identifies potential psychological pain points by analysing:
        1. Repetition patterns in user questions
        2. Emotional signals in conversation history
        3. Vector similarity to known psychological concerns
        4. Recurring themes across sessions

    Therapeutic Approach Selection:
        Based on detected pain points and conversational context, the system selects
        appropriate therapeutic approaches:
        1. anxiety exploration for anxiety-related fixations
        2. Grief reflection for sadness and loss themes
        3. Gentle refocus for persistent thought patterns
        4. Exploratory approaches for general psychological concerns

    Vector Optimization:
        The implementation uses database-side vector operations with pgvector:
        1. Similarity search performed in PostgreSQL/Supabase
        2. Vector caching for performance optimization
        3. Dimensionality-aware operations for memory efficiency
        4. Threshold filtering to maintain therapeutic relevance
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        generator: TextGenerator,
        intelligent_processing_enabled: bool = True,
        associative_memory: Optional[AssociativeMemory] = None,
    ):
        self.db_manager = db_manager
        self.text_generator = generator
        self.generator = generator
        self.safety_handler = SafetyHandler()
        self.prompt_selector = PromptSelector(generator)
        self.intelligent_processing_enabled = intelligent_processing_enabled
        self.associative_memory = associative_memory

        # Initialize ResponseGenerator
        self.response_generator = ResponseGenerator(
            text_generator=self.generator, db_manager=self.db_manager, prompt_selector=self.prompt_selector
        )

        # Use EmbeddingProviderAdapter instead
        self.embedding_provider = EmbeddingProviderAdapter()

        # Get the embedding dimension directly from the provider
        self.embedding_dimension = self.embedding_provider.get_embedding_dimension()

        # Constants for vector retrieval optimization
        self.SIMILARITY_THRESHOLD = 0.7  # Minimum similarity for relevant documents
        self.MAX_KNOWLEDGE_CHARS = 500  # Max characters for knowledge context
        self.MAX_CONVERSATION_EXCHANGES = 2  # Max conversation exchanges to include
        self.VECTOR_CACHE_ENABLED = True  # Enable vector caching for similar questions

    @typechecked
    def generate_response(
        self,
        user_question: str,
        session_id: str = "default_session",
        device: str = "cuda",
        question_id: Optional[Any] = 0,
    ) -> str:
        """Enhanced to properly connect pain point detection to template selection"""

        # Create unique tracking ID for this interaction
        tracking_id = str(question_id)

        # Initialize metadata
        metadata = {
            "session_id": session_id,
            "tracking_id": tracking_id,
            "context_determination_used": True,
        }

        # Handle invalid inputs
        if not self.response_generator.is_valid_input(user_question):
            return self.response_generator.get_default_response(user_question)

        try:
            # Check toxic content
            toxic_result = self.response_generator.check_toxic_content(user_question, session_id)
            if toxic_result:
                return toxic_result

            # Configure embedding provider if specified
            if device and hasattr(self.embedding_provider, "set_device"):
                self.embedding_provider.set_device(device)

            # Process query to get semantic meaning first
            embedding = self.process_query(user_question, session_id)

            # Use both context determination approaches
            # Get chronological context (most recent conversations)
            conversation_context = determine_context(
                self.db_manager, user_question, session_id=session_id, limit=self.MAX_CONVERSATION_EXCHANGES
            )

            # Get semantically similar interactions
            semantic_context = create_context_from_similar_interactions(
                self.db_manager, embedding, session_id=session_id, limit=3
            )

            # Combine both approaches for the richest context
            context_sources = []
            combined_context = ""

            if conversation_context:
                combined_context += f"Recent conversation history:\n{conversation_context}\n\n"
                context_sources.append("chronological")

            if semantic_context:
                combined_context += f"Related past interactions:\n{semantic_context}"
                context_sources.append("semantic")

            # Add context information to metadata
            if combined_context:
                metadata["context_sources"] = context_sources
                metadata["context_length"] = len(combined_context)
                logger.info("Using combined context determination: %d chars", len(combined_context))

            # After pain point detection
            pain_point_results = self.detect_pain_points_from_embedding(
                user_question=user_question, embedding=embedding, session_id=session_id, metadata=metadata
            )

            # Analyze topics and emotions
            topics_context = self.prompt_selector.analyze_question(user_question)

            metadata["topics_context"] = topics_context
            metadata["topic"] = topics_context.get("topic", DEFAULT_TOPIC)
            metadata["emotion"] = topics_context.get("emotion", DEFAULT_EMOTION)

            if pain_point_results and pain_point_results.get("pain_point_detected", False):
                # Get approach type from pain point detection
                approach_type = pain_point_results.get("approach_type", DEFAULT_APPROACH)

                # Map approach to template using your existing utility
                template_name = map_approach_to_template(approach_type)

                # Store in metadata for use by text generator
                metadata["template_used"] = template_name
                metadata["approach_type"] = approach_type

            # Get hot topics
            hot_topics = self._identify_hot_topics(user_question, embedding)

            # Initialize the dynamic retriever
            dynamic_retriever = self.create_dynamic_retriever(session_id=session_id)

            # Build generation context
            generation_context = self.response_generator.build_generation_context(
                user_question=user_question,
                session_id=session_id,
                topics_context=topics_context,
                pain_point_results=pain_point_results,
                query_embedding=embedding,
                dynamic_retriever=dynamic_retriever,
                hot_topics=hot_topics,
            )

            # This gives us rich context using pgvector similarity search
            enhanced_context = self._enhance_context_with_relevant_documents(user_question, embedding, session_id)

            # Add enhanced knowledge context to generation context
            if enhanced_context.get("knowledge_context"):
                generation_context["knowledge_context"] = enhanced_context.get("knowledge_context")

            # Get documents using existing method for relevant documents
            relevant_documents = self.get_relevant_documents(embedding, top_k=5)

            if relevant_documents:
                generation_context = self.enhance_context_with_relevant_documents(
                    generation_context, relevant_documents
                )

            # Add the combined context to the generation context
            if combined_context:
                generation_context["conversation_context"] = combined_context

            # Generate response
            response = self.response_generator.generate_response_with_template(
                user_question=user_question,
                session_id=session_id,
                generation_context=generation_context,
                pain_point_results=pain_point_results,
            )

            # Determine final context using ResponseGenerator's method
            context, updated_metadata = self.response_generator.determine_final_context(
                user_question=user_question,
                topics_context=topics_context,
                pain_point_results=pain_point_results,
                metadata=metadata,
            )

            logger.info("Final context determined: %s", context)

            metadata = updated_metadata
            context_str = context

            # Save interaction with metadata including context sources
            # Convert context to string before passing to save_interaction
            # if topics_context is None:
            #     context_str = "therapeutic_dialogue"
            # elif isinstance(topics_context, (list, tuple, set)):
            #     # Take first item from collection if it exists, otherwise use default
            #     context_str = str(next(iter(topics_context), "therapeutic_dialogue"))
            # else:
            #     context_str = str(topics_context)

            self.db_manager.save_interaction(
                question=user_question,
                answer=response,
                context=context_str,
                session_id=session_id,
                metadata=metadata,
            )

            return response

        # Ensure a string is always returned, even in error cases
        except Exception as e:
            logger.error("Error generating response: %s", e)
            logger.error("Traceback (most recent call last):", exc_info=True)
            return "I apologize, but I'm having trouble generating a response right now. Please try again later."

    def _identify_hot_topics(self, user_question: str, query_embedding: List[float]) -> List[Dict[str, Any]]:
        """
        Identify hot topics in the user's question using vector similarity.

        Args:
            user_question: The user's question
            query_embedding: The embedding of the user's question

        Returns:
            List[Dict]: Hot topics with relevance scores
        """
        try:
            # Use a specialized "hot topics" search
            hot_topics = []

            # Focus on specific psychological themes
            themes = ["anxiety", "depression", "stress", "relationships", "trauma", "grief", "self-esteem", "identity"]

            # Check if any of these themes are directly mentioned
            user_question_lower = user_question.lower()

            for theme in themes:
                if theme in user_question_lower:
                    hot_topics.append(
                        {
                            "topic": theme,
                            "relevance": 0.95,  # High relevance for direct mentions
                            "source": "direct_mention",
                        }
                    )

            # If we found direct mentions, return those
            if hot_topics:
                return hot_topics

            # Otherwise, try vector search
            if query_embedding:
                # Only include relevant hot topics (above threshold)
                threshold = 0.75  # Higher threshold for hot topics

                # Format embedding for PostgreSQL vector format
                vector_str = format_embedding_for_db(query_embedding)

                # SQL to find hot topics
                query = f"""
                WITH hot_topic_embeddings AS (
                    SELECT
                        id,
                        content,
                        embedding,
                        1 - (embedding <=> '{vector_str}'::vector) as similarity
                    FROM
                        public.hot_topics
                    WHERE
                        1 - (embedding <=> '{vector_str}'::vector) > {threshold}
                    ORDER BY
                        similarity DESC
                    LIMIT 2
                )
                SELECT
                    id,
                    content as topic,
                    similarity as relevance
                FROM
                    hot_topic_embeddings;
                """

                try:
                    # Check if hot_topics table exists first
                    check_query = (
                        "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'hot_topics');"
                    )
                    check_result = self.db_manager.supabase.rpc("sql", {"command": check_query}).execute()

                    if check_result.data and (check_result.data[0] == "t" or check_result.data[0] is True):
                        # Table exists, query it
                        result = self.db_manager.supabase.rpc("sql", {"command": query}).execute()

                        if result.data:
                            for row in result.data:
                                if isinstance(row, dict):
                                    hot_topics.append(
                                        {
                                            "topic": row.get("topic", ""),
                                            "relevance": row.get("relevance", 0),
                                            "source": "vector_similarity",
                                        }
                                    )
                                elif isinstance(row, str):
                                    # Parse CSV-formatted response
                                    parts = row.split(",")
                                    if len(parts) >= 3:
                                        hot_topics.append(
                                            {
                                                "topic": parts[1],
                                                "relevance": (
                                                    float(parts[2]) if parts[2].replace(".", "", 1).isdigit() else 0
                                                ),
                                                "source": "vector_similarity",
                                            }
                                        )
                except Exception as inner_e:
                    logger.warning("Error finding hot topics: %s", inner_e)
                    # Continue without hot topics

            return hot_topics

        except Exception as e:
            logger.error("Error identifying hot topics: %s", e)
            return []

    def generate_simple_response(self, user_question: str) -> str:
        """Generates a simple response without any preprocessing or context."""
        return self.text_generator.generate_text(user_question)

    def generate_training_examples(
        self, topic_filter: Optional[str] = None, min_effectiveness: float = 0.7, limit: int = 100
    ) -> List[Dict]:
        """
        Generates high-quality training examples from past interactions.

        Args:
            topic_filter: Optional topic to filter by
            min_effectiveness: Minimum effectiveness score to include
            limit: Maximum number of examples to generate

        Returns:
            List of formatted training examples ready for fine-tuning
        """
        # Retrieve high-quality interactions from database
        interactions = self.db_manager.get_high_quality_interactions(
            topic_filter=topic_filter, min_effectiveness=min_effectiveness, limit=limit
        )

        # Format for training
        training_examples = []
        for interaction in interactions:
            try:
                # Parse metadata
                metadata = interaction.get("metadata", {})
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)

                # Extract key information
                question = interaction.get("question", "")
                answer = interaction.get("answer", "")
                context = interaction.get("context", "")
                topic = metadata.get("topic", "emotional_support")

                # Add psychological context if available
                psychological_context = ""
                if metadata.get("emotional_state"):
                    psychological_context += f"Emotional state: {metadata.get('emotional_state')}\n"

                if metadata.get("recurring_themes"):
                    themes = metadata.get("recurring_themes")
                    if isinstance(themes, list):
                        psychological_context += f"Recurring themes: {', '.join(themes)}\n"

                # Create a formatted training example
                example = {
                    "question": question,
                    "answer": answer,
                    "context": context,
                    "topic": topic,
                    "psychological_context": psychological_context.strip(),
                }

                training_examples.append(example)
            except Exception as e:
                logger.error("Error formatting training example: %s", e)
                continue

        return training_examples

    def _build_psychological_context(
        self, similar_memories: List[Dict], theme_clusters: List[Dict], emotional_trajectory: List[Dict]
    ) -> Dict[str, Any]:
        """
        Build psychological context from vector-retrieved data.

        Args:
            similar_memories: Vector-similar past memories
            theme_clusters: pgvector theme clusters
            emotional_trajectory: Emotional vector trajectory

        Returns:
            Dict with psychological context
        """
        context: Dict[str, Union[str, List[str], List[Dict[str, Any]]]] = {}

        # Extract emotional signals from similar memories
        if similar_memories:
            emotional_signals = []
            recurring_topics: List[str] = []

            for memory in similar_memories:
                # Extract emotion signals
                if memory.get("metadata") and isinstance(memory["metadata"], dict):
                    emotion = memory["metadata"].get("emotional_state")
                    if emotion:
                        emotional_signals.append(emotion)

                # Extract topics
                topic = memory.get("topic") or (
                    memory["metadata"].get("topic")
                    if memory.get("metadata") and isinstance(memory["metadata"], dict)
                    else None
                )
                if topic:
                    recurring_topics.append(topic)

            # Add to psychological context
            if emotional_signals:
                context["emotional_signals"] = emotional_signals
            if recurring_topics:
                context["recurring_topics"] = recurring_topics

        # Add theme clusters
        if theme_clusters:
            context["theme_clusters"] = theme_clusters

            # Extract primary theme if available
            primary_themes: List[str] = []
            for cluster in theme_clusters:
                theme = cluster.get("dominant_theme")
                if theme and isinstance(theme, str):
                    primary_themes.append(theme)
                elif theme:
                    logger.warning("Found non-string dominant_theme in cluster: %s", type(theme))

            if primary_themes:
                context["primary_themes"] = primary_themes

        # Add emotional trajectory
        if emotional_trajectory:
            context["emotional_trajectory"] = emotional_trajectory

            # Extract current emotional state (from most recent interaction)
            if emotional_trajectory and len(emotional_trajectory) > 0:
                context["current_emotional_state"] = emotional_trajectory[-1].get("emotional_state", "")

            # Detect emotional trends
            if len(emotional_trajectory) >= 2:
                # Analyze if emotional states are improving
                valence_scores = [entry.get("valence", 0) for entry in emotional_trajectory if "valence" in entry]

                if valence_scores and len(valence_scores) >= 2:
                    # Check if valence is generally improving
                    is_improving = valence_scores[-1] > valence_scores[0]
                    context["emotional_trend"] = "improving" if is_improving else "stable_or_declining"

        return context  # Ensure context is returned

    def _enhance_context_with_relevant_documents(
        self, user_question: str, question_embedding: List[float], session_id: str
    ) -> Dict:
        """
        Enhance context with only the most relevant documents while maintaining a fixed context size.

        OPTIMIZED: Uses pgvector's similarity search for document selection, limiting data transfer.

        Args:
            user_question (str): User's question to enhance with context
            question_embedding (List[float]): Vector embedding of the question
            session_id (str): Session ID for conversation history

        Returns:
            Dict: Enhanced context with knowledge and conversation data
        """
        try:
            # OPTIMIZED: Use pgvector search with threshold applied in database
            similar_docs = self.db_manager.find_similar_documents(
                embedding=question_embedding, limit=5, min_similarity=self.SIMILARITY_THRESHOLD
            )

            # Initialize knowledge context with fixed maximum size
            knowledge_context = ""
            relevant_docs = []

            if similar_docs:
                for doc in similar_docs:
                    # Handle case where doc is a string (the response format issue)
                    if isinstance(doc, str):
                        # Try to parse JSON if it's a JSON string
                        try:
                            doc_dict = json.loads(doc)
                            content = doc_dict.get("content", "")
                            similarity = doc_dict.get("similarity", 0)
                        except Exception as e:
                            # If not JSON, use the string as content with default similarity
                            logger.error("Error parsing JSON document: %s", e)
                            content = doc
                            similarity = 0.7  # Default similarity above threshold
                    else:
                        # Normal dictionary case
                        content = doc.get("content", "")
                        similarity = doc.get("similarity", 0)

                    # Database filtering should handle this, but double-check
                    if similarity >= self.SIMILARITY_THRESHOLD:
                        relevant_docs.append((content, similarity))

                # Build knowledge context, keeping track of total length
                total_length = 0
                final_docs: List[str] = []

                for content, similarity in relevant_docs:
                    # Calculate how much this document would add
                    content_length = len(content)

                    # If adding this document would exceed our limit, stop
                    if total_length + content_length > self.MAX_KNOWLEDGE_CHARS:
                        # If this is the first document and it's too long, truncate it
                        if not final_docs:
                            truncated = content[: self.MAX_KNOWLEDGE_CHARS] + "..."
                            final_docs.append(truncated)
                        break

                    # Otherwise add the full document
                    final_docs.append(content)
                    total_length += content_length

                # Join the final set of documents
                knowledge_context = "\n\n".join(final_docs)

                # Log what we're including
                logger.info("Using %d documents (%d chars) for knowledge context", len(final_docs), total_length)

            # OPTIMIZATION: Get conversation history with limited exchanges
            conversation_context = ""
            try:
                conversation_history = self.db_manager.get_conversation_history(session_id)

                if conversation_history and len(conversation_history) > 0:
                    recent_exchanges = conversation_history[-self.MAX_CONVERSATION_EXCHANGES :]

                    conversation_parts = []
                    for exchange in recent_exchanges:
                        q = exchange.get("question", exchange.get("question", ""))
                        a = exchange.get("answer", exchange.get("answer", ""))
                        if q and a:
                            # Truncate if needed
                            q_short = q if len(q) < 100 else q[:97] + "..."
                            a_short = a if len(a) < 150 else a[:147] + "..."
                            conversation_parts.append(f"User: {q_short}")
                            conversation_parts.append(f"Assistant: {a_short}")

                    conversation_context = "\n".join(conversation_parts)

                    # Log the conversation context
                    if conversation_context:
                        logger.debug("Added conversation context (%d chars)", len(conversation_context))
            except Exception as e:
                logger.error("Error retrieving conversation history: %s", e)
                # Continue with empty conversation context

            # Create enhanced context dictionary with consistent size limits
            enhanced_context = {
                "knowledge_context": knowledge_context.strip(),
                "conversation_context": conversation_context.strip(),
                "session_id": session_id,
                "has_knowledge": bool(knowledge_context.strip()),
                "has_conversation": bool(conversation_context.strip()),
                "vector_threshold": self.SIMILARITY_THRESHOLD,
                "user_question": user_question,
            }

            # Log context sizes
            logger.info("Knowledge context: %d chars from vector similarity search", len(knowledge_context))
            logger.info("Conversation context: %d chars", len(conversation_context))
            logger.info("Total prompt context: %d chars", len(knowledge_context) + len(conversation_context))

            return enhanced_context
        except Exception as e:
            logger.error("Error enhancing context: %s", e)
            logger.error(traceback.format_exc())
            return {
                "knowledge_context": "",
                "conversation_context": "",
                "session_id": session_id,
                "has_knowledge": False,
                "has_conversation": False,
                "user_question": user_question,  # Include user question even in error case
            }

    def enhance_context_with_relevant_documents(
        self, context: Optional[Dict[str, Any]], relevant_documents: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Enhance generation context with relevant documents from retrieval.

        Args:
            context: The generation context to enhance
            relevant_documents: List of documents with relevance scores

        Returns:
            Enhanced context with added document content
        """
        # Make a copy to avoid modifying the original
        enhanced_context = context.copy() if context else {}

        # Initialize the relevant documents list
        enhanced_context["relevant_documents"] = []

        # Handle None case defensively
        if not relevant_documents:
            return enhanced_context

        # Process and add each relevant document
        for doc in relevant_documents:
            if isinstance(doc, dict) and "content" in doc:
                enhanced_context["relevant_documents"].append(doc["content"])

        return enhanced_context

    def _generate_pain_point_approach(
        self, original_question: str, current_question: str, emotions: List[Dict], repetition_pattern: Dict
    ) -> Dict:
        """
        Generate appropriate therapeutic approach based on detected pain points and question evolution.

        This method implements a clinical decision system that analyses:
        1. Question evolution: Whether the user is asking essentially the same question repeatedly
        or if their questions are evolving in a healthy way
        2. Emotional patterns: The dominant emotional states detected in user messages
        3. Fixation patterns: Whether the user is fixating on specific topics or concerns

        Based on this analysis, the method selects between different therapeutic approaches:

        * 'anxiety_exploration': Used when detecting fixation on anxiety-related topics with
        minimal question evolution. This approach helps users explore the root of their
        anxiety rather than reinforcing repetitive thought patterns.

        * 'exploratory': Used when questions show evolution or when no strong fixation is
        detected. This approach encourages continued exploration and self-discovery.

        * Other specialized approaches may be selected for depression, grief, etc.

        Clinical rationale: Research in cognitive behavioral therapy suggests that identifying
        and redirecting repetitive thought patterns is more effective than directly answering
        repetitive questions, which can reinforce rumination.

        Args:
            original_question (str): The user's previous question or statement
            current_question (str): The user's current question or statement
            emotions (List[Dict]): List of detected emotions with their intensities
                                [{'emotional_state': 'anxious', 'intensity': 0.8}, ...]
            repetition_pattern (Dict): Analysis of repetition and fixation patterns
                                    {'count': 3, 'recurring_terms': [...], 'is_fixation': True}

        Returns:
            Dict: Therapeutic approach configuration with:
                - approach_type (str): The selected therapeutic approach
                - emotional_tone (str): Dominant emotional tone detected
                - guidance_question (str): A question to guide the user's exploration
                - should_redirect (bool): Whether to redirect the conversation

        Example:
            When a user repeatedly asks about anxiety symptoms with minimal variation:
            {'approach_type': 'anxiety_exploration',
            'emotional_tone': 'anxious',
            'guidance_question': 'I notice you've mentioned feeling anxious several times...',
            'should_redirect': True}
        """
        # Determine if this seems to be a fixation pattern
        is_fixation = repetition_pattern.get("is_fixation", False)
        recurring_terms = repetition_pattern.get("recurring_terms", [])

        # Analyze emotional tone from emotion data
        emotional_tone = "neutral"
        if emotions:
            # Get the most common emotion
            emotion_counter: Dict[str, int] = {}
            for e in emotions:
                emotion = e.get("emotional_state", "").lower()
                if emotion:
                    emotion_counter[emotion] = emotion_counter.get(emotion, 0) + 1

            if emotion_counter:
                emotional_tone = max(emotion_counter.items(), key=lambda x: x[1])[0]

        logger.debug("Generating approach for original question: %s...", original_question[:50])

        # Analyze if current_question has changed significantly
        question_change = len(set(current_question.lower().split()) - set(original_question.lower().split()))
        question_evolved = question_change > 3  # Consider evolved if 3+ new words

        # Adjust approach based on question evolution
        if is_fixation and not question_evolved:
            if emotional_tone in ["anxious", "worried", "fear", "anxiety"]:
                approach_type = "anxiety_exploration"
                guidance_question = (
                    f"I notice you've mentioned {', '.join(recurring_terms[:2])} several times. "
                    f"These topics seem to cause you anxiety. Could you tell me what feels most "
                    f"overwhelming about this situation?"
                )

            elif emotional_tone in ["sad", "depressed", "grief", "depression"]:
                approach_type = "grief_reflection"
                guidance_question = (
                    f"You've brought up {', '.join(recurring_terms[:2])} multiple times, and I sense "
                    f"some sadness there. What feelings come up for you when you think about this?"
                )

            else:
                approach_type = "gentle_refocus"
                guidance_question = (
                    f"I've noticed we've discussed {', '.join(recurring_terms[:2])} several times. "
                    f"I wonder if we could explore what makes this particularly important for you right now?"
                )
        else:
            # Not a fixation, but still a pain point - use a lighter approach
            approach_type = "exploratory"
            guidance_question = (
                f"I notice that {', '.join(recurring_terms[:2]) if recurring_terms else 'this topic'} "
                f"seems meaningful to you. Could you share more about how it affects you?"
            )

        return {  # Ensure dict is returned
            "approach_type": approach_type,
            "emotional_tone": emotional_tone,
            "guidance_question": guidance_question,
            "should_redirect": is_fixation,
        }

    @typechecked
    def detect_pain_points_from_embedding(
        self, user_question: str, embedding: List[float], session_id: str, metadata: Optional[dict] = None
    ) -> Dict[str, Any]:
        """Detect potential pain points from question embedding with proper error handling."""
        # Default response structure, especially for when no pain point is detected.
        # 'template_used' is set to 'dynamic_rag_therapy' to align with previous test fixes.
        default_response: Dict[str, Any] = {
            "pain_point_detected": False,
            "template_used": "dynamic_rag_therapy",
            "approach_type": DEFAULT_APPROACH,
            "similarity": 0.0,
            "pain_point": {},
        }

        try:
            # Fetch pain point data from the database manager
            pain_point_data_from_db = self.db_manager.identify_potential_pain_points(
                question_text=user_question, question_embedding=embedding, session_id=session_id, pain_threshold=0.85
            )

            # Check if a pain point was genuinely detected and data is available
            if pain_point_data_from_db and pain_point_data_from_db.get("detected", False):
                # A pain point IS detected.

                # Extract the 'suggested_approach' dictionary from the pain point data.
                # Defaults to an empty dict if 'suggested_approach' is not found.
                suggested_approach_info = pain_point_data_from_db.get("suggested_approach", {})

                # Determine 'approach_type' using your specified logic:
                # If 'suggested_approach_info' is a dictionary and contains 'approach_type', use it.
                # Otherwise, fall back to DEFAULT_APPROACH.
                approach_type = (
                    suggested_approach_info.get("approach_type")
                    if isinstance(suggested_approach_info, dict)
                    else DEFAULT_APPROACH
                )

                # Map the determined 'approach_type' to a specific template name.
                # Ensure 'map_approach_to_template' is imported (usually at the module level).
                template_used = map_approach_to_template(approach_type)

                # Construct the result dictionary for a detected pain point.
                result = {
                    "pain_point_detected": True,
                    "template_used": template_used,
                    "approach_type": approach_type,
                    "similarity": pain_point_data_from_db.get("similarity", 0.0),
                    "pain_point": pain_point_data_from_db,  # Include the full pain point data from DB
                }

                # If a repetition pattern is part of the pain point data, add it to the result.
                if pain_point_data_from_db.get("repetition_pattern"):
                    result["repetition_pattern"] = pain_point_data_from_db.get("repetition_pattern")

                # If metadata is provided, update it with the pain point detection details.
                if metadata is not None and isinstance(metadata, dict):
                    if "pain_points" not in metadata:
                        metadata["pain_points"] = []
                    metadata["pain_points"].append(
                        {
                            "question": user_question,
                            "detected": result["pain_point_detected"],
                            "similarity": result["similarity"],
                            "template_used": result["template_used"],
                        }
                    )
                return result
            # This block handles cases where:
            # 1. pain_point_data_from_db is None or empty.
            # 2. pain_point_data_from_db.get("detected") is False.
            log_message_db_resp = str(pain_point_data_from_db)[:100] if pain_point_data_from_db is not None else "None"
            logger.info(
                f"No pain point detected or data unavailable for question: '{user_question[:50]}...'. "
                f"DB response: {log_message_db_resp}. Returning default response."
            )
            # Update metadata for the "no pain point" case as well.
            if metadata is not None and isinstance(metadata, dict):
                if "pain_points" not in metadata:
                    metadata["pain_points"] = []
                metadata["pain_points"].append(
                    {
                        "question": user_question,
                        "detected": False,
                        "similarity": (
                            pain_point_data_from_db.get("similarity", 0.0) if pain_point_data_from_db else 0.0
                        ),
                        "template_used": default_response["template_used"],
                    }
                )
            return default_response

        except Exception as e:
            logger.error(
                f"Error in detect_pain_points_from_embedding for question '{user_question[:50]}...': {e}", exc_info=True
            )
            # Update metadata for the error case.
            if metadata is not None and isinstance(metadata, dict):
                if "pain_points" not in metadata:
                    metadata["pain_points"] = []
                metadata["pain_points"].append(
                    {
                        "question": user_question,
                        "detected": False,
                        "error": str(e),
                        "template_used": default_response["template_used"],  # Use default template on error
                    }
                )
            return default_response

    @typechecked
    def get_relevant_documents(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """
        Retrieves the most relevant documents using pgvector similarity.

        OPTIMIZED: Uses direct pgvector similarity search in database instead of Python-side calculation.

        Args:
            query_embedding (List[float]): Vector embedding to find similar documents for
            top_k (int): Maximum number of documents to return

        Returns:
            List[Dict]: List of relevant documents with similarity scores
        """
        try:
            # Let pgvector handle the similarity calculation in the database
            similar_docs = self.db_manager.find_similar_documents(
                embedding=query_embedding,
                limit=top_k,
                min_similarity=self.SIMILARITY_THRESHOLD,  # Apply similarity threshold filter
            )

            # Format the results as needed
            relevant_documents = []
            for doc in similar_docs:
                if isinstance(doc, dict):
                    # Extract required fields and add to results
                    relevant_doc = {
                        "content": doc.get("content", ""),
                        "similarity": doc.get("similarity", 0),
                        "id": doc.get("id"),
                    }
                    relevant_documents.append(relevant_doc)

                    # Log high-quality matches
                    if doc.get("similarity", 0) > 0.8:
                        logger.info("Found highly relevant document (similarity: %.3f)", doc.get("similarity", 0))
            # Add debug logging to see what documents are being retrieved
            for i, doc in enumerate(relevant_documents):
                content_preview = doc.get("content", "")[:100] + "..." if doc.get("content") else ""
                logger.debug("Retrieved document %d (sim: %.3f): %s", i + 1, doc.get("similarity", 0), content_preview)
            return relevant_documents

        except Exception as e:
            logger.error("Error retrieving documents with pgvector: %s", e)
            return []

    @typechecked
    def get_recent_conversation_history(self, session_id: str, limit: int = 2) -> List[Dict]:
        """
        Get recent conversation history with appropriate formatting for psychological context.

        Args:
            session_id: Session identifier
            limit: Maximum number of recent exchanges to include

        Returns:
            List[Dict]: Recent conversation exchanges properly formatted
        """
        try:
            # Get conversation history from database
            conversation_history = self.db_manager.get_conversation_history(session_id)

            if not conversation_history:
                return []

            # Only take the most recent exchanges to limit context size
            recent_history = (
                conversation_history[-limit:] if len(conversation_history) > limit else conversation_history
            )

            # Format the conversation history for the generator
            formatted_history = []
            for item in recent_history:
                # Extract question and answer
                q = item.get("question", item.get("question", ""))
                a = item.get("answer", item.get("answer", ""))

                # Include metadata if available
                metadata = item.get("metadata", {})
                if isinstance(metadata, str):
                    try:
                        # Try to parse metadata if it's a string
                        metadata = json.loads(metadata)
                    except Exception as e:
                        logger.error("Error parsing metadata: %s", e)
                        metadata = {}

                # Create formatted entry
                entry = {
                    "question": q,
                    "answer": a,
                    "metadata": metadata,
                    "timestamp": item.get("created_at", item.get("timestamp", "")),
                }

                # For psychological work, preserve the FULL text of the exchanges
                # DO NOT truncate text here - it's critical for psychological continuity
                formatted_history.append(entry)

            return formatted_history

        except Exception as e:
            logger.error("Error getting conversation history: %s", e)
            logger.error(traceback.format_exc())
            return []

    def _log_pain_point_detection(self, user_question: str, pain_point: Dict, template_used: str) -> None:
        """Log pain point detection for analysis."""
        try:
            # Log structured data for later analysis
            metadata = {
                "event_type": "pain_point_detected",
                "pain_point": pain_point.get("pain_point", ""),
                "recurring_terms": pain_point.get("recurring_terms", []),
                "count": pain_point.get("count", 0),
                "severity": pain_point.get("severity", ""),
                "template_used": template_used,
                "approach": pain_point.get("approach", {}).get("name", ""),
            }

            # Create a special log entry in interactions table
            self.db_manager.save_interaction(
                context="Pain Point System",
                question=user_question,
                answer="Pain point detection triggered",
                metadata=metadata,
                session_id="default_session",  # Use consistent session ID, not None
            )

            logger.info(
                "Pain point detected: %s (count: %d, severity: %s)",
                pain_point.get("pain_point", "unknown"),
                pain_point.get("count", 0),
                pain_point.get("severity", "unknown"),
            )
        except Exception as e:
            logger.error("Error logging pain point detection: %s", e)

    def create_dynamic_retriever(
        self, session_id: Optional[str] = None, rag_options: Optional[dict] = None
    ) -> DynamicRAGRetriever:
        """
        Create a DynamicRAGRetriever with compatible parameters.

        Args:
            session_id: Session ID for context filtering
            rag_options: Additional RAG options

        Returns:
            An instance of DynamicRAGRetriever
        """
        options = rag_options or {}

        # Create and return the retriever by passing arguments directly
        # using original variables and the options dictionary
        if not hasattr(self, "_retriever_cache"):
            self._retriever_cache: Dict[str, Any] = {}
        cache_key = session_id or "default"
        if cache_key not in self._retriever_cache:
            self._retriever_cache[cache_key] = DynamicRAGRetriever(
                db_manager=self.db_manager,
                session_id=session_id,
                persona=options.get("persona"),
                query_mode=options.get("query_mode"),
                embedding_provider=options.get("embedding_provider"),
                rag_processor=options.get("rag_processor", self),
            )
        retriever = self._retriever_cache[cache_key or "default"]
        return retriever

    @typechecked
    def process_query(self, user_question: str, session_id: str) -> List[float]:
        """
        Process the user query: check cache/DB for existing embedding,
        otherwise generate a new one.

        Args:
            user_question: The raw question text from the user.
            session_id: The current session ID.

        Returns:
            List[float]: The vector embedding of the user question.

        Raises:
            ValueError: If embedding generation fails and no existing embedding is found.
        """
        cache_key = f"{session_id}:{user_question}"
        if not hasattr(self, "_query_embedding_cache"):
            self._query_embedding_cache: Dict[str, List[float]] = {}
        if cache_key in self._query_embedding_cache:
            logger.debug("Returning cached embedding for query in session %s", session_id)
            return self._query_embedding_cache[cache_key]

        try:
            logger.debug("Checking DB for existing embedding for query in session %s", session_id)
            # find_similar_question_embedding returns Optional[List[float]]
            existing_embedding = self.db_manager.find_similar_question_embedding(
                question_text=user_question,
                session_id=session_id,
                # Ensure parameter name matches the DB method definition
                similarity_threshold=0.98,
            )

            # Check if embedding is not None AND is a list
            if existing_embedding is not None and isinstance(existing_embedding, list):
                logger.debug("Found existing embedding in DB for query in session %s", session_id)
                # Store in in-memory cache before returning
                self._query_embedding_cache[cache_key] = existing_embedding
                return existing_embedding  # Return the list directly

            logger.debug("No existing/cached embedding found. Generating new one for session %s", session_id)
            embedding = self.embedding_provider.generate_embedding(user_question)
            if embedding is None:
                logger.error("Failed to generate embedding for query in session %s", session_id)
                raise ValueError("Embedding generation returned None")

            logger.debug("Generated new embedding for query in session %s", session_id)
            # Store in in-memory cache before returning
            self._query_embedding_cache[cache_key] = embedding
            return embedding

        except Exception as e:
            logger.error("Error during query processing for session %s: %s", session_id, e)
            logger.error(traceback.format_exc())  # Log full traceback
            # This wraps the original error (e.g., the AttributeError)
            raise ValueError(f"Failed to process query embedding: {e}") from e

    @typechecked
    def get_contextual_data(
        self, user_question: str, session_id: str, query_embedding: List[float], limit: int = 5
    ) -> Dict[str, Any]:
        """
        Retrieves and combines various contextual data sources for response generation.

        Args:
            user_question: The user's input question.
            session_id: The current session ID.
            query_embedding: The embedding vector for the user question.
            limit: The maximum number of items to retrieve for each context type.

        Returns:
            A dictionary containing combined contextual data.
        """
        context_data: Dict[str, Union[str, List[Dict[str, Any]], List[str]]] = {
            "conversation_context": "",
            "knowledge_items": [],
            "past_interactions": [],
            "similar_memories": [],
            "theme_clusters": [],
            "emotional_trajectory": [],
            "hot_topics": [],
        }
        try:
            retriever = DynamicRAGRetriever(db_manager=self.db_manager, session_id=session_id, rag_processor=self)

            # 1. Conversation History
            context_data["conversation_context"] = retriever.get_conversation_context(limit=limit)

            # 2. Relevant Documents/Knowledge Items
            knowledge_items = retriever.get_knowledge_by_query(
                query=user_question, embedding=query_embedding, limit=limit
            )
            context_data["knowledge_items"] = knowledge_items

            # 3. Similar Past Interactions
            context_data["past_interactions"] = retriever.get_past_interactions(session_id=session_id, limit=limit)

            # 4. Associative Memory Retrieval
            if (
                hasattr(self, "associative_memory")
                and self.associative_memory
                and getattr(self, "memory_initialized", False)
            ):
                context_data["similar_memories"] = self.associative_memory.retrieve_memories(user_question, top_k=limit)

            # 5. Theme Clusters
            context_data["theme_clusters"] = self.db_manager.analyze_theme_clusters(session_id, max_clusters=limit)

            # 6. Emotional Trajectory
            context_data["emotional_trajectory"] = self.db_manager.analyze_emotional_vector_trajectory(session_id)

            # 7. Hot Topics (Assuming _identify_hot_topics exists)
            if hasattr(self, "_identify_hot_topics"):
                context_data["hot_topics"] = self._identify_hot_topics(user_question, query_embedding)

            logger.debug("Retrieved contextual data for session %s", session_id)
            return context_data

        except Exception as e:
            logger.error("Error retrieving contextual data for session %s: %s", session_id, e)
            logger.error(traceback.format_exc())
            return {
                "conversation_context": "",
                "knowledge_items": [],
                "past_interactions": [],
                "similar_memories": [],
                "theme_clusters": [],
                "emotional_trajectory": [],
                "hot_topics": [],
            }
