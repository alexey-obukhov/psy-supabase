from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import re
import traceback
from typeguard import typechecked

from school_logging.log import ColoredLogger
from psy_supabase.core.database import DatabaseManager
from psy_supabase.utilities.keep_words import keep_words
from psy_supabase.utilities.prompt_selector import PromptSelector
from psy_supabase.core.text_generator import TextGenerator
from psy_supabase.utilities.safety_handler import SafetyHandler
from psy_supabase.core.dynamic_rag import DynamicRAGRetriever
from psy_supabase.core.model_manager import EmbeddingProviderAdapter
from psy_supabase.utilities.utils_mapping import map_approach_to_template
from psy_supabase.utilities.embedding_utils import format_embedding_for_db


# Set up logging
logger = ColoredLogger(__name__)

class RAGProcessor:
    """Handles retrieval-augmented generation logic with optimized pgvector integration."""

    def __init__(self, db_manager: DatabaseManager, generator: TextGenerator, intelligent_processing_enabled: bool = True):
        self.db_manager = db_manager
        self.generator = generator
        self.text_generator = generator  # Alias for consistency
        self.safety_handler = SafetyHandler()
        self.prompt_selector = PromptSelector(generator)
        self.intelligent_processing_enabled = intelligent_processing_enabled

        # Use EmbeddingProviderAdapter instead
        self.embedding_provider = EmbeddingProviderAdapter()

        # Get the embedding dimension directly from the provider
        self.embedding_dimension = self.embedding_provider.get_embedding_dimension()

        # Constants for vector retrieval optimization
        self.SIMILARITY_THRESHOLD = 0.7       # Minimum similarity for relevant documents
        self.MAX_KNOWLEDGE_CHARS = 500        # Max characters for knowledge context
        self.MAX_CONVERSATION_EXCHANGES = 2   # Max conversation exchanges to include
        self.VECTOR_CACHE_ENABLED = True      # Enable vector caching for similar questions

    @typechecked
    def get_relevant_documents(self, query_embedding: List[float], table_name: str = "knowledge_base", top_k: int = 5) -> List[Dict]:
        """
        Retrieves the most relevant documents using pgvector similarity.

        OPTIMIZED: Uses direct pgvector similarity search in database instead of Python-side calculation.
        """
        try:
            # Let pgvector handle the similarity calculation in the database
            similar_docs = self.db_manager.find_similar_documents(
                embedding=query_embedding,
                table_name=table_name,
                limit=top_k,
                min_similarity=self.SIMILARITY_THRESHOLD  # Apply similarity threshold filter
            )

            # Format the results as needed
            relevant_documents = []
            for doc in similar_docs:
                if isinstance(doc, dict):
                    # Extract required fields and add to results
                    relevant_doc = {
                        'content': doc.get('content', ''),
                        'similarity': doc.get('similarity', 0),
                        'id': doc.get('id')
                    }
                    relevant_documents.append(relevant_doc)

                    # Log high-quality matches
                    if doc.get('similarity', 0) > 0.8:
                        logger.info(f"Found highly relevant document (similarity: {doc.get('similarity', 0):.3f})")

            # Add debug logging to see what documents are being retrieved
            for i, doc in enumerate(relevant_documents):
                content_preview = doc.get('content', '')[:100] + "..." if doc.get('content') else ""
                logger.debug(f"Retrieved document {i+1} (sim: {doc.get('similarity', 0):.3f}): {content_preview}")

            return relevant_documents

        except Exception as e:
            logger.error(f"Error retrieving documents with pgvector: {e}")
            return []

    @typechecked
    def generate_response(self,
                          user_question: str,
                          session_id: str = "default_session",
                          device: Optional[str] = None,
                          question_id: Optional[int] = None
                          ) -> str:
        """
        Generate a response using Dynamic RAG with psychological memory and pain point detection.
        
        Args:
            user_question: The user's question
            session_id: Session ID for database operations (primary identifier)
            device: Optional device to use for generation (used by embedding provider if specified)
            question_id: Optional question ID for tracking (can be used for analytics)
            
        Returns:
            str: Generated response
        """
        try:
            # Pass device to embedding provider if specified
            if device and hasattr(self.embedding_provider, 'set_device'):
                self.embedding_provider.set_device(device)

            # Track query ID if provided (useful for analytics)
            tracking_id = question_id if question_id is not None else f"auto_{int(datetime.now().timestamp())}"

            # Initialize the dynamic retriever
            dynamic_retriever = DynamicRAGRetriever(
                db_manager=self.db_manager,
                session_id=session_id,
                allow_dynamic_queries=True
            )

            # Track pain point detection and template usage
            pain_point_detected = False
            template_used = "dynamic_rag_therapy"  # Default template
            approach_type = "none"
            metadata = {
                "tracking_id": tracking_id,
                "session_id": session_id
            }

            # Process the query to get semantic meaning
            query_embedding = self.process_query(user_question, session_id)

            # Detect pain points based on query embedding
            pain_point_results = self.detect_pain_points_from_embedding(
                user_question,
                query_embedding,
                session_id,
                metadata
            )

            # Update variables with results
            pain_point_detected = pain_point_results['pain_point_detected']
            template_used = pain_point_results['template_used']
            approach_type = pain_point_results['approach_type']
            pain_point = pain_point_results['pain_point']

            # Use the PromptSelector to identify psychological topics
            selector = self.prompt_selector

            # Get detailed analysis of the question
            question_analysis = selector._analyze_question(user_question)
            detected_topic = question_analysis.get('topic', 'general')
            emotion = question_analysis.get('emotion')

            # Get more detailed category information
            category_info = selector.generate_category_info(user_question)

            # Log the analysis results
            logger.info(f"Question analysis: Topic={detected_topic}, Emotion={emotion}")
            logger.info(f"Categories: {list(category_info.keys())}")

            # Extract psychological topics for dynamic retrieval
            extracted_topics = []

            # Primary topic from question analysis
            if detected_topic and detected_topic != "general":
                extracted_topics.append(detected_topic.replace('_', ' '))

            # Add topics from categories (up to 3 total)
            for category in category_info.keys():
                # Convert category names to search terms
                if category == "Empathy and Validation":
                    if "depression" not in extracted_topics:
                        extracted_topics.append("depression")
                elif category == "Affirmation and Reassurance":
                    if "anxiety" not in extracted_topics:
                        extracted_topics.append("anxiety")
                elif category == "Trauma":
                    if "trauma" not in extracted_topics:
                        extracted_topics.append("trauma")
                elif "CBT" in category:
                    if "cognitive behavioral therapy" not in extracted_topics:
                        extracted_topics.append("cognitive behavioral therapy")

            # Set emotion as a topic if appropriate
            if emotion and len(extracted_topics) < 3:
                if emotion not in ["confusion", "surprise"]:  # Skip non-therapeutic emotions
                    extracted_topics.append(emotion)

            # Ensure we have at least one topic
            if not extracted_topics:
                topic_from_text = selector._determine_topic(category_info, user_question)
                if topic_from_text != "emotional_support":
                    extracted_topics.append(topic_from_text)
                else:
                    extracted_topics.append("therapeutic support")

            # Limit to top 3 topics
            extracted_topics = extracted_topics[:3]
            logger.info(f"Extracted topics for RAG retrieval: {extracted_topics}")

            # Only include relevant context in the prompt to keep it small
            context = {
                "user_question": user_question,
                "dynamic_retriever": dynamic_retriever,  # Pass the retriever object
                "use_dynamic_retrieval": True,  # Signal to use dynamic retrieval
                "session_id": session_id,
                "extracted_topics": extracted_topics,  # Add extracted topics for the template
                "psychological_context": {
                    "topic": detected_topic,
                    "emotion": emotion,
                    "categories": list(category_info.keys())
                }
            }

            # Add pain point to context if detected
            if pain_point_detected and pain_point:
                context["pain_point"] = pain_point

            # Add hot topics only if detected
            hot_topics = self._identify_hot_topics(user_question, query_embedding)
            if hot_topics:
                context["hot_topics"] = hot_topics

            # Generate response with dynamic retrieval capability
            try:
                response = self.text_generator.generate_therapeutic_response_with_dynamic_retrieval(
                    user_question=user_question,
                    template_name=template_used,  # Use the appropriate template
                    context=context,
                    conversation_history=self.get_recent_conversation_history(session_id, limit=2)
                )

                # If response is None or empty, generate a fallback response
                if not response:
                    logger.warning("Received empty response from text generator, using fallback")
                    response = "I apologize, but I'm having trouble generating a response right now. Could you please try asking again?"
            except Exception as gen_error:
                logger.error(f"Error generating response with dynamic retrieval: {gen_error}")
                response = "I apologize, but I'm experiencing a technical issue. Please try again with a different question."

            # Update metadata for saving
            metadata.update({
                "pain_point_detected": pain_point_detected,
                "therapeutic_approach": approach_type,
                "template_used": template_used,
                "recurring_themes": context.get('pain_point', {}).get('recurring_terms', []) if pain_point_detected else [],
                "pain_point_similarity": context.get('pain_point', {}).get('count', 0) if pain_point_detected else 0
            })

            # Save the interaction with this metadata
            try:
                # NEW: Also log pain point detection for analytics if detected
                if pain_point_detected:
                    self._log_pain_point_detection(
                        user_question,
                        context.get('pain_point', {}),
                        template_used
                    )

                save_result = self.db_manager.save_interaction(
                    context="therapeutic_dialogue",  # Using session_id as context
                    question=user_question,
                    answer=response if response else "No response generated",
                    metadata=metadata,
                    session_id=session_id
                )
                if not save_result:
                    logger.warning(f"Failed to save interaction for session {session_id}")
            except Exception as save_error:
                logger.error(f"Error saving interaction: {save_error}")

            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            logger.error(traceback.format_exc())
            return "I'm sorry, I encountered an error while generating a response. Could you please try again?"

    @typechecked
    def generate_simple_response(self, user_question: str) -> str:
        """Generates a simple response without any preprocessing or context."""
        return self.generator.generate_text(user_question)

    @typechecked
    def generate_training_examples(self,
                                   topic_filter=None,
                                   min_effectiveness=0.7,
                                   limit=100) -> List[Dict]:
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
            topic_filter=topic_filter,
            min_effectiveness=min_effectiveness,
            limit=limit
        )

        # Format for training
        training_examples = []
        for interaction in interactions:
            try:
                # Parse metadata
                metadata = interaction.get('metadata', {})
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)

                # Extract key information
                question = interaction.get('question', '')
                answer = interaction.get('answer', '')
                context = interaction.get('context', '')
                topic = metadata.get('topic', 'General')

                # Add psychological context if available
                psychological_context = ""
                if metadata.get('emotional_state'):
                    psychological_context += f"Emotional state: {metadata.get('emotional_state')}\n"

                if metadata.get('recurring_themes'):
                    themes = metadata.get('recurring_themes')
                    if isinstance(themes, list):
                        psychological_context += f"Recurring themes: {', '.join(themes)}\n"

                # Create a formatted training example
                example = {
                    'question': question,
                    'answer': answer,
                    'context': context,
                    'topic': topic,
                    'psychological_context': psychological_context.strip()
                }

                training_examples.append(example)
            except Exception as e:
                logger.error(f"Error formatting training example: {e}")
                continue

        return training_examples

    def _build_psychological_context(self, similar_memories: List[Dict],
                                     theme_clusters: List[Dict],
                                     emotional_trajectory: List[Dict]) -> Dict[str, Any]:
        """
        Build psychological context from vector-retrieved data.
        
        Args:
            similar_memories: Vector-similar past memories
            theme_clusters: pgvector theme clusters
            emotional_trajectory: Emotional vector trajectory
            
        Returns:
            Dict with psychological context
        """
        context = {}

        # Extract emotional signals from similar memories
        if similar_memories:
            emotional_signals = []
            recurring_topics = []

            for memory in similar_memories:
                # Extract emotion signals
                if memory.get('metadata') and isinstance(memory['metadata'], dict):
                    emotion = memory['metadata'].get('emotional_state')
                    if emotion:
                        emotional_signals.append(emotion)

                # Extract topics
                topic = memory.get('topic') or (memory['metadata'].get('topic') if memory.get('metadata') and isinstance(memory['metadata'], dict) else None)
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
            primary_themes = []
            for cluster in theme_clusters:
                if cluster.get('dominant_theme'):
                    primary_themes.append(cluster.get('dominant_theme'))

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
                valence_scores = [entry.get('valence', 0) for entry in emotional_trajectory
                                  if 'valence' in entry]

                if valence_scores and len(valence_scores) >= 2:
                    # Check if valence is generally improving
                    is_improving = valence_scores[-1] > valence_scores[0]
                    context["emotional_trend"] = "improving" if is_improving else "stable_or_declining"

        return context

    def _enhance_context_with_relevant_documents(self, user_question: str, question_embedding: List[float], session_id: str) -> Dict:
        """
        Enhance context with only the most relevant documents while maintaining a fixed context size.
        OPTIMIZED: Uses pgvector's similarity search for document selection, limiting data transfer.
        """
        try:
            # OPTIMIZED: Use pgvector search with threshold applied in database
            similar_docs = self.db_manager.find_similar_documents(
                embedding=question_embedding,
                limit=5,
                min_similarity=self.SIMILARITY_THRESHOLD
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
                            content = doc_dict.get('content', '')
                            similarity = doc_dict.get('similarity', 0)
                        except Exception as e:
                            # If not JSON, use the string as content with default similarity
                            logger.error("Error parsing JSON document: %s", e)
                            content = doc
                            similarity = 0.7  # Default similarity above threshold
                    else:
                        # Normal dictionary case
                        content = doc.get('content', '')
                        similarity = doc.get('similarity', 0)

                    # Database filtering should handle this, but double-check
                    if similarity >= self.SIMILARITY_THRESHOLD:
                        relevant_docs.append((content, similarity))

                # Build knowledge context, keeping track of total length
                total_length = 0
                final_docs = []

                for content, similarity in relevant_docs:
                    # Calculate how much this document would add
                    content_length = len(content)

                    # If adding this document would exceed our limit, stop
                    if total_length + content_length > self.MAX_KNOWLEDGE_CHARS:
                        # If this is the first document and it's too long, truncate it
                        if not final_docs:
                            truncated = content[:self.MAX_KNOWLEDGE_CHARS] + "..."
                            final_docs.append(truncated)
                        break

                    # Otherwise add the full document
                    final_docs.append(content)
                    total_length += content_length

                # Join the final set of documents
                knowledge_context = "\n\n".join(final_docs)

                # Log what we're including
                logger.info(f"Using {len(final_docs)} documents ({total_length} chars) for knowledge context")

            # OPTIMIZATION: Get conversation history with limited exchanges
            conversation_context = ""
            try:
                conversation_history = self.db_manager.get_conversation_history(session_id)

                if conversation_history and len(conversation_history) > 0:
                    recent_exchanges = conversation_history[-self.MAX_CONVERSATION_EXCHANGES:]

                    conversation_parts = []
                    for exchange in recent_exchanges:
                        q = exchange.get('questionText', exchange.get('question', ''))
                        a = exchange.get('answerText', exchange.get('answer', ''))
                        if q and a:
                            # Truncate if needed
                            q_short = q if len(q) < 100 else q[:97] + "..."
                            a_short = a if len(a) < 150 else a[:147] + "..."
                            conversation_parts.append(f"User: {q_short}")
                            conversation_parts.append(f"Assistant: {a_short}")

                    conversation_context = "\n".join(conversation_parts)

                    # Log the conversation context
                    if conversation_context:
                        logger.debug(f"Added conversation context ({len(conversation_context)} chars)")
            except Exception as e:
                logger.error(f"Error retrieving conversation history: {e}")
                # Continue with empty conversation context

            # Create enhanced context dictionary with consistent size limits
            enhanced_context = {
                'knowledge_context': knowledge_context.strip(),
                'conversation_context': conversation_context.strip(),
                'session_id': session_id,
                'has_knowledge': bool(knowledge_context.strip()),
                'has_conversation': bool(conversation_context.strip()),
                'vector_threshold': self.SIMILARITY_THRESHOLD,  # Add threshold info for reference
                'user_question': user_question  # Add the user's question for reference
            }

            # Log context sizes
            logger.info(f"Knowledge context: {len(knowledge_context)} chars from vector similarity search")
            logger.info(f"Conversation context: {len(conversation_context)} chars")
            logger.info(f"Total prompt context: {len(knowledge_context) + len(conversation_context)} chars")

            return enhanced_context
        except Exception as e:
            logger.error(f"Error enhancing context: {e}")
            logger.error(traceback.format_exc())
            return {
                'knowledge_context': "",
                'conversation_context': "",
                'session_id': session_id,
                'has_knowledge': False,
                'has_conversation': False,
                'user_question': user_question  # Include user question even in error case
            }

    def detect_repetition_pattern(self, original_question: str, current_question: str, similar_questions: List[Dict]) -> Dict:
        """
        Analyze repetition patterns in similar questions to detect psychological fixation.
        
        Args:
            original_question: The first occurrence of this question
            current_question: The current question
            similar_questions: List of similar questions identified
            
        Returns:
            Dictionary with repetition pattern data
        """
        # Count occurrences of highly similar questions
        high_similarity_count = sum(1 for q in similar_questions if q['similarity'] > 0.7)

        # Extract key terms that appear in both original and current question
        original_terms = set(re.findall(r'\b\w+\b', original_question.lower()))
        current_terms = set(re.findall(r'\b\w+\b', current_question.lower()))

        recurring_terms = original_terms.intersection(current_terms)

        significant_terms = [term for term in recurring_terms if term not in keep_words and len(term) > 2]

        return {
            'count': high_similarity_count,
            'recurring_terms': list(significant_terms),
            'is_fixation': high_similarity_count >= 3,  # Consider it fixation if asked 3+ times
            'intensity': min(high_similarity_count / 5, 1.0)  # Scale intensity, max 1.0
        }

    def _generate_pain_point_approach(self, original_question: str, current_question: str,
                                      emotions: List[Dict], repetition_pattern: Dict) -> Dict:
        """
        Generate a specialized therapeutic approach when a pain point is detected.
        
        Args:
            original_question: The first occurrence of this question
            current_question: The current question being asked
            emotions: Emotional response data
            repetition_pattern: Repetition pattern analysis
            
        Returns:
            Dictionary with therapeutic approach data
        """
        # Determine if this seems to be a fixation pattern
        is_fixation = repetition_pattern.get('is_fixation', False)
        recurring_terms = repetition_pattern.get('recurring_terms', [])

        # Analyze emotional tone from emotion data
        emotional_tone = "neutral"
        if emotions:
            # Get the most common emotion
            emotion_counter = {}
            for e in emotions:
                emotion = e.get('emotional_state', '').lower()
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
            if emotional_tone in ['anxious', 'worried', 'fear', 'anxiety']:
                approach_type = "anxiety_exploration"
                guidance_question = f"I notice you've mentioned {', '.join(recurring_terms[:2])} several times. " \
                                    f"These topics seem to cause you anxiety. Could you tell me what feels most " \
                                    f"overwhelming about this situation?"

            elif emotional_tone in ['sad', 'depressed', 'grief', 'depression']:
                approach_type = "grief_reflection"
                guidance_question = f"You've brought up {', '.join(recurring_terms[:2])} multiple times, and I sense " \
                                    f"some sadness there. What feelings come up for you when you think about this?"

            else:
                approach_type = "gentle_refocus"
                guidance_question = f"I've noticed we've discussed {', '.join(recurring_terms[:2])} several times. " \
                                    f"I wonder if we could explore what makes this particularly important for you right now?"
        else:
            # Not a fixation, but still a pain point - use a lighter approach
            approach_type = "exploratory"
            guidance_question = f"I notice that {', '.join(recurring_terms[:2]) if recurring_terms else 'this topic'} " \
                                f"seems meaningful to you. Could you share more about how it affects you?"

        return {
            'approach_type': approach_type,
            'emotional_tone': emotional_tone,
            'guidance_question': guidance_question,
            'should_redirect': is_fixation,  # Redirect the conversation if we detect fixation
        }

    @typechecked
    def enhance_response_with_pain_point_guidance(self, response: str, pain_point_data: Dict) -> str:
        """
        Enhance the therapeutic response with specialized guidance when a pain point is detected.
        
        Args:
            response: The original response
            pain_point_data: Pain point detection data
            
        Returns:
            Enhanced response with pain point guidance
        """
        if not pain_point_data or not pain_point_data.get('detected'):
            return response

        # Get the suggested approach
        approach = pain_point_data.get('suggested_approach', {})
        guidance_question = approach.get('guidance_question', '')

        if not guidance_question:
            return response

        # Determine how to enhance the response based on approach type
        approach_type = approach.get('approach_type', 'exploratory')

        if approach.get('should_redirect'):
            if approach_type == 'anxiety_exploration':
                enhanced_response = response.rstrip() + "\n\nI notice this topic seems to cause anxiety. " + guidance_question
            elif approach_type == 'grief_reflection':
                enhanced_response = response.rstrip() + "\n\nI sense some sadness in this topic. " + guidance_question
            else:
                enhanced_response = response.rstrip() + "\n\n" + guidance_question
        # For exploratory approaches, append the guidance more gently
        else:
            # Add the guidance question at the end
            enhanced_response = response.rstrip() + "\n\n" + guidance_question

        return enhanced_response

    @typechecked
    def get_contextual_data(self, question: str, session_id: str, max_context_chars: int = 1000) -> Dict:
        """
        Get contextualized data for a user question with database-side vector processing.

        Args:
            question: The user's question
            session_id: Session identifier
            max_context_chars: Maximum context length to return

        Returns:
            Dict containing knowledge and conversation context
        """
        try:
            # DATABASE-FIRST APPROACH:
            # Let PostgreSQL handle the vector similarity search instead of loading everything to RAM/GPU

            # 1. Generate embedding for the question using the EmbeddingProviderAdapter
            question_embedding = self.embedding_provider.generate_embedding(question)
            if question_embedding is None:
                logger.warning("Could not generate embedding for question")
                return {"knowledge_context": "", "conversation_context": ""}

            # 2. Convert embedding to the format expected by find_similar_documents_via_rpc
            # The embedding_provider returns a tensor, so we need to convert it to a list
            if hasattr(question_embedding, 'cpu') and callable(getattr(question_embedding, 'cpu')):
                # It's a torch tensor, convert to list
                embedding_list = question_embedding.cpu().numpy().tolist()
                # If it's a 2D tensor with one row, extract the row
                if isinstance(embedding_list, list) and len(embedding_list) == 1:
                    embedding_list = embedding_list[0]
            else:
                # It might already be a list or numpy array
                embedding_list = question_embedding

            # 3. Use a parameterized SQL query to find similar documents DIRECTLY in PostgreSQL
            similar_docs = self.db_manager.find_similar_documents_via_rpc(
                session_id=session_id,
                embedding=embedding_list,  # Properly formatted embedding
                similarity_threshold=0.7,
                limit=3  # Just get top 3 most relevant docs
            )

            # 4. Construct knowledge context from the results PostgreSQL returns
            knowledge_context = ""
            if similar_docs and len(similar_docs) > 0:
                # Only take as much as we need to stay under max_context_chars
                remaining_chars = max_context_chars
                for doc in similar_docs:
                    content = doc.get("content", "")
                    if len(content) <= remaining_chars:
                        knowledge_context += content + "\n\n"
                        remaining_chars -= len(content) + 2
                    else:
                        # Take a partial document if we're running out of space
                        knowledge_context += content[:remaining_chars] + "..."
                        break

                logger.info(f"Knowledge context: {len(knowledge_context)} chars from vector similarity search")

            # 5. Get minimal conversation context
            conversation_context = ""
            conversation_history = self.db_manager.get_conversation_history(session_id)
            if conversation_history and len(conversation_history) > 0:
                # Only take the last 2 exchanges to limit context size
                recent_history = conversation_history[-2:] if len(conversation_history) > 2 else conversation_history

                # Format as text, but be strict about length limits
                for item in recent_history:
                    q = item.get("questionText", "")[:150]  # Limit question length
                    a = item.get("answerText", "")[:200]    # Limit answer length
                    if q and a:
                        conversation_context += f"User: {q}\nAssistant: {a}\n\n"

                logger.info(f"Conversation context: {len(conversation_context)} chars")

            # 6. Ensure overall context stays within limits
            total_context_chars = len(knowledge_context) + len(conversation_context)
            logger.info(f"Total prompt context: {total_context_chars} chars")

            return {
                "knowledge_context": knowledge_context,
                "conversation_context": conversation_context
            }

        except Exception as e:
            logger.error(f"Error getting contextual data: {e}")
            logger.error(traceback.format_exc())
            return {"knowledge_context": "", "conversation_context": ""}

    @typechecked
    def process_query(self, user_question: str, session_id: Optional[str] = None) -> List[float]:
        """
        Process a user query to generate an embedding.
        Implements caching for similar previous questions.
        
        Args:
            user_question: The user's question text
            session_id: Optional session ID for cache lookup
            
        Returns:
            List[float]: The embedding vector
        """
        try:
            # Check if we can reuse a similar question's embedding (optimization)
            if self.VECTOR_CACHE_ENABLED and session_id:
                cached_embedding = self.db_manager.find_similar_question_embedding(
                    user_question,
                    session_id=session_id,
                    similarity_threshold=0.92  # High threshold for reuse
                )

                if cached_embedding:
                    logger.info("Using cached embedding from similar previous question")
                    return cached_embedding

            # Generate a new embedding
            embedding = self.embedding_provider.generate_embedding(user_question)

            # Convert embedding to list format if needed
            if hasattr(embedding, 'cpu') and callable(getattr(embedding, 'cpu')):
                # It's a torch tensor, convert to list
                embedding_list = embedding.cpu().numpy().tolist()
                # If it's a 2D tensor with one row, extract the row
                if isinstance(embedding_list, list) and len(embedding_list) == 1:
                    embedding_list = embedding_list[0]
            else:
                # It might already be a list or numpy array
                embedding_list = embedding

            return embedding_list

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            logger.error(traceback.format_exc())
            # Return a zero embedding as fallback (will likely not match anything)
            return [0.0] * self.embedding_dimension

    def _identify_hot_topics(self, user_question: str, query_embedding: List[float]) -> List[Dict]:
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
            themes = ["anxiety", "depression", "stress", "relationships",
                      "trauma", "grief", "self-esteem", "identity"]

            # Check if any of these themes are directly mentioned
            user_question_lower = user_question.lower()

            for theme in themes:
                if theme in user_question_lower:
                    hot_topics.append({
                        "topic": theme,
                        "relevance": 0.95,  # High relevance for direct mentions
                        "source": "direct_mention"
                    })

            # If we found direct mentions, return those
            if hot_topics:
                return hot_topics

            # Otherwise, try vector search
            # This could use pgvector to find similar topics in your knowledge base
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
                    check_query = "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'hot_topics');"
                    check_result = self.db_manager.supabase.rpc('sql', {'command': check_query}).execute()

                    if check_result.data and (check_result.data[0] == 't' or check_result.data[0] is True):
                        # Table exists, query it
                        result = self.db_manager.supabase.rpc('sql', {'command': query}).execute()

                        if result.data:
                            for row in result.data:
                                if isinstance(row, dict):
                                    hot_topics.append({
                                        "topic": row.get("topic", ""),
                                        "relevance": row.get("relevance", 0),
                                        "source": "vector_similarity"
                                    })
                                elif isinstance(row, str):
                                    # Parse CSV-formatted response
                                    parts = row.split(',')
                                    if len(parts) >= 3:
                                        hot_topics.append({
                                            "topic": parts[1],
                                            "relevance": float(parts[2]) if parts[2].replace('.','',1).isdigit() else 0,
                                            "source": "vector_similarity"
                                        })
                except Exception as inner_e:
                    logger.warning(f"Error finding hot topics: {inner_e}")
                    # Continue without hot topics

            return hot_topics

        except Exception as e:
            logger.error(f"Error identifying hot topics: {e}")
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
            recent_history = conversation_history[-limit:] if len(conversation_history) > limit else conversation_history

            # Format the conversation history for the generator
            formatted_history = []
            for item in recent_history:
                # Extract question and answer
                q = item.get('questionText', item.get('question', ''))
                a = item.get('answerText', item.get('answer', ''))

                # Include metadata if available
                metadata = item.get('metadata', {})
                if isinstance(metadata, str):
                    try:
                        # Try to parse metadata if it's a string
                        metadata = json.loads(metadata)
                    except Exception as e:
                        logger.error("Error parsing metadata: %s", e)
                        metadata = {}

                # Create formatted entry
                entry = {
                    'question': q,
                    'answer': a,
                    'metadata': metadata,
                    'timestamp': item.get('created_at', item.get('timestamp', ''))
                }

                # For psychological work, preserve the FULL text of the exchanges
                # DO NOT truncate text here - it's critical for psychological continuity
                formatted_history.append(entry)

            return formatted_history

        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            logger.error(traceback.format_exc())
            return []

    @typechecked
    def detect_pain_points_from_embedding(self,
                                          user_question: str,
                                          query_embedding: List[float],
                                          session_id: str,
                                          metadata: Dict
                                          ) -> Dict:
        """
        Detect pain points using vector similarity and extract therapeutic approach.
        
        Args:
            user_question: The user's question
            query_embedding: Vector embedding of the question
            session_id: Session identifier
            metadata: Metadata dictionary to update with pain point information
            
        Returns:
            Dict containing pain point information and template to use
        """
        try:
            # Initialize default return values
            result = {
                'pain_point': {},
                'pain_point_detected': False,
                'template_used': "dynamic_rag_therapy",  # Default template
                'approach_type': "none"
            }

            # Skip if no embedding
            if query_embedding is None:
                return result

            # Detect pain points using database manager
            pain_point = self.db_manager.identify_potential_pain_points(
                user_question,
                query_embedding,
                session_id
            )

            # If pain point detected, extract information
            if pain_point and pain_point.get('detected'):
                logger.info(f"Pain point detected with similarity {pain_point.get('similarity', 0)}")

                # Add to metadata for tracking
                metadata["pain_point_detected"] = True
                result['pain_point_detected'] = True
                metadata["pain_point_similarity"] = pain_point.get("similarity", 0)

                # Get recurring terms if available
                if "repetition_pattern" in pain_point and isinstance(pain_point["repetition_pattern"], dict):
                    recurring_terms = pain_point["repetition_pattern"].get("recurring_terms", [])
                    metadata["recurring_themes"] = recurring_terms

                # Add suggested approach info
                if "suggested_approach" in pain_point and isinstance(pain_point["suggested_approach"], dict):
                    approach = pain_point["suggested_approach"]

                    # Store full suggested approach
                    metadata["suggested_approach"] = approach

                    # Extract approach_type for direct access
                    if "approach_type" in approach:
                        approach_type = approach["approach_type"]
                        metadata["approach_type"] = approach_type
                        result['approach_type'] = approach_type

                        # Map approach_type to template name
                        template_used = map_approach_to_template(approach_type)
                        metadata["template_used"] = template_used
                        result['template_used'] = template_used
                        logger.info(f"Selected template {template_used} from approach {approach_type}")

                    # Extract name for therapeutic approach
                    if "name" in approach:
                        metadata["therapeutic_approach"] = approach["name"]

                # Store pain point in result
                result['pain_point'] = pain_point

            return result

        except Exception as e:
            logger.error(f"Error in pain point detection: {e}")
            logger.error(traceback.format_exc())
            return {
                'pain_point': {},
                'pain_point_detected': False,
                'template_used': "dynamic_rag_therapy",
                'approach_type': "none"
            }

    def _log_pain_point_detection(self, user_question, pain_point, template_used):
        """Log pain point detection for analysis."""
        try:
            # Log structured data for later analysis
            metadata = {
                'event_type': 'pain_point_detected',
                'pain_point': pain_point.get('pain_point', ''),
                'recurring_terms': pain_point.get('recurring_terms', []),
                'count': pain_point.get('count', 0),
                'severity': pain_point.get('severity', ''),
                'template_used': template_used,
                'approach': pain_point.get('approach', {}).get('name', '')
            }

            # Create a special log entry in interactions table
            self.db_manager.save_interaction(
                context="Pain Point System",
                question=user_question,
                answer="Pain point detection triggered",
                metadata=metadata,
                session_id="default_session"  # Use consistent session ID, not None
            )

            logger.info(
                f"Pain point detected: {pain_point.get('pain_point', 'unknown')} "
                f"(count: {pain_point.get('count', 0)}, severity: {pain_point.get('severity', 'unknown')})"
            )
        except Exception as e:
            logger.error(f"Error logging pain point detection: {e}")
