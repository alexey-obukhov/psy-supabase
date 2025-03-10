from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import logging
import numpy as np
import traceback

from psy_supabase.core.database import DatabaseManager
from psy_supabase.utilities.prompt_selector import PromptSelector
from psy_supabase.core.text_generator import TextGenerator
from psy_supabase.utilities.safety_handler import SafetyHandler


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        from psy_supabase.core.model_manager import EmbeddingProviderAdapter
        self.embedding_provider = EmbeddingProviderAdapter()
        
        # Constants for vector retrieval optimization
        self.SIMILARITY_THRESHOLD = 0.7       # Minimum similarity for relevant documents
        self.MAX_KNOWLEDGE_CHARS = 500        # Max characters for knowledge context
        self.MAX_CONVERSATION_EXCHANGES = 2   # Max conversation exchanges to include
        self.VECTOR_CACHE_ENABLED = True      # Enable vector caching for similar questions

    def get_relevant_documents(self, query_embedding: List[float], table_name: str = "knowledge_base", top_k: int = 5) -> List[Dict]:
        """
        Retrieves the most relevant documents using pgvector similarity.
        
        OPTIMIZED: Uses direct pgvector similarity search in database instead of Python-side calculation.
        """
        try:
            # Let pgvector handle the similarity calculation in the database
            similar_docs = self.db_manager.find_similar_documents(
                embedding=query_embedding, 
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

    def generate_response(self, user_question: str, device: str = "cpu", question_id: int = 0, session_id: str = "default_session") -> str:
        """
        Generate a response using RAG with enhanced psychological memory and optimized pgvector retrieval.
        
        Args:
            user_question: The user's question
            device: Device to use for generation
            question_id: Question ID for tracking
            session_id: Session ID for database operations
            
        Returns:
            str: Generated response
        """
        try:
            # Initialize default values
            topic = "general_support"  # Default topic
            template_name = "Basic Answer"  # Default template
            confidence = 0.5  # Default confidence
            
            # Safety check first with detailed logging
            is_harmful, safety_response, metadata = self.safety_handler.process_input(user_question)
            if is_harmful:
                logger.warning(f"Safety filter triggered: {metadata.get('category', 'unknown')}")
                data_point = {
                    'context': "Safety response",
                    'question': user_question,
                    'answer': safety_response,
                    'metadata': metadata or {'topic': 'Safety Response', 'questionID': question_id}
                }
                self.db_manager.add_interaction(data_point, session_id)
                return safety_response
                
            # OPTIMIZATION: Check for cached embedding of similar questions
            query_embedding = None
            if self.VECTOR_CACHE_ENABLED:
                cached_embedding = self.db_manager.find_similar_question_embedding(
                    user_question, 
                    session_id=session_id,
                    similarity_threshold=0.92  # Only use cache for very similar questions
                )
                if cached_embedding:
                    logger.info("Using cached embedding from similar previous question")
                    query_embedding = cached_embedding
            
            # Generate embeddings for the question if not cached
            if query_embedding is None:
                query_embedding = self.embedding_provider.generate_embedding(user_question)
                
            if not query_embedding:
                logger.error("Failed to generate embedding for question")
                query_embedding = [0.0] * 768  # Default embedding dimension
                
            # OPTIMIZED pgvECTOR RETRIEVAL:
            # 1. Find similar documents with threshold filtering done in the database
            similar_documents = self.db_manager.find_similar_documents(
                embedding=query_embedding, 
                limit=5,
                min_similarity=self.SIMILARITY_THRESHOLD  # Apply database-side filtering
            )
            
            # 2. Find relevant memories with pgvector similarity and threshold filtering
            similar_memories = self.db_manager.find_similar_memories(
                embedding=query_embedding, 
                session_id=session_id, 
                limit=3,
                threshold=0.65  # Slightly lower threshold for memories to capture broader context
            )
            
            # 3. Use pgvector clustering to identify conversation themes
            theme_clusters = self.db_manager.analyze_theme_clusters(
                session_id=session_id,
                min_similarity=0.7,
                max_clusters=2  # Limit clusters for efficiency
            )
            
            # 4. Vector-based emotional trajectory analysis
            emotional_trajectory = self.db_manager.analyze_emotional_vector_trajectory(
                session_id=session_id, 
                # window_size=5  # Only analyze recent interactions
            )
                
            # Build enhanced context with relevant documents and conversation history
            enhanced_context = self._enhance_context_with_relevant_documents(
                user_question, 
                query_embedding, 
                session_id
            )
            
            # Add psychological insights based on vector-retrieved data
            psychological_context = self._build_psychological_context(
                similar_memories, 
                theme_clusters, 
                emotional_trajectory
            )
            
            # Template selection
            try:
                # Select the appropriate template based on the question
                template_name, template_context = self.prompt_selector.select_prompt_template(user_question)
                
                # Merge template context with our enhanced context
                enhanced_context.update(template_context)
                
                # Add psychological context 
                enhanced_context["psychological_context"] = psychological_context
                
                # Add topic details
                topic = template_context.get("detected_topic", topic)
                confidence = template_context.get("confidence", confidence)
                
            except Exception as e:
                logger.error(f"Error in template selection: {e}", exc_info=True)
                # Continue with the default template
            
            # Get conversation history - already included in enhanced_context from _enhance_context_with_relevant_documents
            conversation_history = self.db_manager.get_conversation_history(session_id)

            # Generate text response with optimized context
            try:
                response = self.text_generator.generate_therapeutic_response(
                    user_question, 
                    template_name, 
                    enhanced_context,
                    conversation_history[-self.MAX_CONVERSATION_EXCHANGES:] if conversation_history else None
                )
            except Exception as e:
                logger.error(f"Error in text generation: {e}", exc_info=True)
                response = "I apologize, but I encountered an error while processing your question. Could you try rephrasing it?"
            
            # Store interaction with optimized metadata
            try:
                # Extract relevant document context (limiting size)
                context_text = ""
                if similar_documents:
                    # Take only the first two most similar documents
                    context_docs = []
                    
                    for doc in similar_documents[:2]:
                        # Handle string objects
                        if isinstance(doc, str):
                            # Try to parse JSON if it's a JSON string
                            try:
                                import json
                                doc_dict = json.loads(doc)
                                content = doc_dict.get('content', '')[:150]
                                similarity = doc_dict.get('similarity', 0)
                            except:
                                # If parsing fails, use the string as content
                                content = doc[:150]  # Truncate if needed
                                similarity = 0.7  # Default above threshold
                        else:
                            # Normal dictionary case
                            content = doc.get('content', '')[:150]
                            similarity = doc.get('similarity', 0)
                            
                        if similarity >= self.SIMILARITY_THRESHOLD:
                            context_docs.append(content)
                            
                    context_text = ' | '.join(context_docs)
                
                # Extract the current emotional state if available
                emotional_state = psychological_context.get("current_emotional_state", "")
                
                # Store with metadata including vector context
                metadata = {
                    'topic': topic,
                    'template_used': template_name,
                    'confidence': confidence,
                    'vector_enriched': True,
                    'similarity_threshold': self.SIMILARITY_THRESHOLD,
                    'num_relevant_docs': len([d for d in similar_documents if d]) if similar_documents else 0,
                    'num_similar_memories': len(similar_memories) if similar_memories else 0
                }
                
                # Add emotional state if available
                if emotional_state:
                    metadata["emotional_state"] = emotional_state
                    
                # Store interaction
                self.db_manager.add_interaction({
                    'context': context_text[:500],  # Limit context size
                    'question': user_question,
                    'answer': response,
                    'metadata': metadata
                }, session_id)
                
                # OPTIMIZATION: Store embedding for future vector similarity
                try:
                    all_history = self.db_manager.get_conversation_history(session_id)
                    latest_history = all_history[-1:] if all_history else []
                    
                    if latest_history and len(latest_history) > 0:
                        interaction_id = latest_history[0].get('interaction_id')
                        if interaction_id and query_embedding:
                            self.db_manager.add_embedding_to_interaction(
                                interaction_id,
                                query_embedding,
                                session_id
                            )
                except Exception as e:
                    logger.error(f"Error storing embedding: {e}")
                    
            except Exception as e:
                logger.error(f"Error storing interaction: {e}")
                logger.error(traceback.format_exc())
                
            return response
        except Exception as e:
            logger.error(f"Error in generate_response: {e}")
            logger.error(traceback.format_exc())
            return "I apologize, but I encountered an error processing your question. Would you mind rephrasing it?"

    def generate_simple_response(self, user_question: str) -> str:
        """Generates a simple response without any preprocessing or context."""
        return self.generator.generate_text(user_question)

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
                # FIXED: Handle both dictionary and string responses
                for doc in similar_docs:
                    # Handle case where doc is a string (the response format issue)
                    if isinstance(doc, str):
                        # Try to parse JSON if it's a JSON string
                        try:
                            import json
                            doc_dict = json.loads(doc)
                            content = doc_dict.get('content', '')
                            similarity = doc_dict.get('similarity', 0)
                        except:
                            # If not JSON, use the string as content with default similarity
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
