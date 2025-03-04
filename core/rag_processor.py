from typing import List, Dict
import json
import logging
import numpy as np

from psy_supabase.core.database import DatabaseManager
from psy_supabase.utilities.prompt_selector import PromptSelector
from psy_supabase.core.text_generator import TextGenerator
from psy_supabase.utilities.safety_handler import SafetyHandler


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGProcessor:
    """Handles retrieval-augmented generation logic."""

    def __init__(self, db_manager: DatabaseManager, generator: TextGenerator, intelligent_processing_enabled: bool = True):
        self.db_manager = db_manager
        self.generator = generator
        self.safety_handler = SafetyHandler()  # Initialize safety handler
        self.prompt_selector = PromptSelector(generator)  # Initialize prompt selector here
        self.intelligent_processing_enabled = intelligent_processing_enabled

    def get_relevant_documents(self, query_embedding: List[float], table_name: str = "knowledge_base", top_k: int = 5) -> List[str]:
        """Retrieves the most relevant documents using cosine similarity."""
        try:
            all_docs = self.db_manager.get_all_documents_and_embeddings(table_name)
            if not all_docs:
                return []

            similarities = []
            for doc in all_docs:
                # Handle potential JSON parsing issues and ensure embedding is a list
                if isinstance(doc['embedding'], str):
                    try:
                        doc_embedding = json.loads(doc['embedding'])
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON for document ID {doc['id']}: {doc['embedding']}")
                        continue #skip this doc
                elif isinstance(doc['embedding'], list):
                    doc_embedding = doc['embedding']
                else:
                    print(f"Unexpected embedding type for document ID {doc['id']}: {type(doc['embedding'])}")
                    continue  # Skip

                if not isinstance(doc_embedding, list):
                    print(f"Embedding for document ID {doc['id']} is not a list: {doc_embedding}")
                    continue

                try:
                    # Convert to NumPy arrays for calculation
                    doc_embedding_np = np.array(doc_embedding, dtype=np.float32)
                    query_embedding_np = np.array(query_embedding, dtype=np.float32) # No tolist()

                    similarity = np.dot(query_embedding_np, doc_embedding_np) / (np.linalg.norm(query_embedding_np) * np.linalg.norm(doc_embedding_np))
                    similarities.append((doc['content'], similarity))

                except Exception as e:
                    print(f"Error in similarity calculation: {e}")
                    continue # Skip this document on error

            sorted_documents = sorted(similarities, key=lambda x: x[1], reverse=True)  # Sort by similarity (descending)
            top_documents = [doc[0] for doc in sorted_documents[:top_k]]
            
            # Add debug logging to see what documents are being retrieved
            for i, doc in enumerate(top_documents):
                logger.debug(f"Retrieved document {i+1}: {doc[:100]}...")  # Log first 100 chars
            
            return top_documents

        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []

    def generate_response(self, user_question: str, device: str, question_id: int = 0, session_id: str = "default_session") -> str:
        """Generates a response using RAG with enhanced prompt selection and stability measures."""
        try:
            if not self.intelligent_processing_enabled:
                logger.info("Intelligent processing disabled, generating simple response")
                return self.generate_simple_response(user_question)

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

            # Select the best prompt template with enhanced error handling
            try:
                template_name, enhanced_context = self.prompt_selector.select_prompt_template(user_question)
                topic = enhanced_context.get("detected_topic", "emotional_support")
                confidence = enhanced_context.get("confidence", 0.5)
                
                # Log detailed selection information
                logger.info(f"Selected template '{template_name}' for topic '{topic}' with confidence {confidence:.2f}")
                
                # Apply confidence threshold for template selection stability
                if confidence < 0.25:  # Lower threshold from 0.3 to 0.25
                    logger.warning(f"Low confidence in template selection ({confidence:.2f}), applying fallback")
                    
                    # Special case for breakups with depression
                    if ("broke up" in user_question.lower() or "breakup" in user_question.lower()) and "depress" in user_question.lower():
                        template_name = "Empathy and Validation"
                        enhanced_context["detected_topic"] = "Relationship Issues"
                        enhanced_context["confidence"] = 0.6  # Override with higher confidence
                        logger.info("Detected breakup with depression pattern, overriding with higher confidence")
                    elif "anxiety" in user_question.lower() or "worry" in user_question.lower():
                        template_name = "Affirmation and Reassurance"
                    elif "sad" in user_question.lower() or "depress" in user_question.lower():
                        template_name = "Empathy and Validation"
                    else:
                        template_name = "Empathy and Validation"  # Safe default
                    
                    # Update context with the overridden template
                    enhanced_context["detected_template"] = template_name
                    enhanced_context["template_override"] = "low_confidence_fallback"
            except Exception as e:
                logger.error(f"Error in template selection: {e}", exc_info=True)
                # Fallback to a safe template
                template_name = "Empathy and Validation"
                enhanced_context = {"detected_topic": "emotional_support", "error": str(e)}

            # Retrieve conversation history with robust error handling
            history = []
            try:
                history = self.db_manager.get_conversation_history(session_id)
            except Exception as e:
                logger.error(f"Error retrieving conversation history: {e}")
            
            # Format conversation history for the model
            previous_conversation = []
            if history:
                try:
                    for turn in history[-5:]:  # Last 5 turns
                        previous_conversation.append({
                            'user_message': turn.get('questionText', ''),
                            'ai_message': turn.get('answerText', '')
                        })
                except Exception as e:
                    logger.error(f"Error formatting conversation history: {e}")
            
            # Get relevant documents from knowledge base with tiered retrieval strategy
            try:
                query_embedding = self.generator.get_embedding(user_question)
                if query_embedding is None:
                    logger.error("Failed to generate question embedding")
                    return "I'm sorry, I encountered an error processing your question."

                query_embedding_list = query_embedding.tolist()
                
                # Try user-specific knowledge base first, then fall back to general knowledge base
                relevant_documents = []
                try:
                    # Try session-specific knowledge base
                    relevant_documents = self.get_relevant_documents(query_embedding_list, 
                                                                table_name=f"{session_id}.knowledge_base", 
                                                                top_k=3)
                    
                    # If no documents found, try general knowledge base
                    if not relevant_documents:
                        relevant_documents = self.get_relevant_documents(query_embedding_list, top_k=3)
                        
                    # If still no documents, try with lower similarity threshold
                    if not relevant_documents:
                        logger.info("No relevant documents found, trying with lower similarity threshold")
                        relevant_documents = self.get_relevant_documents(query_embedding_list, top_k=5)
                        
                except Exception as e:
                    logger.error(f"Error in document retrieval: {e}", exc_info=True)
                    # Continue without documents
                
                # Log retrieval results
                logger.info(f"Retrieved {len(relevant_documents)} relevant documents")
                
                # Process and add relevant documents to enhanced context
                if relevant_documents:
                    # Deduplicate and clean documents
                    clean_docs = []
                    seen_content = set()
                    for doc in relevant_documents:
                        # Extract first 100 chars as signature for deduplication
                        doc_signature = doc[:100].strip()
                        if doc_signature not in seen_content:
                            seen_content.add(doc_signature)
                            clean_docs.append(doc)
                    
                    enhanced_context["relevant_documents"] = clean_docs
            except Exception as e:
                logger.error(f"Error retrieving relevant documents: {e}", exc_info=True)
                # Continue without documents

            # Generate response with comprehensive error handling
            try:
                response = self.generator.generate_therapeutic_response(
                    question=user_question,
                    template_name=template_name,
                    enhanced_context=enhanced_context,
                    previous_conversation=previous_conversation
                )
            except Exception as e:
                logger.error(f"Error in response generation: {e}", exc_info=True)
                response = "I apologize, but I'm having difficulty processing your question right now. Would you mind sharing more about what's on your mind so I can better understand how to help?"

            # Check for problematic content
            try:
                if self.generator.is_toxic(response):
                    logger.warning("Generated toxic response, returning safe alternative")
                    response = "I want to be helpful, but I need to ensure my responses are appropriate. Could you rephrase your question or share more about what you're looking for help with?"
            except Exception as e:
                logger.error(f"Error in toxicity check: {e}")
                
            # Store interaction with enhanced metadata
            try:
                effectiveness_metrics = self.prompt_selector.analyze_response_effectiveness(
                    user_question, response, template_name)
            except Exception as e:
                logger.error(f"Error analyzing response: {e}")
                effectiveness_metrics = {"error": str(e)}

            data_point = {
                'context': f"Template: {template_name}, Topic: {topic}",
                'question': user_question,
                'answer': response,
                'metadata': json.dumps({
                    'topic': topic,
                    'prompt_template': template_name,
                    'effectiveness': effectiveness_metrics.get('metrics', {}),
                    'questionID': str(question_id),
                    'confidence': confidence,
                    'has_relevant_docs': len(relevant_documents) > 0 if 'relevant_documents' in locals() else False
                })  
            }
            self.db_manager.add_interaction(data_point, session_id)
            
            return response

        except Exception as e:
            logger.error(f"Error in generate_response: {e}", exc_info=True)
            return "I apologize for the inconvenience. I encountered an error while processing your question. Could you try expressing your concern in a different way?"

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
                topic = metadata.get('topic', 'emotional_support')
                template_used = metadata.get('prompt_template', 'Others')
                
                # Create training example with format:
                # Instruction: The task description
                # Input: User's query
                # Output: Model's response
                training_example = {
                    "instruction": f"You are a therapeutic AI assistant skilled in {template_used}. Provide a compassionate and helpful response about {topic}.",
                    "input": interaction['question'],
                    "output": interaction['answer'],
                    "metadata": {
                        "topic": topic,
                        "therapeutic_approach": template_used,
                        "effectiveness": metadata.get('effectiveness', {})
                    }
                }
                
                training_examples.append(training_example)
                
            except Exception as e:
                logger.error(f"Error creating training example: {e}")
                continue
                
        return training_examples
