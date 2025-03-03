from text_generator import TextGenerator
from database import DatabaseManager
from typing import List, Dict
from utilities.therapeutic_promt import prompt_templates
from prompt_selector import PromptSelector  # Add this import
import json
import logging
import numpy as np
from flask import g
from safety_handler import SafetyHandler


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
        """Generates a response using RAG with enhanced prompt selection."""
        try:
            if not self.intelligent_processing_enabled:
                logger.info("Intelligent processing disabled, generating simple response")
                return self.generate_simple_response(user_question)

            # Safety check first
            is_harmful, safety_response, metadata = self.safety_handler.process_input(user_question)
            if is_harmful:
                data_point = {
                    'context': "Safety response",
                    'question': user_question,
                    'answer': safety_response,
                    'metadata': metadata or {'topic': 'Safety Response', 'questionID': question_id}
                }
                self.db_manager.add_interaction(data_point, session_id)
                return safety_response

            # Select the best prompt template using your data_augmentation approach
            template_name, enhanced_context = self.prompt_selector.select_prompt_template(user_question)
            topic = enhanced_context.get("detected_topic", "emotional_support")
            
            # Log what template and topic were selected 
            logger.info(f"Selected template '{template_name}' for topic '{topic}'")
            
            # Retrieve conversation history
            history = self.db_manager.get_conversation_history(session_id)
            
            # Format conversation history for the model
            previous_conversation = []
            if history:
                for turn in history[-5:]:  # Last 5 turns
                    previous_conversation.append({
                        'user_message': turn['questionText'],
                        'ai_message': turn['answerText']
                    })
            
            # Get relevant documents from knowledge base
            query_embedding = self.generator.get_embedding(user_question)
            if query_embedding is None:
                return "I'm sorry, I encountered an error processing your question."

            query_embedding_list = query_embedding.tolist()
            try:
                relevant_documents = self.get_relevant_documents(query_embedding_list, 
                                                                 table_name=f"{session_id}.knowledge_base", 
                                                                 top_k=3)
                if not relevant_documents:
                    relevant_documents = self.get_relevant_documents(query_embedding_list, top_k=3)
            except Exception as e:
                logger.error(f"Error retrieving relevant documents: {e}")
                relevant_documents = []

            # Process and add relevant documents to enhanced context
            if relevant_documents:
                enhanced_context["relevant_documents"] = relevant_documents
            
            # IMPORTANT CHANGE: Use generate_therapeutic_response instead of manually building prompt
            response = self.generator.generate_therapeutic_response(
                question=user_question,
                template_name=template_name,
                enhanced_context=enhanced_context,
                previous_conversation=previous_conversation
            )
            
            # Check for problematic content
            try:
                if self.generator.is_toxic(response):
                    logger.warning("Generated toxic response, returning safe alternative")
                    response = "I want to be helpful, but I need to ensure my responses are appropriate. Could you rephrase your question?"
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
                })  
            }
            self.db_manager.add_interaction(data_point, session_id)
            
            return response

        except Exception as e:
            logger.error(f"Error in generate_response: {e}", exc_info=True)
            return "I apologize for the inconvenience. I encountered an error while processing your question."

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
