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

    def __init__(self, db_manager: DatabaseManager, generator: TextGenerator):
        self.db_manager = db_manager
        self.generator = generator
        self.safety_handler = SafetyHandler()  # Initialize safety handler
        self.prompt_selector = PromptSelector(generator)  # Initialize prompt selector here

    def get_relevant_documents(self, query_embedding: List[float], table_name: str = "knowledge_base", top_k: int = 5) -> List[str]:
        """Retrieves the most relevant documents using cosine similarity (calculated in Python)."""
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
            return [doc[0] for doc in sorted_documents[:top_k]]

        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []

    def generate_response(self, user_question: str, device: str, question_id: int = 0) -> str:
        """Generates a response using RAG with enhanced prompt selection."""
        
        # Safety check first
        is_harmful, safety_response, metadata = self.safety_handler.process_input(user_question)
        if is_harmful:
            data_point = {
                'context': "Safety response",
                'question': user_question,
                'answer': safety_response,
                'metadata': metadata or {'topic': 'Safety Response', 'questionID': question_id}
            }
            self.db_manager.add_interaction(data_point)
            return safety_response

        # Select the best prompt template using your data_augmentation approach
        template_name, enhanced_context = self.prompt_selector.select_prompt_template(user_question)
        topic = enhanced_context.get("detected_topic", "emotional_support")  # Changed default from "General"
        
        # Retrieve conversation history
        history = self.db_manager.get_conversation_history()
        
        # After detecting the topic and before building the context
        primary_topic = enhanced_context.get("detected_topic", "emotional_support")  # Changed default

        # Get topic-specific history
        topic_history = self.db_manager.get_topic_interactions(primary_topic, limit=3)
        
        # Build context string
        context_parts = []

        # Add topic-specific context first if available
        if topic_history:
            topic_context = "".join(
                f"USER: {turn['questionText']}\nTHERAPIST: {turn['answerText']}\n"
                for turn in topic_history
            )
            context_parts.append(f"\n[Previous discussions about {primary_topic}]:\n{topic_context}")

        # Add general history
        max_history_turns = 3 if topic_history else 5  # Use fewer turns if we have topic history
        general_context = "".join(
            f"USER: {turn['questionText']}\nTHERAPIST: {turn['answerText']}\n"
            for turn in history[-max_history_turns:]
        )
        context_parts.append(general_context)

        # Combine all context parts
        context_string = "\n".join(context_parts)

        # Get embeddings and relevant documents
        query_embedding = self.generator.get_embedding(user_question)
        if query_embedding is None:
            return "I'm sorry, I encountered an error processing your question."

        query_embedding_list = query_embedding.tolist()
        relevant_documents = self.get_relevant_documents(query_embedding_list, top_k=3)
        knowledge_base_context = "\n\n".join(relevant_documents)

        # Get the selected template
        selected_template = prompt_templates[template_name]

        # Enhanced context with category information
        category_context = ""
        if enhanced_context.get("category_info"):
            category = enhanced_context["category_info"].get(template_name, "")
            if category:
                category_context = f"\nTherapeutic Approach: {template_name}\n{category}\n"

        # Format conversation history more clearly
        history_text = ""
        if history:
            history_text = "\n\n".join([
                f"USER: {turn['questionText']}\nTHERAPIST: {turn['answerText']}"
                for turn in history[-max_history_turns:]
            ])

        # Create a cleaner, more structured prompt
        prompt = f"""You are a compassionate AI therapist having a therapeutic conversation about {topic}.

{category_context}

PREVIOUS CONVERSATION:
{history_text}

KNOWLEDGE BASE:
{knowledge_base_context}

USER: {user_question}
THERAPIST:"""

        # Generate response
        response = self.generator.generate_text(prompt)

        if self.generator.is_toxic(response):
            return "I'm sorry, I can't respond to that in a helpful way."

        # Add a try/except block around the effectiveness analysis
        effectiveness_metrics = {}
        try:
            effectiveness_metrics = self.prompt_selector.analyze_response_effectiveness(
                user_question, response, template_name)
        except Exception as e:
            logger.error(f"Error analyzing response: {e}")
            effectiveness_metrics = {"error": str(e)}

        # Store interaction with enhanced metadata
        data_point = {
            'context': prompt,
            'question': user_question,
            'answer': response,
            'metadata': json.dumps({  # Explicitly convert to JSON string
                'topic': topic,
                'prompt_template': template_name,
                'effectiveness': effectiveness_metrics.get('metrics', {}),
                'questionID': str(question_id),  # Convert to string explicitly
            })  
        }
        self.db_manager.add_interaction(data_point)
        
        return response

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
