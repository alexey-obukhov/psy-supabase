"""
ResponseGenerator Module
=======================

This module manages the generation of therapeutic responses based on user questions,
context, and potential pain points. It handles the selection of appropriate templates,
content safety filtering, and enhancement of responses with psychological insights.

Classes:
-------
ResponseGenerator: Generates therapeutic responses with appropriate context and templates

Dependencies:
------------
TextGenerator: For LLM-based text generation
DatabaseManager: For saving interactions and retrieving conversation history
PromptSelector: For template selection and question analysis
"""

from typing import Dict, List, Optional, Any, Tuple
import json
import traceback
import re
from datetime import datetime

from school_logging.log import ColoredLogger
from psy_supabase.core.text_generator import TextGenerator
from psy_supabase.core.database import DatabaseManager
from psy_supabase.utilities.prompt_selector import PromptSelector
from psy_supabase.core.dynamic_rag import DynamicRAGRetriever

# Set up logging
logger = ColoredLogger(__name__)

class ResponseGenerator:
    """
    Generates therapeutic responses with appropriate psychological context.

    This class is responsible for:
    1. Processing user questions for response generation
    2. Selecting appropriate templates based on psychological context
    3. Safety filtering of user input
    4. Determining the appropriate conversation context for database storage
    5. Saving interactions to the database

    Attributes:
        text_generator (TextGenerator): Text generator for response generation
        db_manager (DatabaseManager): Database manager for saving interactions
        prompt_selector (PromptSelector): Selector for therapeutic prompts
    """

    def __init__(self, text_generator: TextGenerator, db_manager: DatabaseManager, prompt_selector: PromptSelector):
        """Initialize with required dependencies."""
        self.text_generator = text_generator
        self.db_manager = db_manager
        self.prompt_selector = prompt_selector

    def is_valid_input(self, user_question: str) -> bool:
        """Check if the input is valid for processing."""
        if not user_question or user_question.strip() == "":
            return False

        # Check for high ratio of special characters
        special_char_count = len(re.sub(r'[a-zA-Z0-9\s]', '', user_question))
        if len(user_question) > 0 and special_char_count / len(user_question) > 0.5:
            logger.info(f"Detected high ratio of special characters in input ({special_char_count}/{len(user_question)})")
            return False

        # Check for excessively long inputs
        if len(user_question) > 1000:
            logger.info(f"Processing very long input ({len(user_question)} chars)")
            return False

        return True

    def get_default_response(self, user_question: str) -> str:
        """Get a default response for invalid inputs."""
        if not user_question or user_question.strip() == "":
            return "I'm here to help and support you. What would you like to talk about today?"

        # Handle high ratio of special characters
        special_char_count = len(re.sub(r'[a-zA-Z0-9\s]', '', user_question))
        if len(user_question) > 0 and special_char_count / len(user_question) > 0.5:
            return "I notice your message contains special characters. I'm here to support you with whatever you'd like to discuss. How can I help you today?"

        # Handle excessively long inputs
        if len(user_question) > 1000:
            return "Thank you for sharing so much detail. I'm here to help and support you. Which specific aspect would you like me to focus on first?"

        return "I'm here to help and support you. What would you like to talk about today?"

    def check_toxic_content(self, user_question: str, session_id: str) -> Optional[str]:
        """Check for toxic content and return response if toxic."""
        is_toxic = False

        # Check if text_generator has is_toxic method
        if hasattr(self.text_generator, 'is_toxic'):
            is_toxic = self.text_generator.is_toxic(user_question)

        if is_toxic:
            logger.warning(f"Toxic user input detected: {user_question[:50]}...")
            response = "I cannot respond to this type of content. Please use respectful and appropriate language."

            # Save interaction for toxic content
            metadata = {
                "model": getattr(self.text_generator, "model_name", "unknown"),
                "toxicity": is_toxic,
                "rejected": True
            }

            self.db_manager.save_interaction(
                context="content_policy_violation",
                question=user_question,
                answer=response,
                metadata=metadata,
                session_id=session_id
            )

            return response

        return None

    def initialize_tracking(self, question_id: Optional[int] = None) -> str:
        """Initialize tracking for the query."""
        tracking_id = question_id if question_id is not None else f"auto_{int(datetime.now().timestamp())}"
        return tracking_id

    def extract_psychological_topics(self, user_question: str) -> Dict:
        """Extract psychological topics and context from the user's question."""
        # Get detailed analysis of the question
        question_analysis = self.prompt_selector.analyze_question(user_question)
        detected_topic = question_analysis.get('topic', 'general')
        emotion = question_analysis.get('emotion')

        # Get more detailed category information
        category_info = self.prompt_selector.generate_category_info(user_question)

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
            topic_from_text = self.prompt_selector._determine_topic(category_info, user_question)
            if topic_from_text != "emotional support":
                extracted_topics.append(topic_from_text)
            else:
                extracted_topics.append("therapeutic support")

        # Limit to top 3 topics
        extracted_topics = extracted_topics[:3]
        logger.info(f"Extracted topics for RAG retrieval: {extracted_topics}")

        # Return structured context information
        return {
            "detected_topic": detected_topic,
            "emotion": emotion,
            "category_info": category_info,
            "extracted_topics": extracted_topics
        }

    def build_generation_context(self, user_question: str, session_id: str,
                               topics_context: Dict, pain_point_results: Dict,
                               query_embedding: List[float], dynamic_retriever: DynamicRAGRetriever,
                               hot_topics: List[Dict]) -> Dict:
        """Build the context for response generation."""
        # Extract required information
        extracted_topics = topics_context["extracted_topics"]
        detected_topic = topics_context["detected_topic"]
        emotion = topics_context["emotion"]

        # Create context for response generation
        context = {
            "user_question": user_question,
            "dynamic_retriever": dynamic_retriever,  # Pass the retriever object
            "use_dynamic_retrieval": True,  # Signal to use dynamic retrieval
            "session_id": session_id,
            "extracted_topics": extracted_topics,  # Add extracted topics for the template
            "psychological_context": {
                "topic": detected_topic,
                "emotion": emotion,
                "categories": list(topics_context["category_info"].keys())
            }
        }

        # Add pain point to context if detected
        if pain_point_results["pain_point_detected"] and pain_point_results["pain_point"]:
            context["pain_point"] = pain_point_results["pain_point"]

        # Add hot topics only if detected
        if hot_topics:
            context["hot_topics"] = hot_topics

        return context

    def generate_response_with_template(self, user_question: str, session_id: str,
                                      generation_context: Dict, pain_point_results: Dict,
                                      conversation_history: List[Dict]) -> str:
        """Generate a response using the appropriate template."""
        template_used = pain_point_results["template_used"]

        try:
            # Generate response with dynamic retrieval capability
            response = self.text_generator.generate_therapeutic_response_with_dynamic_retrieval(
                user_question=user_question,
                template_name=template_used,  # Use the appropriate template
                context=generation_context,
                conversation_history=conversation_history
            )

            # If response is None or empty, generate a fallback response
            if not response:
                logger.warning("Received empty response from text generator, using fallback")
                response = "I apologize, but I'm having trouble generating a response right now. Could you please try asking again?"

            return response

        except Exception as gen_error:
            logger.error(f"Error generating response with dynamic retrieval: {gen_error}")
            return "I apologize, but I'm experiencing a technical issue. Please try again with a different question."

    def determine_final_context(self, user_question: str, topics_context: Dict,
                              pain_point_results: Dict, metadata: Dict) -> Tuple[str, Dict]:
        """Determine the final context for database storage."""
        # Access needed information from previous steps
        detected_topic = topics_context.get("detected_topic")
        pain_point = pain_point_results.get("pain_point", {})
        approach_type = pain_point_results.get("approach_type")

        # Import the mapping function
        from psy_supabase.utilities.utils_mapping import map_approach_to_template

        # Now determine the context
        context = "therapeutic_dialogue"  # Default fallback

        # Get extracted topics from _determine_topic if needed
        extracted_topics = []
        category_info = self.prompt_selector.generate_category_info(user_question)
        if category_info:
            determined_topic = self.prompt_selector._determine_topic(category_info, user_question)
            if determined_topic:
                extracted_topics = [determined_topic]

        # Now determine the best context using our priority rules
        if detected_topic and detected_topic != "general":
            context = detected_topic.replace(" ", "_").lower()
            logger.info(f"Using detected_topic as context: {context}")
        elif extracted_topics and extracted_topics[0] != "therapeutic_dialogue":
            context = extracted_topics[0]
            logger.info(f"Using extracted_topic as context: {context}")
        elif approach_type and approach_type != "default_approach" and approach_type != "none":
            context = map_approach_to_template(approach_type)
            logger.info(f"Using approach_type mapped to therapeutic method: {context}")
        elif pain_point and 'topic' in pain_point and pain_point['topic']:
            context = pain_point['topic']
            logger.info(f"Using pain_point topic as context: {context}")
        else:
            logger.info("No specific topic found, using default context: therapeutic_dialogue")
            context = "therapeutic_dialogue"

        # Update metadata for saving
        updated_metadata = metadata.copy() if metadata else {}
        updated_metadata.update({
            "pain_point_detected": pain_point_results.get("pain_point_detected", False),
            "therapeutic_approach": approach_type,
            "template_used": pain_point_results.get("template_used", "default"),
            "recurring_themes": pain_point_results.get("pain_point", {}).get("recurring_terms", []),
            "pain_point_similarity": pain_point_results.get("pain_point", {}).get("count", 0),
            "context": context  # Add context to metadata for consistency
        })

        return context, updated_metadata

    def save_interaction(self, context: str, user_question: str, response: str,
                       metadata: Dict, session_id: str) -> None:
        """Save the interaction to the database with error handling."""
        # Use the determined context when saving
        logger.info(f"About to save_interaction with context: {context}, session_id: {session_id}")
        try:
            self.db_manager.save_interaction(
                context=context,
                question=user_question,
                answer=response if response else "No response generated",
                metadata=metadata,
                session_id=session_id
            )
            logger.info("save_interaction completed successfully")
        except Exception as e:
            logger.error(f"Error in save_interaction: {e}")
            logger.error(traceback.format_exc())

    def debug_context_determination(self, user_question: str, detected_topic: str,
                                  extracted_topics: List[str], approach_type: str,
                                  pain_point: Dict) -> None:
        """Debug function to log context determination process."""
        logger.debug("DEBUG CONTEXT DETERMINATION:")
        logger.debug(f"- User question: {user_question[:50]}...")
        logger.debug(f"- Detected topic: {detected_topic}")
        logger.debug(f"- Extracted topics: {extracted_topics}")
        logger.debug(f"- Approach type: {approach_type}")
        logger.debug(f"- Pain point: {str(pain_point)[:100]}...")
