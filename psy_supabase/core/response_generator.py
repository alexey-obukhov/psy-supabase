"""
ResponseGenerator Module
========================

This module manages the generation of therapeutic responses based on user questions,
context, and potential pain points. It handles the selection of appropriate templates,
content safety filtering, and enhancement of responses with psychological insights.

Classes:
--------
ResponseGenerator: Generates therapeutic responses with appropriate context and templates

Dependencies:
-------------
TextGenerator: For LLM-based text generation
DatabaseManager: For saving interactions and retrieving conversation history
PromptSelector: For template selection and question analysis
"""

import re
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from prismalog.log import get_logger
from typeguard import typechecked

from psy_supabase.config import DEFAULT_APPROACH, DEFAULT_EMOTION, DEFAULT_THEME, DEFAULT_TOPIC
from psy_supabase.core.database import DatabaseManager
from psy_supabase.core.dynamic_rag import DynamicRAGRetriever
from psy_supabase.core.text_generator import TextGenerator
from psy_supabase.utilities.prompt_selector import PromptSelector
from psy_supabase.utilities.semantic_emotion_detector import SemanticEmotionDetector
from psy_supabase.utilities.utils_mapping import map_approach_to_template

# Set up logging
logger = get_logger(__name__)


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

    @typechecked
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
        special_char_count = len(re.sub(r"[a-zA-Z0-9\s]", "", user_question))
        if len(user_question) > 0 and special_char_count / len(user_question) > 0.5:
            logger.info(
                "Detected high ratio of special characters in input (%d/%d)", special_char_count, len(user_question)
            )
            return False

        # Check for excessively long inputs
        if len(user_question) > 1000:
            logger.info("Processing very long input (%d chars)", len(user_question))
            return False

        return True

    def get_default_response(self, user_question: str) -> str:
        """Get a default response for invalid inputs."""
        if not user_question or user_question.strip() == "":
            return "I'm here to help and support you. What would you like to talk about today?"

        # Handle high ratio of special characters
        special_char_count = len(re.sub(r"[a-zA-Z0-9\s]", "", user_question))
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
        if hasattr(self.text_generator, "is_toxic"):
            is_toxic = self.text_generator.is_toxic(user_question)

        if is_toxic:
            logger.warning("Toxic user input detected: %s...", user_question[:50])
            response = "I cannot respond to this type of content. Please use respectful and appropriate language."

            # Save interaction for toxic content
            metadata = {
                "model": getattr(self.text_generator, "model_name", "unknown"),
                "toxicity": is_toxic,
                "rejected": True,
            }

            self.db_manager.save_interaction(
                context="content_policy_violation",
                question=user_question,
                answer=response,
                metadata=metadata,
                session_id=session_id,
            )

            return response

        return None

    def initialize_tracking(self, question_id: Optional[int] = None) -> str:
        """Initialize tracking for the query."""
        tracking_id = str(question_id) if question_id is not None else f"auto_{int(datetime.now().timestamp())}"
        return tracking_id

    def extract_psychological_topics(self, user_question: str) -> Dict:
        """Extract psychological topics and context from the user's question."""
        # Get detailed analysis of the question
        question_analysis = self.prompt_selector.analyze_question(user_question)
        detected_topic = question_analysis.get("topic", DEFAULT_TOPIC)
        emotion = question_analysis.get("emotion", DEFAULT_EMOTION)

        detector = SemanticEmotionDetector()
        standardized_topic = detector.get_standardized_topic(detected_topic)
        standardized_emotion = detector.get_standardized_emotion(emotion)

        # Log both original and standardized values
        logger.info(
            "Question analysis: Topic=%s (std: %s), Emotion=%s (std: %s)",
            detected_topic,
            standardized_topic,
            emotion,
            standardized_emotion,
        )

        # Get more detailed category information
        category_info = self.prompt_selector.generate_category_info(user_question)

        # Log the analysis results
        logger.info("Question analysis: Topic=%s, Emotion=%s", detected_topic, emotion)
        logger.info("Categories: %s", list(category_info.keys()))

        # Extract psychological topics for dynamic retrieval
        extracted_topics = []

        # Primary topic from question analysis
        if detected_topic and detected_topic != DEFAULT_TOPIC:
            extracted_topics.append(detected_topic)

        # Add topics from categories (up to 3 total)
        for category in category_info.keys():
            if category == "empathy_validation" and "depression" not in extracted_topics:
                extracted_topics.append("depression")
            elif category == "affirmation_reassurance" and "anxiety" not in extracted_topics:
                extracted_topics.append("anxiety")
            elif category == "trauma" and "trauma" not in extracted_topics:
                extracted_topics.append("trauma")
            elif "cbt" in category and "cognitive_behavioral_therapy" not in extracted_topics:
                extracted_topics.append("cognitive_behavioral_therapy")

        # Set emotion as a topic if appropriate
        if emotion and len(extracted_topics) < 3:
            if emotion not in ["confusion", "surprise"]:  # Skip non-therapeutic emotions
                extracted_topics.append(emotion)

        # Ensure we have at least one topic
        if not extracted_topics:
            topic_from_text = self.prompt_selector.determine_topic(category_info, user_question)
            if topic_from_text != DEFAULT_TOPIC:
                extracted_topics.append(topic_from_text)
            else:
                extracted_topics.append(DEFAULT_TOPIC)

        # Limit to top 3 topics
        extracted_topics = extracted_topics[:3]
        logger.info("Extracted topics for RAG retrieval: %s", extracted_topics)

        # Return structured context information with both original and standardized values
        return {
            "topic": detected_topic,
            "standardized_topic": standardized_topic,
            "emotion": emotion,
            "standardized_emotion": standardized_emotion,
            "category_info": category_info,
            "extracted_topics": extracted_topics,
        }

    def build_generation_context(
        self,
        user_question: str,
        session_id: str,
        topics_context: Dict,
        pain_point_results: Optional[Dict],
        query_embedding: Optional[List[float]] = None,
        dynamic_retriever: Optional[Any] = None,
        hot_topics: Optional[List[str]] = None,
    ) -> Dict:
        """Build the context for response generation with complete enhanced_context structure."""
        topics_context = topics_context or {}
        pain_point_results = pain_point_results or {}

        # Handle missing extracted_topics
        extracted_topics = topics_context.get("extracted_topics", [])
        if not extracted_topics and "topic" in topics_context:
            extracted_topics = [topics_context["topic"]]
        if not extracted_topics:
            extracted_topics = ["therapeutic_support"]

        # Get topic with fallback
        detected_topic = topics_context.get("topic", DEFAULT_TOPIC)
        emotion = topics_context.get("emotion", DEFAULT_EMOTION)

        is_pain_point_detected = pain_point_results.get("pain_point_detected", False) or (
            pain_point_results.get("pain_point", {}).get("detected", False)
        )

        # Create a fully structured enhanced_context with all required fields
        enhanced_context = {
            "has_knowledge": False,
            "has_conversation": False,
            "knowledge_context": "",
            "conversation_context": "",
            "psychological_context": {
                "topic": detected_topic,
                "emotion": emotion,
                "emotional_signals": [],
                "pain_point": None,
                "pain_point_detected": is_pain_point_detected,
            },
        }

        # Create the main context with the properly structured enhanced_context
        context = {
            "user_question": user_question,
            "dynamic_retriever": dynamic_retriever,
            "use_dynamic_retrieval": True,
            "session_id": session_id,
            "extracted_topics": extracted_topics,
            "query_embedding": query_embedding,
            "psychological_context": {
                "topic": detected_topic,
                "emotion": emotion,
                "categories": (
                    topics_context.get("category_info", {}).keys() if "category_info" in topics_context else []
                ),
            },
            # Add complete enhanced_context structure
            "enhanced_context": enhanced_context,
        }

        # Safely add optional data
        if pain_point_results.get("pain_point_detected") and pain_point_results.get("pain_point"):
            context["pain_point"] = pain_point_results["pain_point"]

            # Ensure psychological_context is a dictionary
            if "psychological_context" not in enhanced_context or not isinstance(
                enhanced_context["psychological_context"], dict
            ):
                enhanced_context["psychological_context"] = {}

            # Safely update psychological_context
            psychological_context = enhanced_context["psychological_context"]
            if isinstance(psychological_context, dict):  # Ensure it's a dictionary
                psychological_context["pain_point"] = pain_point_results.get("pain_point", {})
                psychological_context["pain_point_detected"] = True
            else:
                logger.error("psychological_context is not a dictionary. Skipping updates.")

            pain_point_value = pain_point_results.get("pain_point") if pain_point_results else None

            # Check if pain_point is a dict or a string
            if isinstance(pain_point_value, dict):
                # It's a dict, safe to use .get()
                enhanced_context["recurring_themes"] = pain_point_value.get("recurring_terms", [])
            else:
                # It's a string or None, just use an empty list
                enhanced_context["recurring_themes"] = []

            enhanced_context["approach"] = pain_point_results.get("suggested_approach", {})

        if hot_topics:
            context["hot_topics"] = hot_topics
            enhanced_context["hot_topics"] = hot_topics

        return context

    def generate_response_with_template(
        self,
        user_question: str,
        session_id: str,
        generation_context: Dict,
        pain_point_results: Optional[Dict] = None,
    ) -> str:
        """Generate response with appropriate template with improved template tracking."""
        # Initialize pain_point_results to empty dict if None to avoid type errors
        pain_point_results = pain_point_results or {}

        # Extract topic from context
        topic = generation_context.get("psychological_context", {}).get("topic", DEFAULT_TOPIC)

        # This will store our final template choice for logging and metadata
        selected_template = None
        template_selection_reason = None
        approach_type = None

        if pain_point_results:
            approach_type = pain_point_results.get("approach_type")
            if approach_type:
                # Use this approach_type to select template
                selected_template = map_approach_to_template(approach_type)
                template_selection_reason = f"pain_point_approach:{approach_type}"
                logger.info(f"Selected template '{selected_template}' for direct approach '{approach_type}'")

        # Select template based on pain point or topic
        elif pain_point_results.get("pain_point_detected", False) and "suggested_approach" in pain_point_results:
            # Use approach type from pain point detection
            approach_type = pain_point_results["suggested_approach"].get("approach_type", DEFAULT_APPROACH)
            selected_template = map_approach_to_template(approach_type)
            template_selection_reason = f"pain_point:{approach_type}"
            logger.info(f"Selected template '{selected_template}' for approach '{approach_type}'")
        else:
            # Fall back to topic-based template
            selected_template = map_approach_to_template(topic)
            template_selection_reason = f"topic:{topic}"
            logger.info(f"Using topic as approach for template selection: {topic}")

        if not selected_template:
            selected_template = map_approach_to_template(None)  # Should return "empathy_validation"
            template_selection_reason = "fallback:default"
            logger.info(f"Using default template after all selection paths failed: {selected_template}")

        # Store the template choice in the generation_context for later retrieval
        if "metadata" not in generation_context:
            generation_context["metadata"] = {}

        generation_context["metadata"].update(
            {"template_used": selected_template, "template_selection_reason": template_selection_reason}
        )

        # Use dynamic retriever if available - don't check for use_dynamic_retrieval flag
        if "dynamic_retriever" in generation_context:
            try:
                dynamic_retriever = generation_context["dynamic_retriever"]
                # Check if the method exists to avoid AttributeError
                if hasattr(dynamic_retriever, "get_conversation_context"):
                    conversation_context = dynamic_retriever.get_conversation_context()
                    if conversation_context:
                        generation_context["conversation_context"] = conversation_context
                        logger.debug(f"Added conversation context from dynamic retriever")
            except Exception as e:
                logger.error(f"Error using dynamic retriever: {str(e)}")

        try:
            # Check if the text generator has the dynamic retrieval method
            if hasattr(self.text_generator, "generate_therapeutic_response_with_dynamic_retrieval"):
                # Use the specialized method with dynamic retrieval
                response = self.text_generator.generate_therapeutic_response_with_dynamic_retrieval(
                    user_question=user_question, template_name=selected_template, context=generation_context
                )
                logger.info(f"Generated response with dynamic retrieval template '{selected_template}'")
            else:
                # Fall back to standard method
                response = self.text_generator.generate_therapeutic_response(
                    user_question=user_question,
                    template_name=selected_template,
                    context=generation_context,
                    conversation_history=self.db_manager.get_conversation_history(session_id),
                )
                logger.info(f"Generated response with standard template '{selected_template}'")

            return response
        except Exception as e:
            logger.error(f"Error generating response with template {selected_template}: {str(e)}")
            # Fallback to empathy_validation in case of template error
            try:
                logger.info("Attempting fallback to empathy_validation template")
                return self.text_generator.generate_therapeutic_response(
                    user_question=user_question,
                    template_name=DEFAULT_APPROACH,
                    context=generation_context,
                    conversation_history=self.db_manager.get_conversation_history(session_id),
                )
            except Exception as fallback_e:
                logger.error(f"Fallback template also failed: {str(fallback_e)}")
                return "I'm here to support you. Could you share a bit more about what's on your mind?"

    @typechecked
    def determine_final_context(
        self, user_question: str, topics_context: Dict, pain_point_results: Dict, metadata: Dict
    ) -> Tuple[str, Dict]:
        """Determine the final context for response generation using systematic source checking."""
        try:
            detector = SemanticEmotionDetector()

            # 1. Initialize variables and copy metadata once
            topics_context = topics_context or {}
            pain_point_results = pain_point_results or {}
            updated_metadata = metadata.copy() if metadata else {}

            # 2. Extract ALL potential context sources systematically
            context_sources = {
                "from_topic": topics_context.get("topic", ""),
                "from_extracted_topics": topics_context.get("extracted_topics", []),
                "from_pain_point": pain_point_results.get("pain_point", {}).get("topic", ""),
                "from_approach": pain_point_results.get("suggested_approach", {}).get("approach_type", ""),
            }

            # 3. Standardize ALL sources using existing methods
            standardized_sources = {
                "topic": detector.get_standardized_topic(context_sources["from_topic"]),
                "extracted_topics": [
                    detector.get_standardized_topic(t) for t in context_sources["from_extracted_topics"]
                ],
                "pain_point": detector.get_standardized_topic(context_sources["from_pain_point"]),
                "approach": detector.get_standardized_approach(context_sources["from_approach"]),
            }

            # 4. Determine context with clear priority rules
            context = "therapeutic_dialogue"  # Default fallback

            extracted_topics_list = standardized_sources["extracted_topics"]  # type: ignore
            if extracted_topics_list and len(extracted_topics_list) > 0:
                # Extract a single topic to avoid type errors
                context = extracted_topics_list[0]
            elif isinstance(standardized_sources["topic"], str) and standardized_sources["topic"] not in [
                "supportive_listening",
                "general_support",
                "therapeutic_dialogue",
            ]:
                context = standardized_sources["topic"]
            elif isinstance(standardized_sources["pain_point"], str) and standardized_sources["pain_point"] not in [
                "supportive_listening",
                "general_support",
                "therapeutic_dialogue",
            ]:
                context = standardized_sources["pain_point"]
            elif (
                isinstance(standardized_sources["approach"], str)
                and standardized_sources["approach"].strip()  # Ensure approach is non-empty
                and standardized_sources["approach"]
                not in [
                    "supportive_listening",
                    "general_support",
                    "therapeutic_dialogue",
                ]
            ):
                # Accept any non-empty approach, including 'stress_management'
                context = standardized_sources["approach"]

            # 5. Update metadata with all standardized information
            is_pain_point_detected = pain_point_results.get("pain_point_detected", False) or (
                pain_point_results.get("pain_point", {}).get("detected", False)
            )

            approach_type = (
                pain_point_results.get("approach_type")
                or pain_point_results.get("suggested_approach", {}).get("approach_type")
                or standardized_sources["approach"]
                or "supportive_listening"
            )

            template_used = pain_point_results.get("template_used")
            if not template_used and approach_type:
                template_used = map_approach_to_template(approach_type)
            if not template_used:
                template_used = "default_template"

            logger.info(f"FINAL: approach_type={approach_type}, template_used={template_used}")

            updated_metadata.update(
                {
                    "pain_point_detected": is_pain_point_detected,
                    "therapeutic_approach": approach_type,
                    "standardized_approach": approach_type,
                    "context": context,
                    "standardized_topic": standardized_sources["topic"],
                    "template_used": template_used,
                }
            )

            logger.info(f"DEBUG: updated_metadata={updated_metadata}")
            # Add extracted topics to metadata
            if context_sources["from_extracted_topics"]:
                updated_metadata["extracted_topics"] = context_sources["from_extracted_topics"]
                updated_metadata["primary_extracted_topic"] = context_sources["from_extracted_topics"][0]

            logger.info(f"Determined context: {context}")
            return context, updated_metadata

        except Exception as e:
            logger.error(f"Error determining context: {str(e)}")
            return "therapeutic_dialogue", metadata or {}

    def save_interaction(
        self, context: str, user_question: str, response: str, metadata: Dict, session_id: str
    ) -> None:
        """Save the interaction to the database with error handling."""
        # Use the determined context when saving
        logger.info("About to save_interaction with context: %s, session_id: %s", context, session_id)
        try:
            self.db_manager.save_interaction(
                context=context,
                question=user_question,
                answer=response if response else "No response generated",
                metadata=metadata,
                session_id=session_id,
            )
            logger.info("save_interaction completed successfully")
        except Exception as e:
            logger.error("Error in save_interaction: %s", e)
            logger.error(traceback.format_exc())

    def debug_context_determination(
        self, user_question: str, detected_topic: str, extracted_topics: List[str], approach_type: str, pain_point: Dict
    ) -> None:
        """Debug function to log context determination process."""
        logger.debug("DEBUG CONTEXT DETERMINATION:")
        logger.debug("- User question: %s...", user_question[:50])
        logger.debug("- Detected topic: %s", detected_topic)
        logger.debug("- Extracted topics: %s", extracted_topics)
        logger.debug("- Approach type: %s", approach_type)
        logger.debug("- Pain point: %s...", str(pain_point)[:100])
