import re
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from prismalog.log import get_logger

from psy_supabase.config import DEFAULT_EMOTION, DEFAULT_TOPIC
from psy_supabase.utilities.semantic_emotion_detector import SemanticEmotionDetector
from psy_supabase.utilities.therapeutic_mappings import TherapeuticMappings
from psy_supabase.utilities.utils import get_spacy_model

if TYPE_CHECKING:
    from psy_supabase.core.text_generator import TextGenerator

logger = get_logger(__name__)


class PromptSelector:
    """Selects the most appropriate therapeutic prompt template based on user input."""

    def __init__(self, generator: "TextGenerator"):
        """Initialize the prompt selector with the text generator."""
        self.generator = generator
        self.nlp = get_spacy_model()
        if self.nlp is None:
            logger.error("Failed to load spaCy model. Some functionality may be limited.")

        # Add semantic helper - loads lazily only when needed
        self.semantic_helper = SemanticEmotionDetector()

    def analyze_question(self, question_text: str, min_confidence: float = 0.5) -> Dict[str, Any]:
        try:
            # Initialize default analysis results
            analysis: Dict[str, Any] = {
                "topic": DEFAULT_TOPIC,
                "confidence": 0.5,
                "topic_confidence": 0.5,
                "emotion": None,
                "emotion_intensity": 0.0,
                "emotion_confidence": 0.0,
            }

            # Skip analysis for empty questions
            if not question_text or len(question_text.strip()) < 3:
                return analysis

            # Check for crisis keywords first (safety priority)
            crisis_keywords = [
                "suicide",
                "kill myself",
                "want to die",
                "end my life",
                "don't want to live",
                "do not want to live",
                "suicidal",
                "harm myself",
            ]
            if any(keyword in question_text.lower() for keyword in crisis_keywords):
                analysis["topic"] = "crisis"
                analysis["confidence"] = 0.95
                analysis["topic_confidence"] = 0.95
                analysis["emotion"] = "distress"
                analysis["emotion_intensity"] = 0.9
                analysis["emotion_confidence"] = 0.9
                return analysis

            # Get topic from regex patterns
            topic_scores: Dict[str, float] = {}
            for topic, patterns in TherapeuticMappings.get_all_patterns().items():
                topic_score: float = 0.0
                pattern_matches: List[Tuple[str, float]] = []

                for pattern, weight in patterns:
                    try:
                        if re.search(pattern, question_text, re.IGNORECASE):
                            pattern_matches.append((pattern, weight))
                            topic_score += weight
                    except re.error as regex_err:
                        logger.error(f"Invalid regex pattern '{pattern}' for topic '{topic}': {regex_err}")
                        # Continue processing other patterns
                        continue

                if pattern_matches:
                    # Calculate weighted confidence score
                    normalized_score = min(0.95, topic_score / (len(pattern_matches) * 2.0))
                    topic_scores[topic] = normalized_score

            # Select topic with highest regex score
            if topic_scores:
                max_topic = max(topic_scores.items(), key=lambda x: x[1])
                analysis["topic"] = max_topic[0]
                analysis["confidence"] = max_topic[1]
                analysis["topic_confidence"] = max_topic[1]
                regex_confidence = max_topic[1]
            else:
                regex_confidence = 0.0

            # Process semantic topic results
            semantic_topic, topic_confidence = self.semantic_helper.detect_topic_standardized(question_text)
            logger.info(f"Semantic topic detected: {semantic_topic} ({topic_confidence:.2f})")

            # Map "greeting" topic to "supportive_listening" for consistency with tests
            if semantic_topic == "greeting":
                semantic_topic = "supportive_listening"
                logger.debug(f"Mapped 'greeting' topic to 'supportive_listening' for consistency")

            # Use semantic topic if regex confidence is low or not detected
            if semantic_topic and topic_confidence > 0.7 and (regex_confidence < 0.7 or not topic_scores):
                analysis["topic"] = semantic_topic
                analysis["confidence"] = topic_confidence
                analysis["topic_confidence"] = topic_confidence
                logger.debug(f"Using semantic topic: {semantic_topic} ({topic_confidence:.2f})")

            # PRIORITY HANDLING: Define priority topics that should be preserved
            # even with lower confidence scores
            priority_topics = ["crisis", "anxiety", "depression", "trauma", "grief_loss"]

            # Apply confidence threshold with priority consideration
            if min_confidence is not None and analysis["topic_confidence"] < min_confidence:
                # Only default to supportive_listening if the current topic is NOT a priority topic
                if analysis["topic"] not in priority_topics:
                    logger.debug(
                        f"Low confidence ({analysis['topic_confidence']:.2f}) for non-priority topic '{analysis['topic']}', defaulting to supportive_listening"
                    )
                    analysis["topic"] = "supportive_listening"
                    analysis["confidence"] = 0.4
                    analysis["topic_confidence"] = 0.4
                else:
                    logger.debug(
                        f"Keeping priority topic '{analysis['topic']}' despite low confidence ({analysis['topic_confidence']:.2f})"
                    )
                    # Optionally boost confidence slightly for priority topics to ensure they're used
                    analysis["confidence"] = max(analysis["confidence"], min_confidence * 0.8)
                    analysis["topic_confidence"] = max(analysis["topic_confidence"], min_confidence * 0.8)

            # HYBRID APPROACH - STEP 3: Now detect emotion with regex first
            emotion_scores: Dict[str, float] = {}
            for emotion, patterns in TherapeuticMappings.get_emotion_patterns().items():
                emotion_score: float = 0.0

                for pattern, weight in patterns:
                    try:
                        if re.search(pattern, question_text, re.IGNORECASE):
                            pattern_matches.append((pattern, weight))
                            emotion_score += weight
                    except re.error as regex_err:
                        logger.error(f"Invalid regex pattern '{pattern}' for emotion '{emotion}': {regex_err}")
                        continue  # Skip if it's still broken after repair attempt

                if pattern_matches:
                    # Calculate weighted confidence score
                    normalized_score = min(0.95, emotion_score / (len(pattern_matches) * 2.0))
                    emotion_scores[emotion] = normalized_score

            # Select emotion with highest regex score
            if emotion_scores:
                max_emotion = max(emotion_scores.items(), key=lambda x: x[1])
                analysis["emotion"] = max_emotion[0]
                analysis["emotion_intensity"] = max_emotion[1]
                analysis["emotion_confidence"] = max_emotion[1]
                regex_emotion_confidence = max_emotion[1]
            else:
                regex_emotion_confidence = 0.0

            # HYBRID APPROACH - STEP 4: Get semantic emotion detection
            semantic_emotion, semantic_confidence = self.semantic_helper.detect_emotion_standardized(question_text)
            logger.info(f"Semantic emotion detected: {semantic_emotion} ({semantic_confidence:.2f})")

            # Map "greeting" emotion to "concern" for consistency
            if semantic_emotion == "greeting":
                semantic_emotion = "concern"
                logger.debug(f"Mapped 'greeting' emotion to 'concern' for consistency")

            # Use semantic emotion if regex confidence is low or not detected
            if (
                semantic_emotion
                and semantic_confidence > 0.7
                and (regex_emotion_confidence < 0.6 or not emotion_scores)
            ):
                analysis["emotion"] = semantic_emotion
                analysis["emotion_intensity"] = semantic_confidence
                analysis["emotion_confidence"] = semantic_confidence
                logger.debug(f"Using semantic emotion: {semantic_emotion} ({semantic_confidence:.2f})")

            # 3. Strategic combination of regex and semantic results for topic
            if semantic_topic and topic_confidence > 0.75:
                # Case 1: No regex topic found - use semantic
                if not topic_scores:
                    analysis["topic"] = semantic_topic
                    analysis["confidence"] = topic_confidence
                    analysis["topic_confidence"] = topic_confidence
                    logger.info(f"Using semantic topic (no regex match): {semantic_topic} ({topic_confidence:.2f})")
                # Case 2: If semantic topic isn't "supportive_listening" (default) but regex found default, prefer semantic
                elif semantic_topic != "supportive_listening" and analysis["topic"] == "supportive_listening":
                    analysis["topic"] = semantic_topic
                    analysis["confidence"] = topic_confidence
                    analysis["topic_confidence"] = topic_confidence
                    logger.info(f"Preferring non-default topic '{semantic_topic}' over default 'supportive_listening'")
                # Case 3: If semantic has higher confidence and isn't the default
                elif topic_confidence > regex_confidence + 0.15 and semantic_topic != "supportive_listening":
                    analysis["topic"] = semantic_topic
                    analysis["confidence"] = topic_confidence
                    analysis["topic_confidence"] = topic_confidence
                    logger.info(
                        f"Using semantic topic (stronger than regex): {semantic_topic} ({topic_confidence:.2f})"
                    )
                # Otherwise, keep regex result

            # Ensure emotion is never None
            if analysis["emotion"] is None:
                analysis["emotion"] = "concern"
                analysis["emotion_intensity"] = 0.5
                analysis["emotion_confidence"] = 0.5

            # Log the analysis results
            logger.info(
                f"Question analysed - Topic: {analysis['topic']} ({analysis['confidence']:.2f}), "
                f"Emotion: {analysis['emotion']} ({analysis['emotion_intensity']:.2f})"
            )

            return analysis

        except Exception as e:
            logger.error(f"Error in question analysis: {e}")
            return {
                "topic": "supportive_listening",
                "confidence": 0.5,
                "topic_confidence": 0.5,
                "emotion": "concern",
                "emotion_intensity": 0.5,
                "emotion_confidence": 0.5,
            }

    def generate_category_info(self, question: str) -> Dict[str, Any]:
        """
        Generate category information for a question based on the detected topic.

        Args:
            question: The user question or statement.

        Returns:
            A dictionary with category information.
        """
        # Use our existing topic detection
        analysis = self.analyze_question(question)
        topic = analysis.get("topic", "supportive_listening")

        # Map the topic to category info
        category_info = {}

        # Get description from therapeutic themes if available
        theme_data = TherapeuticMappings.THERAPEUTIC_THEMES.get(topic, {})
        description = theme_data.get("human_readable") or theme_data.get("template") or "therapeutic_support"

        # Add techniques based on the topic
        techniques = theme_data.get("techniques", ["active listening", "validation", "reflection"])

        # Build category info
        category_info[topic] = {"description": description, "techniques": techniques}

        # If no specific topic was found, provide a default
        if not category_info or topic == "supportive_listening":
            category_info["supportive_listening"] = {
                "description": "General therapeutic support and active listening",
                "techniques": ["active listening", "validation", "reflection"],
            }

        return category_info

    def determine_topic(self, category_info: Dict[str, Any], user_question: str) -> str:
        """
        Determine the primary topic from category info or fall back to analyzing user question.

        This method attempts to extract a topic from the category_info dictionary first.
        If unsuccessful, it falls back to analyzing the user_question directly.

        Args:
            category_info: Dictionary containing category information
            user_question: The original user question to analyze if category_info doesn't yield a topic

        Returns:
            The detected topic as a string.
        """
        # First try to get the primary topic from category_info keys
        if category_info and isinstance(category_info, dict):
            # Return the first category key as the topic
            return next(iter(category_info.keys()), "supportive_listening")

        # If no topic was found in category_info, analyze the user question
        analysis = self.analyze_question(user_question)
        return analysis.get("topic", "supportive_listening")
