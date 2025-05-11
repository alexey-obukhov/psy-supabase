"""
test_pain_point_detection.py

This module contains tests for evaluating the pain point detection capabilities of the RAG (Retrieval-Augmented Generation) system.
It simulates user conversations with recurring themes and verifies the system's ability to detect psychological pain points,
identify recurring themes, and recommend appropriate therapeutic approaches.

Key Features:
- **Simulated Conversations**:
  - Tests various conversation scenarios with predefined questions and expected pain points.
  - Covers themes such as workplace trauma, relationship insecurity, family dynamics, health anxiety, and self-worth struggles.

- **Pain Point Detection**:
  - Verifies the system's ability to detect recurring psychological pain points in user interactions.
  - Ensures the system identifies relevant therapeutic approaches and recurring themes.

- **Vector-Based Topic Analysis**:
  - Uses pgvector clustering to analyse conversation topics and compare them with expected themes.
  - Evaluates the match between detected and expected themes for accuracy.

- **Test Environment Setup**:
  - Initializes a test schema in the database for isolated testing.
  - Ensures vector indexes and knowledge base initialization for proper functionality.

Classes:
- `PainPointDetectionTester`: A class that sets up the test environment, simulates conversations, and evaluates pain point detection.

Functions:
- `analyze_test_results(results)`: Analyzes and summarizes the results of the pain point detection tests.
- `main()`: Entry point for running the pain point detection tests.

Dependencies:
- `psy_supabase.core.database.DatabaseManager`: Manages database operations for storing and retrieving interactions.
- `psy_supabase.core.rag_processor.RAGProcessor`: Handles response generation and pain point detection.
- `psy_supabase.core.text_generator.TextGenerator`: Generates therapeutic responses for user queries.
- `prismalog.log.ColoredLogger`: Provides enhanced logging for debugging and monitoring.

Usage:
    # Run the pain point detection tests
    python test_pain_point_detection.py

    # Example output:
    # === PAIN POINT DETECTION TEST RESULTS ===
    # Conversations tested: 5
    # Total exchanges: 20
    # Total pain points detected: #
    # Overall detection rate: #
"""

import json
import os
import sys
import uuid
from collections import Counter
from logging import Logger
from typing import Any, Dict, List, Optional, Set, Union

from prismalog.log import get_logger
from typeguard import typechecked

# import multiprocessing as mp
from psy_supabase.utilities.common import is_github_actions
from psy_supabase.utilities.utils import cleanup_memory

# Set up logging
logger: Logger = get_logger(__name__)

# Conditionally import dotenv
if not is_github_actions():
    from dotenv import load_dotenv

    load_dotenv()  # Load environment variables from .env file
    logger.info("Local development: Loading environment from .env file")
else:
    logger.info("CI environment: Using GitHub secrets")

device: str = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Using device: %s", device)

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Require Supabase credentials from environment
supabase_url: Optional[str] = os.environ.get("SUPABASE_URL")
supabase_key: Optional[str] = os.environ.get("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    logger.critical("SUPABASE_URL and SUPABASE_KEY must be set in environment or .env file")
    sys.exit(1)

from psy_supabase.core.database import DatabaseManager
from psy_supabase.core.rag_processor import RAGProcessor
from psy_supabase.core.text_generator import TextGenerator

# Test conversation scenarios with recurring themes
TEST_CONVERSATIONS: List[Dict[str, Any]] = [
    {
        "name": "workplace_trauma Pattern",
        "questions": [
            "I had a really difficult day at work today. My boss criticized me in front of everyone again.",
            "Why do I always feel so nervous before team meetings? I'm prepared but still worry about being called out.",
            "Should I start looking for another job? I'm tired of feeling inadequate every day in this place.",
            "How do I stop obsessing over every email my supervisor sends? I keep looking for hidden criticisms.",
        ],
        "expected_pain_point": {
            "themes": ["workplace", "criticism", "anxiety", "humiliation", "inadequacy"],
            "approach_types": ["subtle", "gentle", "direct", "cognitive_behavioral"],
            "expected_topics": ["workplace_stress", "anxiety", "self-esteem"],
            "expected_emotions": ["anxiety", "concern", "sadness"],
        },
    },
    {
        "name": "Relationship Insecurity Pattern",
        "questions": [
            "My partner was texting someone and smiling, but wouldn't tell me who it was.",
            "Is it normal to check your partner's phone when they're sleeping?",
            "I can't stop thinking about who my partner might be talking to when we're apart.",
            "Sometimes I make up excuses to call my partner just to check where they are.",
        ],
        "expected_pain_point": {
            "themes": ["jealousy", "insecurity", "trust", "relationship", "anxiety"],
            "approach_types": ["gentle", "direct", "compassionate", "psychodynamic"],
            "expected_topics": ["relationship_trust", "jealousy", "anxiety"],
            "expected_emotions": ["jealousy", "insecurity", "anxiety"],
        },
    },
    {
        "name": "family_dynamics Conflict",
        "questions": [
            "My mother always favors my sister over me, no matter what I achieve.",
            "I dread family gatherings because I always feel like an outsider.",
            "Why do I still seek my parents' approval even though I'm in my 30s?",
            "I find myself acting like a teenager again whenever I visit my childhood home.",
        ],
        "expected_pain_point": {
            "themes": ["family", "rejection", "childhood", "approval", "favoritism"],
            "approach_types": ["psychodynamic", "insight_oriented", "compassionate"],
            "expected_topics": ["family_conflict", "childhood_issues", "approval_seeking"],
            "expected_emotions": ["rejection", "sadness", "longing"],
        },
    },
    {
        "name": "health_anxiety",
        "questions": [
            "I found a small lump on my neck and I'm convinced it's cancer.",
            "The doctor said my tests were normal but I still feel something is wrong with my body.",
            "I spend hours researching symptoms online and always find something that matches.",
            "How can I stop checking my pulse and blood pressure multiple times a day?",
        ],
        "expected_pain_point": {
            "themes": ["health", "anxiety", "catastrophizing", "obsession", "control"],
            "approach_types": ["cognitive_behavioral", "mindfulness", "practical"],
            "expected_topics": ["health_anxiety", "obsession", "control"],
            "expected_emotions": ["fear", "anxiety", "obsession"],
        },
    },
    {
        "name": "Self-Worth Struggle",
        "questions": [
            "Sometimes I feel like I'm just taking up space in this world.",
            "Why do I always apologise for things that aren't my fault?",
            "I turned down a promotion because I don't think I'm good enough for it.",
            "I can't accept compliments without explaining why the person is actually wrong about me.",
        ],
        "expected_pain_point": {
            "themes": ["self-esteem", "worthlessness", "impostor syndrome", "shame"],
            "approach_types": ["compassionate", "humanistic", "schema-focused", "gentle"],
            "expected_topics": ["self-worth", "impostor_syndrome", "shame"],
            "expected_emotions": ["worthlessness", "shame", "self_doubt"],
        },
    },
]


class PainPointDetectionTester:
    """Tests the pain point detection capabilities of the RAG system."""

    test_user_id: str
    test_session_id: str
    db_manager: DatabaseManager
    generator: TextGenerator
    rag_processor: RAGProcessor

    def __init__(self) -> None:
        """Initialize test environment and components."""
        # Create a unique test user ID for this test run
        self.test_user_id = f"test_user_{uuid.uuid4().hex[:8]}"
        logger.info("Using test user ID: %s", self.test_user_id)

        # Create a unique schema name for testing based on the user ID
        self.test_session_id = f"test_pain_point_{uuid.uuid4().hex[:10]}"
        logger.info("Using test session ID: %s", self.test_session_id)

        # Add assertions to satisfy mypy
        assert supabase_url is not None, "Supabase URL must be set"
        assert supabase_key is not None, "Supabase Key must be set"

        # Initialize database manager with Supabase credentials and test user
        self.db_manager = DatabaseManager(
            supabase_url=supabase_url, supabase_key=supabase_key, user_id=self.test_user_id
        )

        self.generator = TextGenerator(model_name="rasyosef/Phi-1_5-Instruct-v0.1", device=device)

        # Initialize RAG processor with the test schema
        self.rag_processor = RAGProcessor(
            db_manager=self.db_manager, generator=self.generator, intelligent_processing_enabled=True
        )

        # Ensure we can access the database
        self.db_manager.schema_name = self.db_manager.schema_name

        # Set up test environment
        self._setup_test_environment()

    def _setup_test_environment(self) -> None:
        """Set up the test environment, creating necessary tables."""
        logger.info("Setting up test environment...")

        # Initialize the database schema for testing
        self.db_manager.create_user_schema_sync()

        logger.info("Test environment setup complete")

    @typechecked
    # Update in the _simulate_conversation method to extract data differently
    def _simulate_conversation(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate a conversation and track pain point detection.

        Args:
            conversation: Dictionary with conversation name and questions

        Returns:
            Dictionary with conversation results
        """
        logger.info("Testing conversation: %s", conversation["name"])

        results: Dict[str, Any] = {
            "name": conversation["name"],
            "exchanges": [],
            "pain_points_detected": 0,
            "templates_used": [],
            "first_detection_at": None,
        }

        # Process each question in sequence
        for i, question in enumerate(conversation["questions"]):
            logger.info("Question %d: %s...", i + 1, question[:50])

            # Generate response through the RAG processor
            response: str = self.rag_processor.generate_response(
                user_question=question, session_id=self.test_session_id, device=device, question_id=i
            )

            # Get the most recent interaction's metadata
            try:
                history: List[Dict[str, Any]] = self.db_manager.get_conversation_history(self.test_session_id)

                # Get just the latest message (the one we just added)
                latest_interaction: Dict[str, Any] = history[-1] if history else {}

                # Convert metadata from string to dict if needed
                metadata, _ = latest_interaction.get("metadata", [])
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except Exception as e:
                        logger.error("Error parsing metadata JSON: %s", e)
                        metadata = {}

                pain_point_detected: bool = False
                if metadata.get("pain_points") and isinstance(metadata.get("pain_points"), list):
                    # Check the most recent pain point entry
                    pain_points = metadata.get("pain_points")
                    if pain_points and pain_points[-1].get("detected"):
                        pain_point_detected = True

                # If pain_point_detected is False, try to determine it from other indicators
                if not pain_point_detected:
                    # Check if pain_point exists in metadata
                    if "pain_point" in metadata and metadata["pain_point"]:
                        pain_point_detected = True

                    # Check if approach_type exists, which usually indicates pain point detection
                    elif "approach_type" in metadata and metadata["approach_type"]:
                        pain_point_detected = True

                    # Check if template_used is not the default generic template
                    elif metadata.get("template_used") and metadata.get("template_used") != "default":
                        pain_point_detected = True

                # Log the exact state of pain point detection for debugging
                logger.info(
                    f"Pain point detection indicators: detected={pain_point_detected}, pain_point_key_exists={'pain_point' in metadata}"
                )

                # Handle topic and emotion from the new analyzer
                # Topics now come from analyze_question method
                topics_context = metadata.get("topics_context", {})
                detected_topic = topics_context.get("topic", "empathy_validation")
                extracted_topics = topics_context.get("extracted_topics", []) or []
                emotion = topics_context.get("emotion", "concern")

                # Get template used
                template_used: str = metadata.get("template_used", "dynamic_rag_therapy")

                # Get similarity score with fallback
                similarity: float = metadata.get("pain_point_similarity") or metadata.get("similarity") or 0.0

                # Get recurring themes based on the new structure
                recurring_themes: List[str] = extracted_topics
                if not recurring_themes and "pain_point" in metadata:
                    pain_point = metadata.get("pain_point", {})
                    if isinstance(pain_point, dict) and "keywords" in pain_point:
                        recurring_themes = pain_point["keywords"][:3]
                    elif isinstance(pain_point, dict) and "name" in pain_point:
                        recurring_themes = [pain_point["name"]]

                # Get approach type with multiple fallbacks
                approach_type: Optional[Union[str, List[str]]] = None
                if "approach_type" in metadata:
                    approach_type = metadata.get("approach_type")
                elif "therapeutic_approach" in metadata:
                    approach_type = metadata.get("therapeutic_approach")
                elif "suggested_approach" in metadata and isinstance(metadata["suggested_approach"], dict):
                    approach_type = metadata["suggested_approach"].get("approach_type")
                else:
                    # Extract from template if possible
                    template = metadata.get("template_used", "")
                    if "_" in template:
                        approach_type = template.split("_")[-1]

                # Record results
                exchange_result: Dict[str, Any] = {
                    "question": question,
                    "response": response if response else "No response",
                    "pain_points_detected": pain_point_detected,
                    "therapeutic_approach": approach_type or "empathy_validation",
                    "approach_type": approach_type,
                    "template_used": template_used,
                    "similarity": similarity,
                    "recurring_themes": recurring_themes,
                    "topic": detected_topic,
                    "emotion": emotion,
                }

                results["exchanges"].append(exchange_result)

                # Update summary statistics
                if pain_point_detected:
                    results["pain_points_detected"] += 1
                    if not results["first_detection_at"]:
                        results["first_detection_at"] = i + 1

                results["templates_used"].append(template_used)

                logger.info(
                    f"Response generated. Topic: {detected_topic}, Emotion: {emotion}, "
                    f"Pain point detected: {pain_point_detected}, "
                    f"Approach: {approach_type or 'unknown'}, Template: {template_used}"
                )

            except Exception as e:
                logger.error("Error processing results: %s", e)
                # Add fallback for error handling...

                # Simple fallback with just the question and response
                results["exchanges"].append(
                    {
                        "question": question,
                        "response": response if response else "No response",
                        "error": str(e),
                        "pain_points_detected": False,
                        "therapeutic_approach": "error",
                        "template_used": "error",
                        "similarity": 0.0,
                        "recurring_themes": [],
                        "topic": "error",
                        "emotion": "error",
                    }
                )
        return results

    def run_tests(self) -> List[Dict[str, Any]]:
        """
        Run all test conversations and collect results.

        Returns:
            List of result dictionaries for each conversation
        """
        all_results: List[Dict[str, Any]] = []

        for conversation in TEST_CONVERSATIONS:
            # Add a small delay between conversations
            logger.info("Starting test for: %s", conversation["name"])

            # Simulate the conversation
            result: Dict[str, Any] = self._simulate_conversation(conversation)
            all_results.append(result)

            # Run topic analysis after the conversation is complete
            logger.info("Performing vector-based topic analysis...")
            result["vector_topics"] = self.analyze_conversation_topics()

            # Log summary of this conversation test
            logger.info("Completed test for: %s", conversation["name"])
            logger.info(
                "  Pain points detected: %s out of %d", result["pain_points_detected"], len(conversation["questions"])
            )
            if result["first_detection_at"]:
                logger.info("  First detected at question #%s", result["first_detection_at"])
            logger.info("  Templates used: %s", ", ".join(result["templates_used"]))

            # Log topic analysis results
            if result["vector_topics"]:
                topic_str = ", ".join(
                    [
                        f"{t['topic']}({t['frequency']})"
                        for t in result["vector_topics"]
                        if not t["topic"].startswith("Error") and not t["topic"].startswith("No ")
                    ]
                )
                if topic_str:
                    logger.info("  Vector topics identified: %s", topic_str)

            logger.info("----------------------------------------")

        return all_results

    def cleanup(self) -> None:
        """Clean up test environment to avoid cluttering the database."""
        logger.info("Cleaning up test environment...")
        cleanup_memory()

    def analyze_conversation_topics(self) -> List[Dict[str, Any]]:
        """
        Analyze topics in the test conversation using pgvector clustering.

        Returns:
            List of topic dictionaries with topic name and frequency
        """
        try:
            # Call the pgvector-powered topic analysis function via RPC
            response = self.db_manager.supabase.rpc(
                "analyze_conversation_topics",
                {"p_schema_name": self.db_manager.schema_name, "p_session_id": self.test_session_id, "p_min_count": 1},
            ).execute()

            topics = []
            if response.data:
                for item in response.data:
                    topics.append({"topic": item.get("topic", "Unknown topic"), "frequency": item.get("frequency", 0)})
                logger.info("Topic analysis found %d topics", len(topics))
                return topics
            else:
                logger.info("No significant topics identified")
                return [{"topic": "No significant topics identified", "frequency": 0}]

        except Exception as e:
            logger.error("Error analysing topics: %s", e)
            return [{"topic": f"Error: {str(e)}", "frequency": 0}]


def analyze_test_results(results: List[Dict[str, Any]]) -> None:
    """
    Analyze and print summary of test results.

    Args:
        results: List of test results from run_tests()
    """
    logger.info("=== PAIN POINT DETECTION TEST RESULTS ===")
    logger.info("Conversations tested: %d", len(results))

    # Overall statistics
    total_exchanges: int = sum(len(r["exchanges"]) for r in results)
    total_detected: int = sum(r["pain_points_detected"] for r in results)
    detection_rate: float = (total_detected / total_exchanges) * 100 if total_exchanges > 0 else 0

    logger.info("Total exchanges: %d", total_exchanges)
    logger.info("Total pain points detected: %d", total_detected)
    logger.info(f"Overall detection rate: {detection_rate:.2f}%")

    # Analyze each conversation
    logger.info("\nDetailed results by conversation:")
    for i, result in enumerate(results):
        logger.info("\n%s:", result["name"])
        logger.info(f"  Detection rate: {(result['pain_points_detected'] / len(result['exchanges'])) * 100:.2f}%")
        logger.info(
            "  First detected at: Question #%s", result["first_detection_at"] if result["first_detection_at"] else "N/A"
        )

        # Process data using only fields that exist in the actual results

        # 1. Analyze detected topics - use "detected_topic" from exchange results
        all_topics: List[str] = []
        for exchange in result["exchanges"]:
            topic = exchange.get("topic", "supportive_listening")
            if topic and topic not in all_topics and topic != "error":
                all_topics.append(topic)

        if all_topics:
            topic_counts: Counter = Counter(all_topics)
            logger.info(f"  Detected topics: {', '.join([f'{t}({c})' for t, c in topic_counts.most_common()])}")

        # 2. Analyze detected emotions - use "emotion" from exchange results
        all_emotions: List[str] = []
        for exchange in result["exchanges"]:
            emotion = exchange.get("emotion")
            if emotion and emotion != "unknown" and emotion != "error":
                all_emotions.append(emotion)

        if all_emotions:
            emotion_counts: Counter = Counter(all_emotions)
            logger.info(f"  Detected emotions: {', '.join([f'{e}({c})' for e, c in emotion_counts.most_common()])}")

        # 3. Analyze approach types - use "approach_type" or "therapeutic_approach" with consistent naming
        flat_all_approach_types: List[str] = []
        raw_approach_types_from_exchanges: List[Any] = []

        for exchange in result["exchanges"]:
            approach = exchange.get("approach_type") or exchange.get("therapeutic_approach")
            if approach and approach != "unknown" and approach != "error":
                raw_approach_types_from_exchanges.append(approach)

        for item in raw_approach_types_from_exchanges:
            if isinstance(item, str):
                flat_all_approach_types.append(item)
            elif isinstance(item, list):
                for sub_item in item:
                    if isinstance(sub_item, str):
                        flat_all_approach_types.append(sub_item)

        if flat_all_approach_types:
            approach_type_counts: Counter = Counter(flat_all_approach_types)
            logger.info(f"  Approach types: {', '.join([f'{t}({c})' for t, c in approach_type_counts.most_common()])}")

            expected_conversation: Dict[str, Any] = TEST_CONVERSATIONS[i]
            if (
                "expected_pain_point" in expected_conversation
                and "approach_types" in expected_conversation["expected_pain_point"]
            ):
                expected_types: Set[str] = set(expected_conversation["expected_pain_point"]["approach_types"])
                detected_types_for_comparison: Set[str] = set(flat_all_approach_types)

                normalized_expected: Set[str] = {t.lower() for t in expected_types}
                normalized_detected: Set[str] = {(t.lower() if t else "") for t in detected_types_for_comparison}

                matches = set()
                for exp in normalized_expected:
                    for det in normalized_detected:
                        if exp in det or det in exp:
                            matches.add(exp)
                            break

                misses = normalized_expected - matches

                match_percentage: float = (len(matches) / len(normalized_expected)) * 100 if normalized_expected else 0
                logger.info(f"  Approach type match: {match_percentage:.2f}%")
                if matches:
                    logger.info("    Matched: %s", ", ".join(sorted(list(matches))))
                if misses:
                    logger.info("    Missed: %s", ", ".join(sorted(list(misses))))

        # 4. Analyze recurring themes using "recurring_themes" from exchange results
        all_themes: List[str] = []
        for exchange in result["exchanges"]:
            themes = exchange.get("recurring_themes", [])
            if themes and isinstance(themes, list):
                all_themes.extend(themes)

        if all_themes:
            theme_counts: Counter = Counter(all_themes)
            logger.info(f"  Recurring themes: {', '.join([f'{t}({c})' for t, c in theme_counts.most_common(3)])}")

        # 5. Check templates used
        template_counts: Counter = Counter([t for t in result["templates_used"] if t != "error"])
        if template_counts:
            logger.info(f"  Templates: {', '.join([f'{t}({c})' for t, c in template_counts.most_common()])}")


def main() -> None:
    """Run the pain point detection tests."""
    logger.info("Starting pain point detection tests")

    # Verify Supabase credentials before proceeding
    if not supabase_url or not supabase_key:
        logger.critical("SUPABASE_URL and SUPABASE_KEY environment variables must be set")
        return

    try:
        tester: PainPointDetectionTester = PainPointDetectionTester()
        results: List[Dict[str, Any]] = tester.run_tests()
        analyze_test_results(results)
        tester.cleanup()
    except Exception as e:
        logger.error("Error running tests: %s", e)
        import traceback

        logger.error(traceback.format_exc())


if __name__ == "__main__":
    logger = get_logger(__name__)
    main()
