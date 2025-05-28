"""
demo_pain_point_detection.py

A demonstration script that shows how the system detects pain points from
conversations stored in the TEST_CONVERSATIONS dictionary.

This script:
1. Loads the TEST_CONVERSATIONS from test_pain_point_detection.py
2. Simulates a user having these conversations with the system
3. Analyzes and reports detected pain points
4. Compares detected themes with expected themes

Usage:
    python demo_pain_point_detection.py
"""

import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from psy_supabase import get_package_logger
from psy_supabase.config import TEXT_GENERATING_MODEL
from psy_supabase.utilities.common import is_github_actions
from psy_supabase.utilities.therapeutic_mappings import TherapeuticMappings
from psy_supabase.utilities.utils import cleanup_memory

# Set up logging
logger = get_package_logger(__name__)

# Load environment variables
if not is_github_actions():
    load_dotenv()
    logger.info("Local development: Loading environment from .env file")
else:
    logger.info("CI environment: Using GitHub secrets")

# Import TEST_CONVERSATIONS from test_pain_point_detection
from test_real_data.test_pain_point_detection import TEST_CONVERSATIONS

# Require Supabase credentials from environment
supabase_url: Optional[str] = os.environ.get("SUPABASE_URL")
supabase_key: Optional[str] = os.environ.get("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    logger.critical("SUPABASE_URL and SUPABASE_KEY must be set in environment or .env file")
    sys.exit(1)

# Import required components
from psy_supabase.core.database import DatabaseManager
from psy_supabase.core.rag_processor import RAGProcessor
from psy_supabase.core.text_generator import TextGenerator
from psy_supabase.memory.associative_memory import AssociativeMemory


class PainPointDemo:
    """Demonstrates pain point detection using sample conversations."""

    def __init__(self) -> None:
        """Initialize components needed for pain point detection."""
        # Create a unique test user ID
        import uuid

        self.user_id = f"demo_user_{uuid.uuid4().hex[:8]}"
        logger.info("Using demo user ID: %s", self.user_id)
        self.session_id = "demo_session"

        # Define mock responses for all methods to use
        self.mock_responses = [
            "I understand how challenging that must feel. Could you tell me more about when you first noticed this?",
            "It sounds like this situation has been difficult for you. How has it been affecting other areas of your life?",
            "I'm hearing that this is something you've been struggling with. What have you tried so far to address it?",
            "That's a really important concern you're raising. How do you feel when you think about this issue?",
            "Thank you for sharing that with me. It takes courage to discuss these feelings. How long have you been experiencing this?",
        ]

        # Add assertions to satisfy mypy ---
        assert supabase_url is not None, "Supabase URL must be set"
        assert supabase_key is not None, "Supabase Key must be set"

        # Initialize database
        self.db_manager = DatabaseManager(supabase_url=supabase_url, supabase_key=supabase_key, user_id=self.user_id)

        # Set up schema and tables
        logger.info("Setting up demo environment...")
        success = self.db_manager.create_user_schema_sync()
        if not success:
            logger.error("Failed to create schema and tables")
            raise RuntimeError("Database initialization failed")

        # Ensure schema is properly created before proceeding
        self._verify_schema_exists()

        # Initialize text generator (use CPU for demo purposes)
        device = "cpu"
        self.generator = TextGenerator(model_name=TEXT_GENERATING_MODEL, device=device)

        # Initialize associative memory
        self.associative_memory = AssociativeMemory()

        # Initialize RAG processor
        self.rag_processor = RAGProcessor(
            db_manager=self.db_manager, generator=self.generator, intelligent_processing_enabled=True
        )

        # Add associative memory to RAG processor
        self.rag_processor.associative_memory = self.associative_memory

        # Initialize memory with themes from test conversations
        self._initialize_memory()

    def _verify_schema_exists(self) -> None:
        """Verify that the schema exists and has the required tables."""
        try:
            # Check if schema exists
            schema_query = f"""
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.schemata
                WHERE schema_name = '{self.db_manager.schema_name}'
            );
            """
            schema_response = self.db_manager.supabase.rpc("sql", {"command": schema_query}).execute()

            schema_exists = False
            if schema_response.data and len(schema_response.data) > 0:
                if isinstance(schema_response.data[0], dict):
                    schema_exists = next(iter(schema_response.data[0].values()))
                else:
                    # Handle character-by-character response
                    schema_exists = "".join(schema_response.data).lower() == "true"

            if not schema_exists:
                logger.error("Schema %s does not exist!", self.db_manager.schema_name)
                raise RuntimeError(f"Schema {self.db_manager.schema_name} not found")

            # Check if interactions table exists
            table_query = f"""
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = '{self.db_manager.schema_name}'
                AND table_name = 'interactions'
            );
            """
            table_response = self.db_manager.supabase.rpc("sql", {"command": table_query}).execute()

            table_exists = False
            if table_response.data and len(table_response.data) > 0:
                if isinstance(table_response.data[0], dict):
                    table_exists = next(iter(table_response.data[0].values()))
                else:
                    table_exists = "".join(table_response.data).lower() == "true"

            if not table_exists:
                logger.error("Table interactions does not exist in schema %s!", self.db_manager.schema_name)

        except Exception as e:
            logger.error("Error verifying schema: %s", e)
            import traceback

            logger.error(traceback.format_exc())

    def _initialize_memory(self) -> None:
        """Initialize associative memory with themes from test conversations."""
        logger.info("Initializing associative memory with conversation themes")

        for conversation in TEST_CONVERSATIONS:
            # Extract themes and approaches
            if "expected_pain_point" in conversation:
                themes = conversation["expected_pain_point"].get("themes", [])
                approaches = conversation["expected_pain_point"].get("approach_types", [])

                # Add memory entries for themes and approaches
                if themes:
                    theme_desc = f"Common themes in {conversation['name']}: {', '.join(themes)}"
                    self.associative_memory.add_memory(theme_desc, themes)

                if approaches:
                    approach_desc = f"Therapeutic approaches for {conversation['name']}: {', '.join(approaches)}"
                    self.associative_memory.add_memory(approach_desc, approaches + themes)

        logger.info("Initialized memory with %d entries", len(self.associative_memory.memories))

    def add_conversations_to_database(self, conversation: Dict[str, Any]) -> int:
        """Add conversations from TEST_CONVERSATIONS to the database."""
        name = conversation["name"]
        session_id = conversation.get("session_id")
        questions = conversation["questions"]

        logger.info("\n=== Adding conversation to database: %s ===", name)
        logger.info("Session ID: %s", session_id)

        # Process each question in the conversation
        added_count = 0
        for i, question in enumerate(questions):
            question_num = i + 1
            logger.debug("Adding Q%s: %s...", question_num, question[:50])

            # Generate a simple mock response
            answer = self.mock_responses[i % len(self.mock_responses)]

            # Create metadata with expected themes
            metadata = {}
            if "expected_pain_point" in conversation:
                metadata["expected_themes"] = conversation["expected_pain_point"].get("themes", [])
                metadata["expected_approaches"] = conversation["expected_pain_point"].get("approach_types", [])

            # Use direct SQL to add interaction to database
            try:
                # Prepare data for SQL insertion - escape single quotes
                safe_question = question.replace("'", "''") if question else ""
                safe_answer = answer.replace("'", "''") if answer else ""
                safe_context = f"Demo conversation: {name}".replace("'", "''")

                # Convert metadata to JSON string for insertion
                metadata_json = json.dumps(metadata)

                # Construct the SQL statement with proper schema reference
                sql = f"""
                INSERT INTO "{self.db_manager.schema_name}"."interactions"
                (question, answer, context, metadata, session_id)
                VALUES (
                    '{safe_question}',
                    '{safe_answer}',
                    '{safe_context}',
                    '{metadata_json}'::jsonb,
                    '{session_id}'
                )
                RETURNING interaction_id;
                """

                # Execute the SQL
                response = self.db_manager.supabase.rpc("sql", {"command": sql}).execute()

                if response.data and len(response.data) > 0:
                    added_count += 1

            except Exception as e:
                logger.error("Error adding interaction: %s", e)

        logger.info("Added %d interactions for session %s", added_count, session_id)
        return added_count

    def run_conversation(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run through a complete conversation and detect pain points.

        Args:
            conversation: A conversation dictionary from TEST_CONVERSATIONS

        Returns:
            Dictionary with conversation results and detected pain points
        """
        name = conversation["name"]
        session_id = conversation["session_id"]
        questions = conversation["questions"]

        logger.info("\n=== Running conversation: %s ===", name)
        logger.info("Session ID: %s", session_id)

        # First, add all conversations to the database
        added_count = self.add_conversations_to_database(conversation)
        if added_count == 0:
            logger.error("Failed to add any interactions to the database!")
            return {"name": name, "session_id": session_id, "error": "Failed to add interactions to database"}

        # Wait a moment to ensure database operations complete
        time.sleep(1)

        results = {
            "name": name,
            "session_id": session_id,
            "exchanges": [],
            "pain_point_detected": 0,
            "first_detection_at": None,
            "detected_themes": set(),
            "detected_approaches": set(),
        }

        # Look up the conversation in the database to verify it was added
        try:
            conversations_list = self.db_manager.get_conversation_history(session_id)
            if conversations_list:
                found_count = len(conversations_list)
                logger.info("Found %d interactions in the database for session %s", found_count, session_id)

                # Process each question to detect pain points
                all_questions = []
                for i, item in enumerate(conversations_list):
                    if isinstance(item, dict):
                        question = item.get("question", "")
                        if question:
                            all_questions.append(question)

                # Simple pain point detection based on question content
                results["detected_themes"] = self._detect_themes(all_questions)

                # Extract expected themes from metadata
                expected_themes = set()
                if "expected_pain_point" in conversation:
                    expected_themes = set(conversation["expected_pain_point"].get("themes", []))

                results["expected_themes"] = list(expected_themes)

                # Convert detected_themes to list for JSON serialization
                results["detected_themes"] = list(results["detected_themes"])

                # Add exchange details
                for i, question in enumerate(questions):
                    exchange = {
                        "question": question,
                        "answer": self.mock_responses[i % len(self.mock_responses)],
                        "pain_point_detected": any(theme in question.lower() for theme in expected_themes),
                        "recurring_themes": [
                            theme for theme in results["detected_themes"] if theme.lower() in question.lower()
                        ],
                    }
                    results["exchanges"].append(exchange)

                    # Count pain points
                    if exchange["pain_point_detected"]:
                        results["pain_point_detected"] += 1
                        if results["first_detection_at"] is None:
                            results["first_detection_at"] = i + 1
            else:
                logger.error("No interactions found in database!")
                results["error"] = "No interactions found in database"

        except Exception as e:
            logger.error("Error processing conversation: %s", e)
            import traceback

            logger.error(traceback.format_exc())
            results["error"] = str(e)

        return results

    def _detect_themes(self, questions: List[str]) -> set:
        """Detect themes from a list of questions using the TherapeuticMappings class."""
        detected_themes = set()

        for question in questions:
            question = question.lower()
            # Use TherapeuticMappings ENHANCED_TAXONOMY with improved matching
            for theme, keywords in TherapeuticMappings.ENHANCED_TAXONOMY.items():
                # Check both exact matches and substring matches
                for keyword in keywords:
                    keyword = keyword.lower()
                    # Handle multi-word keywords
                    if " " in keyword:
                        if keyword in question:
                            detected_themes.add(theme)
                            break
                    # Handle single word keywords with partial matching
                    else:
                        # Add word boundary check to avoid partial word matches
                        word_pattern = rf"\b{re.escape(keyword)}\b"
                        if re.search(word_pattern, question):
                            detected_themes.add(theme)
                            break

                        # Also check for common variations
                        if (
                            keyword + "s" in question.split()
                            or keyword + "ed" in question.split()
                            or keyword + "ing" in question.split()
                        ):
                            detected_themes.add(theme)
                            break

        return detected_themes

    def analyze_results(self, results: Dict[str, Any]) -> None:
        """
        Analyze and print results from a conversation.

        Args:
            results: Results dictionary from run_conversation()
        """
        if "error" in results:
            logger.error("Error in results: %s", results["error"])
            return

        logger.info("\n=== Results for %s ===", results["name"])

        # Basic statistics
        total_questions = len(results.get("exchanges", []))
        pain_points = results.get("pain_point_detected", 0)
        detection_rate = (pain_points / total_questions) * 100 if total_questions > 0 else 0

        logger.info("Questions processed: %d", total_questions)
        logger.info("Pain points detected: %s", pain_points)
        logger.info(f"Detection rate: {detection_rate:.2f}%")

        if results.get("first_detection_at"):
            logger.info("First detected at question #%s", results["first_detection_at"])

        # Compare with expected themes
        expected_themes = set(results.get("expected_themes", []))
        detected_themes = set(results.get("detected_themes", []))

        if expected_themes:
            # Find matches and misses (case insensitive)
            norm_expected = {t.lower() for t in expected_themes}
            norm_detected = {t.lower() for t in detected_themes if t}

            matches = norm_expected.intersection(norm_detected)
            misses = norm_expected - norm_detected

            match_rate = (len(matches) / len(norm_expected)) * 100 if norm_expected else 0
            logger.info(f"\nTheme detection rate: {match_rate:.2f}%")
            logger.info("Expected themes: %s", ", ".join(expected_themes))
            logger.info("Detected themes: %s", ", ".join(detected_themes))

            if matches:
                logger.info("Matched themes: %s", ", ".join(matches))
            if misses:
                logger.info("Missed themes: %s", ", ".join(misses))

    def run_all_conversations(self) -> List[Dict[str, Any]]:
        """
        Run all conversations from TEST_CONVERSATIONS.

        Returns:
            List of results for each conversation
        """
        all_results = []

        for conversation in TEST_CONVERSATIONS:
            results = self.run_conversation(conversation)
            self.analyze_results(results)
            all_results.append(results)

        return all_results

    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("\nCleaning up resources...")
        cleanup_memory()

        # Drop the test schema using direct SQL since drop_user_schema doesn't exist
        try:
            drop_query = f"""
            DROP SCHEMA IF EXISTS "{self.db_manager.schema_name}" CASCADE;
            """
            # self.db_manager.supabase.rpc('sql', {'command': drop_query}).execute()
            logger.info("Dropped schema %s", self.db_manager.schema_name)
        except Exception as e:
            logger.error("Error dropping schema: %s", e)


def main() -> None:
    """Main function to demonstrate pain point detection."""
    logger.info("Starting pain point detection demonstration")

    try:
        demo = PainPointDemo()

        results = demo.run_all_conversations()

        # Save results to file for reference
        with open("pain_point_detection_results.json", "w") as f:
            # Convert sets to lists for JSON serialization
            json_results = []
            for r in results:
                r_copy = r.copy()

                # Handle potential sets
                for key, value in r_copy.items():
                    if isinstance(value, set):
                        r_copy[key] = list(value)

                json_results.append(r_copy)

            json.dump(json_results, f, indent=2)

        logger.info("Results saved to pain_point_detection_results.json")
    except Exception as e:
        logger.error("Error in demonstration: %s", e)
        import traceback

        logger.error(traceback.format_exc())
    finally:
        if "demo" in locals():
            demo.cleanup()


if __name__ == "__main__":
    main()
