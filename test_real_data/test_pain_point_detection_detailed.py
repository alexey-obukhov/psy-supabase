""" Validate pain point detection with detailed metrics. """

import json
import os
import sys
import time
import uuid
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, cast

import matplotlib.pyplot as plt
import pandas as pd
import torch
from prismalog.log import get_logger

# Add project to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from psy_supabase.core.database import DatabaseManager
from psy_supabase.core.rag_processor import RAGProcessor
from psy_supabase.core.text_generator import TextGenerator
from psy_supabase.utilities.common import is_github_actions
from psy_supabase.utilities.utils import cleanup_memory

# Configure logging
logger = get_logger(__name__)

# Load environment variables
if not is_github_actions():
    from dotenv import load_dotenv

    load_dotenv()

# Require Supabase credentials
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    logger.critical("Missing Supabase credentials")
    sys.exit(1)

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define test conversation scenarios with progressive complexity
TEST_PAIN_POINTS = [
    {
        "name": "Single Statement Workplace Anxiety",
        "description": "Tests if a clear workplace anxiety statement is detected immediately",
        "questions": ["I feel terrified every time my boss calls me into his office."],
        "expected": {
            "pain_point_detected": True,
            "topic": "workplace_anxiety",  # Assumes 'workplace_anxiety' is a detectable theme
            "emotion": "anxiety",
            "approach_type": "cognitive_behavioral",  # From 'workplace_anxiety' theme's approaches
        },
    },
    {
        "name": "Escalating Workplace Anxiety",
        "description": "Tests detection with increasing emotional intensity",
        "questions": [
            "I've been feeling nervous about going to work lately.",
            "My hands shake before team meetings when I have to present.",
            "I'm having panic attacks in the bathroom at work and nobody knows.",
        ],
        "expected": {
            "pain_point_detected": [False, True, True],
            "detection_threshold": 2,
            "final_topic": "workplace_anxiety",  # Assumes 'workplace_anxiety' is the consolidated topic
        },
    },
    {
        "name": "Mixed Signals Relationship Issues",
        "description": "Tests ability to identify pain points with ambiguous content",
        "questions": [
            "My partner and I had an argument last night about finances.",
            "I'm not sure if we're compatible anymore, but leaving feels impossible.",
            "Sometimes I wonder if I'm just staying because I'm afraid of being alone.",
        ],
        "expected": {
            "pain_point_detected": [False, True, True],
            # Changed from topic_progression to a more likely final state
            "final_topic": "relationship_issues",
            "final_approach_type": "interpersonal_therapy",  # From 'relationship_issues' theme
        },
    },
    {
        "name": "Masked Depression",
        "description": "Tests detection of depression when not explicitly stated",
        "questions": [
            "I haven't been able to enjoy things that used to make me happy.",
            "Getting out of bed feels like climbing a mountain lately.",
            "Sometimes I wonder what the point of trying is anymore.",
        ],
        "expected": {
            "pain_point_detected": [False, True, True],
            "final_topic": "depression",  # Added expected final topic
            "final_emotion": "depression",
            "approach_type": "self_compassion",  # 'self_compassion' is in 'depression' theme's approaches
        },
    },
    {
        "name": "Topic Switching",
        "description": "Tests if system can track multiple pain points across topics",
        "questions": [
            "I've been arguing with my spouse constantly about little things.",
            "At work, I made a mistake that might cost the company money.",
            "I can't sleep thinking about all the things I'm failing at.",
        ],
        "expected": {"pain_point_detected": [True, True, True], "multiple_topics": True, "tracks_consistently": True},
    },
    {
        "name": "Denial Pattern",
        "description": "Tests detection when user is in denial about issues",
        "questions": [
            "I'm totally fine, just wanted to chat.",
            "Well, sometimes I do feel a bit overwhelmed, but it's normal right?",
            "OK fine, I've been having panic attacks but I don't want to make a big deal of it.",
        ],
        "expected": {
            "pain_point_detected": [False, False, True],
            "detection_threshold": 3,
            "final_topic": "anxiety",
            "emotion_progression": ["neutral", "concern", "anxiety"],
            "final_approach_type": "cognitive_behavioral",  # From 'anxiety' theme
        },
    },
    {
        "name": "Self-Harm Risk",
        "description": "Tests detection of serious mental health risks",
        "questions": [
            "I've been thinking a lot about whether anyone would miss me if I wasn't here.",
            "Sometimes I think about ways to make the pain stop permanently.",
        ],
        "expected": {
            "pain_point_detected": [True, True],
            "high_priority": True,
            "crisis_detection": True,
            "final_topic": "crisis",  # Assuming a 'crisis' topic can be identified
            "final_approach_type": "crisis_intervention",  # Maps to 'crisis_support' template
        },
    },
    {
        "name": "PTSD Indicators",
        "description": "Tests detection of trauma symptoms",
        "questions": [
            "I keep having flashbacks to the accident whenever I hear a loud noise.",
            "I've been avoiding driving altogether since it happened.",
            "Sometimes I feel like I'm right back there, even though it was years ago.",
        ],
        "expected": {
            "pain_point_detected": [True, True, True],
            "topic": "trauma",  # 'trauma' is a defined theme
            "approach_type": "trauma",  # 'trauma' is an approach in the 'trauma' theme
        },
    },
]


class PainPointDetailedTester:
    """Specialized tester for pain point detection mechanisms."""

    def __init__(self) -> None:
        """Initialize the test environment."""
        # Create unique test identifiers
        self.test_user_id = f"test_user_{uuid.uuid4().hex[:8]}"
        self.test_session_id = f"pp_detailed_{uuid.uuid4().hex[:10]}"
        logger.info(f"Test session ID: {self.test_session_id}")

        supabase_url_test = cast(str, supabase_url)
        supabase_key_test = cast(str, supabase_key)

        # Initialize core components
        self.db_manager = DatabaseManager(
            supabase_url=supabase_url_test, supabase_key=supabase_key_test, user_id=self.test_user_id
        )
        self.generator = TextGenerator(model_name="rasyosef/Phi-1_5-Instruct-v0.1", device=device)
        self.rag_processor = RAGProcessor(
            db_manager=self.db_manager, generator=self.generator, intelligent_processing_enabled=True
        )

        # Test setup
        self._setup_test_environment()

        # Test metrics storage
        self.all_results: List[Dict[str, Any]] = []
        self.metrics = {
            "detection_rate": 0,
            "first_detection_avg": 0,
            "topic_accuracy": 0,
            "emotion_accuracy": 0,
            "approach_type_accuracy": 0,
        }

    def _setup_test_environment(self) -> None:
        """Set up test database and environment."""
        logger.info("Setting up test environment...")
        self.db_manager.create_user_schema_sync()
        logger.info("Test environment setup complete.")

    def load_test_cases(self) -> List[Dict[str, Any]]:
        """Load test cases from predefined data."""
        return [
            {
                "expected_pain_points": ["trauma", "flashback", "ptsd"],
                "expected_topic": "trauma",
                "expected_emotion": "anxiety",
                "expected_approach": "trauma",
            },
            {
                "expected_pain_points": ["anxiety", "worry", "stress"],
                "expected_topic": "anxiety",
                "expected_emotion": "anxiety",
                "expected_approach": "cognitive_behavioral",
            },
            {
                "expected_pain_points": ["depression", "sad", "hopeless"],
                "expected_topic": "depression",
                "expected_emotion": "sadness",
                "expected_approach": "behavioral_activation",
            },
            {
                "expected_pain_points": ["relationship", "partner", "breakup"],
                "expected_topic": "relationship",
                "expected_emotion": "concern",
                "expected_approach": "interpersonal_therapy",
            },
            {
                "expected_pain_points": ["grief", "loss", "death"],
                "expected_topic": "grief",
                "expected_emotion": "sadness",
                "expected_approach": "grief_processing",
            },
            {
                "expected_pain_points": ["shame", "embarrassment", "humiliation"],
                "expected_topic": "shame",
                "expected_emotion": "shame",
                "expected_approach": "compassion_focused_therapy",
            },
            {
                "expected_pain_points": ["guilt", "regret", "remorse"],
                "expected_topic": "guilt",
                "expected_emotion": "guilt",
                "expected_approach": "cognitive_behavioral",
            },
            {
                "expected_pain_points": ["work", "job", "career", "boss"],
                "expected_topic": "workplace_anxiety",
                "expected_emotion": "anxiety",
                "expected_approach": "cognitive_behavioral",
            },
        ]

    def run_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single test case and gather detailed metrics.

        Args:
            test_case: The test configuration

        Returns:
            Dictionary with test results and metrics
        """
        logger.info(f"Running test: {test_case['name']} - {test_case['description']}")
        test_case_start_time = time.time()  # DEBUG: Start timer for the whole test case

        results = {
            "name": test_case["name"],
            "exchanges": [],
            "pain_points_detected": 0,
            "total_questions": len(test_case["questions"]),
            "first_detection_at": None,
            "topics_detected": [],
            "emotions_detected": [],
            "approach_types_used": [],
            "detection_timeline": [],
            "metrics": {},
        }

        # Process each question
        for i, question in enumerate(test_case["questions"]):
            start_time = time.time()
            logger.info(f"Question {i+1}: {question[:50]}...")
            logger.debug(f"DEBUG: Starting processing for question {i+1} in test '{test_case['name']}'")  # DEBUG

            # Generate response
            logger.debug(f"DEBUG: Calling RAGProcessor.generate_response for question {i+1}...")  # DEBUG
            rag_call_start_time = time.time()  # DEBUG
            response = self.rag_processor.generate_response(
                user_question=question, session_id=self.test_session_id, device=device, question_id=i
            )
            rag_call_duration = time.time() - rag_call_start_time  # DEBUG
            logger.debug(
                f"DEBUG: RAGProcessor.generate_response for question {i+1} took {rag_call_duration:.2f}s"
            )  # DEBUG
            response_time = time.time() - start_time  # This is overall for the question processing after RAG

            logger.debug(f"DEBUG: Fetching conversation history for question {i+1}...")  # DEBUG
            db_call_start_time = time.time()  # DEBUG
            history = self.db_manager.get_conversation_history(self.test_session_id)
            db_call_duration = time.time() - db_call_start_time  # DEBUG
            logger.debug(
                f"DEBUG: Fetching conversation history for question {i+1} took {db_call_duration:.2f}s"
            )  # DEBUG
            latest_interaction = history[-1] if history else {}

            current_metadata: Dict[str, Any] = {}  # Default to empty dict
            raw_metadata_from_db = latest_interaction.get("metadata")

            if isinstance(raw_metadata_from_db, list):
                logger.debug(f"Raw metadata from DB is a list. Content: {str(raw_metadata_from_db)[:250]}...")
                if raw_metadata_from_db:  # If list is not empty
                    found_dict_from_string_in_list = False
                    # Prioritize parsing a JSON string from the list
                    for item in raw_metadata_from_db:
                        if isinstance(item, str):
                            try:
                                parsed_item = json.loads(item)
                                if isinstance(parsed_item, dict):
                                    logger.info("Successfully parsed a string item from metadata list into a dict.")
                                    current_metadata = parsed_item
                                    found_dict_from_string_in_list = True
                                    break  # Use the first successfully parsed dict from a string
                            except json.JSONDecodeError:
                                logger.debug(f"Could not parse string item from list: {str(item)[:100]}")
                                continue

                    # If no dict was found from parsing strings in the list,
                    # check if any list item is already a suitable dict
                    if not found_dict_from_string_in_list:
                        candidate_dict_from_list: Dict[str, Any] = {}
                        for item in raw_metadata_from_db:
                            if isinstance(item, dict):
                                # Prefer a dict that seems more complete or is found first
                                if not candidate_dict_from_list or len(item) > len(candidate_dict_from_list):
                                    candidate_dict_from_list = item

                        if candidate_dict_from_list:
                            logger.info(
                                "Using a direct dictionary item from metadata list as no parsable string was primary."
                            )
                            current_metadata = candidate_dict_from_list
                        else:
                            logger.warning(
                                "Metadata list did not contain a parsable JSON string or a direct dictionary. Using empty metadata."
                            )
                else:
                    logger.warning("Metadata list from DB was empty. Using empty metadata.")

            elif isinstance(raw_metadata_from_db, dict):
                logger.debug("Raw metadata from DB is a dict.")
                current_metadata = raw_metadata_from_db
            elif isinstance(raw_metadata_from_db, str):
                logger.debug(f"Raw metadata from DB is a string: {raw_metadata_from_db[:100]}...")
                try:
                    parsed_str_metadata = json.loads(raw_metadata_from_db)
                    if isinstance(parsed_str_metadata, dict):
                        current_metadata = parsed_str_metadata
                    else:
                        logger.error(
                            f"Parsed metadata string is not a dict: {type(parsed_str_metadata)}. Value: {str(parsed_str_metadata)[:200]}"
                        )
                        # current_metadata remains {}
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing metadata string: '{raw_metadata_from_db}', Error: {e}")
                    # current_metadata remains {}
            elif raw_metadata_from_db is None:
                logger.debug("Raw metadata from DB is None. Using empty metadata.")
            else:  # Other unexpected types
                logger.warning(
                    f"Unexpected metadata type from DB: {type(raw_metadata_from_db)}. Value: {str(raw_metadata_from_db)[:200]}. Using empty metadata."
                )

            logger.debug(f"DEBUG: Metadata processing for question {i+1} complete.")  # DEBUG

            # Extract pain point data using the correctly processed metadata
            pain_point_detected = self._extract_pain_point_detection(current_metadata)
            topic = self._extract_topic(current_metadata)
            emotion = self._extract_emotion(current_metadata)
            approach_type = self._extract_approach_type(current_metadata)

            # Extract template and similarity more reliably from pain_point_results if available
            pain_point_results_data = current_metadata.get("pain_point_results", {})
            if not isinstance(pain_point_results_data, dict):  # Ensure it's a dict
                pain_point_results_data = {}

            template = pain_point_results_data.get(
                "template_used", current_metadata.get("template_used", "unknown_template")
            )
            similarity = pain_point_results_data.get("similarity", 0.0)

            logger.info(
                f"Result: Pain point: {pain_point_detected}, "
                f"Topic: {topic}, Emotion: {emotion}, "
                f"Approach: {approach_type}, Template: {template}, Similarity: {similarity:.2f}"
            )
            logger.debug(f"DEBUG: Question {i+1} processing took {response_time:.2f}s (incl. RAG and DB)")  # DEBUG

            exchange = {
                "question": question,
                "response": response,
                "pain_point_detected": pain_point_detected,
                "topic": topic,
                "emotion": emotion,
                "approach_type": approach_type,
                "template": template,
                "similarity": similarity,
                "response_time": response_time,
                "metadata": current_metadata,  # Store corrected metadata
            }
            results["exchanges"].append(exchange)
            results["detection_timeline"].append(pain_point_detected)

            # Update statistics
            if pain_point_detected:
                results["pain_points_detected"] += 1
                if results["first_detection_at"] is None:
                    results["first_detection_at"] = i + 1

            if topic and topic != "unknown":
                results["topics_detected"].append(topic)

            if emotion and emotion != "unknown":
                results["emotions_detected"].append(emotion)

            if approach_type and approach_type != "empathy_validation":
                results["approach_types_used"].append(approach_type)

            # Debug final metadata structure on the last question
            if i == len(test_case["questions"]) - 1:  # On the last question
                logger.info(f"DEBUG - Final processed metadata structure: {json.dumps(current_metadata, indent=2)}")

        # Calculate metrics
        results["metrics"] = self._calculate_metrics(results, test_case["expected"])
        test_case_duration = time.time() - test_case_start_time
        logger.info(
            f"Test complete. Detected {results['pain_points_detected']}/{results['total_questions']} pain points. Test case duration: {test_case_duration:.2f}s"
        )

        return results

    def _extract_pain_point_detection(self, metadata: Dict[str, Any]) -> bool:
        """Extract pain point detection status from metadata."""
        # First check direct flag
        if "pain_point_detected" in metadata:
            return metadata["pain_point_detected"]

        # Then check pain_points list structure
        if metadata.get("pain_points") and isinstance(metadata.get("pain_points"), list):
            pain_points = metadata.get("pain_points")
            if pain_points and pain_points[-1].get("detected"):
                return True

        # Then check other indicators
        if "pain_point" in metadata and metadata["pain_point"]:
            return True

        if "approach_type" in metadata and metadata["approach_type"]:
            return True

        return False

    def _extract_topic(self, metadata: Dict[str, Any]) -> str:
        """Extract detected topic from metadata with improved path checking."""
        # First check direct topic
        if "topic" in metadata and metadata["topic"] not in ("unknown", ""):
            return metadata["topic"]

        # Check topics_context object
        topics_context = metadata.get("topics_context", {})
        if topics_context and isinstance(topics_context, dict):
            topic = topics_context.get("topic")
            if topic and topic not in ("unknown", ""):
                return topic

        # Check pain_points list
        pain_points = metadata.get("pain_points", [])
        if pain_points and isinstance(pain_points, list) and len(pain_points) > 0:
            if isinstance(pain_points[-1], dict) and "name" in pain_points[-1]:
                return pain_points[-1]["name"]

        return "unknown"

    def _extract_emotion(self, metadata: Dict[str, Any]) -> str:
        """Extract detected emotion with improved path checking."""
        # Direct emotion check
        if "emotion" in metadata and metadata["emotion"] not in ("unknown", ""):
            return metadata["emotion"]

        # Check topics_context
        topics_context = metadata.get("topics_context", {})
        if topics_context and isinstance(topics_context, dict):
            emotion = topics_context.get("emotion")
            if emotion and emotion not in ("unknown", ""):
                return emotion

        # Check emotion_detected or emotional_state
        for key in ["emotion_detected", "emotional_state", "primary_emotion"]:
            if key in metadata and metadata[key]:
                return metadata[key]

        return "unknown"

    def _extract_approach_type(self, metadata: Dict[str, Any]) -> str:
        """Extract therapeutic approach with improved path checking."""
        # Check all possible paths for approach type
        approach_keys = ["approach_type", "therapeutic_approach", "therapy_approach", "intervention_strategy"]

        for key in approach_keys:
            if key in metadata and metadata[key] and metadata[key] != "unknown":
                return metadata[key]

        # Check nested structures
        if "suggested_approach" in metadata and isinstance(metadata["suggested_approach"], dict):
            for key in approach_keys:
                if key in metadata["suggested_approach"] and metadata["suggested_approach"][key]:
                    return metadata["suggested_approach"][key]

        # Try to derive from template
        template = metadata.get("template_used", "")
        if "_" in template and "therapy" in template.lower():
            parts = template.split("_")
            if len(parts) > 2:  # Ensure there's a meaningful approach part
                return parts[-1]

        return "unknown"

    def _calculate_metrics(self, results: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate test metrics based on expected outcomes."""
        metrics = {}

        # Detection rate
        detection_rate = results["pain_points_detected"] / results["total_questions"] * 100
        metrics["detection_rate"] = detection_rate

        # Analyze expected detection patterns
        if "pain_point_detected" in expected and isinstance(expected["pain_point_detected"], list):
            expected_detections = expected["pain_point_detected"]
            actual_detections = results["detection_timeline"]

            # Calculate accuracy of detection pattern
            matches = sum(1 for e, a in zip(expected_detections, actual_detections) if e == a)
            metrics["detection_pattern_accuracy"] = (matches / len(expected_detections)) * 100

        # Check for detection threshold
        if "detection_threshold" in expected and results["first_detection_at"]:
            meets_threshold = results["first_detection_at"] <= expected["detection_threshold"]
            metrics["meets_detection_threshold"] = meets_threshold

        # Topic analysis
        if "topic" in expected and results["topics_detected"]:
            # Check if expected topic appears in detected topics
            topic_found = expected["topic"] in results["topics_detected"]
            metrics["topic_found"] = topic_found

        if "final_topic" in expected and results["topics_detected"]:
            # Check if final topic matches expectations
            final_topic_match = results["topics_detected"][-1] == expected["final_topic"]
            metrics["final_topic_match"] = final_topic_match

        # Emotion analysis
        if "final_emotion" in expected and results["emotions_detected"]:
            final_emotion_match = results["emotions_detected"][-1] == expected["final_emotion"]
            metrics["final_emotion_match"] = final_emotion_match

        # Approach type check
        if "approach_type" in expected and results["approach_types_used"]:
            # Check if expected approach type is used
            approach_found = expected["approach_type"] in results["approach_types_used"]
            metrics["approach_found"] = approach_found

        # Special case checks
        if "multiple_topics" in expected:
            unique_topics = len(set(results["topics_detected"]))
            metrics["multiple_topics_detected"] = unique_topics > 1

        if "crisis_detection" in expected:
            # Add specific check for crisis detection
            metrics["crisis_detected"] = False

            # Check for crisis indicators in metadata
            for exchange in results["exchanges"]:
                metadata = exchange.get("metadata", {})
                if metadata.get("crisis_indicators") or metadata.get("high_risk"):
                    metrics["crisis_detected"] = True
                    break

        return metrics

    def run_all_tests(self) -> List[Dict[str, Any]]:
        """Run all test cases and compile results."""
        all_results = []

        for test_case in TEST_PAIN_POINTS:
            result = self.run_test(test_case)
            all_results.append(result)
            logger.info(f"Completed test: {test_case['name']}")
            logger.info("-" * 50)

        self.all_results = all_results
        return all_results

    def analyze_results(self) -> Dict[str, Any]:
        """Analyze aggregate results and calculate overall metrics."""
        if not self.all_results:
            logger.warning("No test results to analyze")
            return {}

        # Overall stats
        total_exchanges = sum(r["total_questions"] for r in self.all_results)
        total_detected = sum(r["pain_points_detected"] for r in self.all_results)
        overall_detection_rate = (total_detected / total_exchanges) * 100 if total_exchanges else 0

        # Detection timing stats
        first_detection_points = [r["first_detection_at"] for r in self.all_results if r["first_detection_at"]]
        avg_first_detection = sum(first_detection_points) / len(first_detection_points) if first_detection_points else 0

        # Accuracy metrics
        pattern_accuracy_scores = [
            r["metrics"].get("detection_pattern_accuracy", 0)
            for r in self.all_results
            if "detection_pattern_accuracy" in r["metrics"]
        ]
        avg_pattern_accuracy = (
            sum(pattern_accuracy_scores) / len(pattern_accuracy_scores) if pattern_accuracy_scores else 0
        )

        # Topic and emotion accuracy
        topic_scores = [
            1 if r["metrics"].get("topic_found", False) or r["metrics"].get("final_topic_match", False) else 0
            for r in self.all_results
        ]
        topic_accuracy = sum(topic_scores) / len(topic_scores) * 100 if topic_scores else 0

        emotion_scores = [
            1 if r["metrics"].get("final_emotion_match", False) else 0
            for r in self.all_results
            if "final_emotion_match" in r["metrics"]
        ]
        emotion_accuracy = sum(emotion_scores) / len(emotion_scores) * 100 if emotion_scores else 0

        # Approach type accuracy
        approach_scores = [
            1 if r["metrics"].get("approach_found", False) else 0
            for r in self.all_results
            if "approach_found" in r["metrics"]
        ]
        approach_accuracy = sum(approach_scores) / len(approach_scores) * 100 if approach_scores else 0

        # Crisis detection accuracy
        crisis_test_results = [r for r in self.all_results if "crisis_detection" in r.get("expected", {})]
        crisis_scores = [1 if r["metrics"].get("crisis_detected", False) else 0 for r in crisis_test_results]
        crisis_accuracy = sum(crisis_scores) / len(crisis_scores) * 100 if crisis_scores else 0

        # Compile overall results
        metrics = {
            "overall_detection_rate": overall_detection_rate,
            "avg_first_detection": avg_first_detection,
            "pattern_accuracy": avg_pattern_accuracy,
            "topic_accuracy": topic_accuracy,
            "emotion_accuracy": emotion_accuracy,
            "approach_accuracy": approach_accuracy,
            "crisis_accuracy": crisis_accuracy,
        }

        self.metrics = metrics
        return metrics

    def generate_report(self, output_dir: str = ".") -> None:
        """
        Generate comprehensive report with visualizations.

        Args:
            output_dir: Directory to save report files
        """
        if not self.all_results:
            logger.warning("No results to generate report from")
            return

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        report_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create dataframes for analysis
        test_summary = []
        exchange_data = []

        for test in self.all_results:
            # Test level data
            test_summary.append(
                {
                    "test_name": test["name"],
                    "questions": test["total_questions"],
                    "detections": test["pain_points_detected"],
                    "detection_rate": test["pain_points_detected"] / test["total_questions"] * 100,
                    "first_detection": test["first_detection_at"],
                    **test["metrics"],
                }
            )

            # Exchange level data
            for i, ex in enumerate(test["exchanges"]):
                exchange_data.append(
                    {
                        "test_name": test["name"],
                        "question_num": i + 1,
                        "pain_point_detected": ex["pain_point_detected"],
                        "topic": ex["topic"],
                        "emotion": ex["emotion"],
                        "approach_type": ex["approach_type"],
                        "template": ex["template"],
                        "similarity": ex["similarity"],
                        "response_time": ex["response_time"],
                    }
                )

        # Convert to dataframes
        df_tests = pd.DataFrame(test_summary)
        df_exchanges = pd.DataFrame(exchange_data)

        # Generate visualizations
        plt.figure(figsize=(14, 8))

        # Plot 1: Detection rates by test
        plt.subplot(2, 2, 1)
        bar_heights = df_tests["detection_rate"].values
        plt.bar(df_tests["test_name"], bar_heights)
        plt.xticks(rotation=45, ha="right")
        plt.title("Pain Point Detection Rate by Test")
        plt.ylabel("Detection Rate (%)")
        plt.tight_layout()

        # Plot 2: First detection timing
        plt.subplot(2, 2, 2)
        valid_indices = ~df_tests["first_detection"].isna()
        if sum(valid_indices) > 0:
            plt.bar(df_tests.loc[valid_indices, "test_name"], df_tests.loc[valid_indices, "first_detection"])
            plt.xticks(rotation=45, ha="right")
            plt.title("First Pain Point Detection (Question #)")
            plt.ylabel("Question Number")
            plt.tight_layout()

        # Plot 3: Topic distribution
        plt.subplot(2, 2, 3)
        topic_counts = df_exchanges["topic"].value_counts()
        topic_counts = topic_counts[topic_counts.index != "unknown"]
        if not topic_counts.empty:
            topic_counts.plot(kind="bar")
            plt.title("Detected Topics Distribution")
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

        # Plot 4: Emotion distribution
        plt.subplot(2, 2, 4)
        emotion_counts = df_exchanges["emotion"].value_counts()
        emotion_counts = emotion_counts[emotion_counts.index != "unknown"]
        if not emotion_counts.empty:
            emotion_counts.plot(kind="bar")
            plt.title("Detected Emotions Distribution")
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

        # Save visualization
        plt.savefig(f"{output_dir}/pain_point_analysis_{report_time}.png", dpi=300, bbox_inches="tight")

        # Save data files
        df_tests.to_csv(f"{output_dir}/test_summary_{report_time}.csv", index=False)
        df_exchanges.to_csv(f"{output_dir}/exchange_data_{report_time}.csv", index=False)

        # Generate text report
        with open(f"{output_dir}/pain_point_report_{report_time}.txt", "w") as f:
            f.write("=== PAIN POINT DETECTION DETAILED ANALYSIS ===\n\n")
            f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("OVERALL METRICS:\n")
            for metric, value in self.metrics.items():
                f.write(f"  {metric.replace('_', ' ').title()}: {value:.2f}\n")

            f.write("\nTEST RESULTS:\n")
            for test in self.all_results:
                f.write(f"\n{test['name']}:\n")
                f.write(f"  Detection Rate: {test['pain_points_detected']}/{test['total_questions']} ")
                f.write(f"({test['pain_points_detected']/test['total_questions']*100:.2f}%)\n")
                f.write(
                    f"  First Detection: {'N/A' if test['first_detection_at'] is None else 'Question #' + str(test['first_detection_at'])}\n"
                )
                if test["topics_detected"]:
                    topic_counts = Counter(test["topics_detected"])
                    f.write(f"  Topics: {', '.join([f'{t}({c})' for t, c in topic_counts.most_common()])}\n")

                if test["emotions_detected"]:
                    emotion_counts = Counter(test["emotions_detected"])
                    f.write(f"  Emotions: {', '.join([f'{e}({c})' for e, c in emotion_counts.most_common()])}\n")

                f.write("  Metrics:\n")
                for metric, value in test["metrics"].items():
                    if isinstance(value, bool):
                        f.write(f"    {metric.replace('_', ' ').title()}: {'Yes' if value else 'No'}\n")
                    elif isinstance(value, (int, float)):
                        f.write(f"    {metric.replace('_', ' ').title()}: {value:.2f}\n")
                    else:
                        f.write(f"    {metric.replace('_', ' ').title()}: {value}\n")

        logger.info(f"Report generated in {output_dir}")

    def cleanup(self) -> None:
        """Clean up testing resources."""
        logger.info("Cleaning up test environment...")
        cleanup_memory()


def main() -> None:
    """Run the detailed pain point detection test suite."""
    logger.info("Starting detailed pain point detection test suite")

    try:
        # Initialize and run tests
        tester = PainPointDetailedTester()
        results = tester.run_all_tests()

        # Analyze results
        metrics = tester.analyze_results()

        # Print summary
        logger.info("=== PAIN POINT DETECTION TEST RESULTS ===")
        logger.info(f"Overall detection rate: {metrics['overall_detection_rate']:.2f}%")
        logger.info(f"Average first detection at question #{metrics['avg_first_detection']:.2f}")
        logger.info(f"Topic detection accuracy: {metrics['topic_accuracy']:.2f}%")
        # logger.info(f"Emotion detection accuracy: {metrics['emotion_accuracy']:.2f}%")
        logger.info(f"Approach selection accuracy: {metrics['approach_accuracy']:.2f}%")

        # Generate detailed report
        tester.generate_report("test_results")

        # Cleanup
        tester.cleanup()

    except Exception as e:
        logger.error(f"Error running tests: {e}")
        import traceback

        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
