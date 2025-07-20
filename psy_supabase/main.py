"""
# Psy-Supabase: An AI Therapy Assistant

This project implements a therapeutic AI assistant using retrieval-augmented generation (RAG)
and associative memory to provide mental health support conversations.

## Technical Overview

This project uses a multi-model architecture to provide comprehensive therapeutic capabilities:

1. **Microsoft Phi-1.5**: A lightweight yet capable language model with 1.3 billion parameters,
   used for primary response generation. This model offers good performance with reasonable
   hardware requirements, making it accessible for deployment on consumer hardware.

2. **Sentence-Transformers (all-MiniLM-L6-v2)**: Powers the associative memory component,
   creating 384-dimensional embeddings that connect related psychological concepts and enable
   semantic similarity calculations.

3. **Facebook RoBERTa Hate Speech Detector**: Uses the facebook/roberta-hate-speech-dynabench-r4-target
   model to perform content moderation and toxicity detection. This specialized model ensures
   conversations remain safe, appropriate, and therapeutic by identifying potentially harmful
   content before processing.

These models are augmented with:

1. **Retrieval-Augmented Generation (RAG)**: Enhances responses by retrieving relevant past
   interactions and knowledge documents from a Supabase vector database.

2. **Associative Memory**: Creates connections between related psychological concepts,
   allowing the system to draw on related information (e.g., connecting "anxiety" with
   "stress management techniques").

3. **Dynamic Template Selection**: Chooses appropriate therapeutic response templates based
   on detected pain points and conversation context.

4. **PostgreSQL Vector Search**: Implements semantic similarity search using pgvector
   to find relevant knowledge and past interactions.

## Key Components

- **RAGProcessor**: Orchestrates the retrieval and generation process
- **DynamicRAGRetriever**: Handles semantic search with caching
- **AssociativeMemory**: Links related psychological concepts
- **DatabaseManager**: Manages the Supabase vector database
- **Flask API**: Provides REST endpoints for the frontend application

## Configuration

The system supports various environment variables:
- `SUPABASE_URL` and `SUPABASE_KEY`: Required for database access
- `INTELLIGENT_PROCESS_ENABLED`: Toggle advanced processing features (default: true)

## Hardware Requirements

- **CPU Mode**: Works on any modern CPU with 8GB+ RAM
- **GPU Mode (Recommended)**: CUDA-compatible GPU with 6GB+ VRAM
- **Storage**: Minimum 1GB for model weights and application

## License

The code in this project is licensed under the MIT License. See LICENSE for details.

> Note: While this project uses the rasyosef/Phi-1_5-Instruct-v0.1 model, users should ensure they comply
> with Microsoft's licensing terms for the model itself, which may differ from the project code license.

## Citation

If you use this project in your research or derivative work, please cite:
@software{psy_supabase, author = {Alexey Obukhov},
title = {Psy-Supabase: An AI Therapy Assistant}, year = {2025},
url = {https://github.com/alexey-obukhov/psy-supabase} }

## Disclaimer

This AI assistant is designed as a research tool and technology demonstration. It is not a
replacement for professional mental health services. The system should not be used to diagnose
or treat any medical or psychological condition.
"""

import multiprocessing as mp

# Standard library imports
import os
import random
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import spacy
import torch

# Third-party imports
from flask import Flask, Response, g, jsonify, request
from flask_cors import CORS  # type: ignore[import-untyped]
from typeguard import install_import_hook

from psy_supabase.config import TEXT_GENERATING_MODEL
from psy_supabase.core.database import DatabaseManager
from psy_supabase.core.model_manager import get_model_manager
from psy_supabase.core.pain_point_detector import PainPointDetector
from psy_supabase.core.rag_processor import RAGProcessor

# Local imports
from psy_supabase.utilities.common import is_github_actions
from psy_supabase.utilities.nlp_utils import get_spacy_model
from psy_supabase.utilities.utils import cleanup_memory, parse_bool_env

# Install type checking
install_import_hook("psy_supabase")


from psy_supabase import get_package_logger

logger = get_package_logger(__name__)


# Initialize spaCy model at startup
nlp = get_spacy_model()
if nlp is None:
    logger.warning("Failed to initialize spaCy model. Some functionality will be limited.")

if not is_github_actions():
    from dotenv import load_dotenv

    load_dotenv()  # Load environment variables from .env file
    logger.info("Local development: Loading environment from .env file")
else:
    logger.info("CI environment: Using GitHub secrets")

# Memory management variables
last_memory_cleanup = time.time()
REQUEST_COUNTER = 0

# Pain point monitoring sessions - stores active monitoring sessions
PAIN_POINT_SESSIONS = {}


class PainPointSession:
    """Manages pain point detection for a user session."""

    def __init__(self, user_id: str, db_manager: DatabaseManager):
        self.user_id = user_id
        self.session_start = datetime.now()
        self.messages: List[Dict[str, Any]] = []
        self.pain_points: List[Dict[str, Any]] = []
        self.detector = PainPointDetector(db_manager)
        self.analytics: Dict[str, Any] = {
            "total_interactions": 0,
            "pain_points_detected": 0,
            "themes": Counter(),
            "similarity_scores": [],
            "last_activity": datetime.now(),
        }

    def analyze_message(self, message: str, response: str) -> dict:
        """Analyze a user message and AI response for pain points."""
        from psy_supabase.utilities.embedding_utils import calculate_similarity

        # Create message analysis
        message_data: Dict[str, Any] = {
            "id": len(self.messages) + 1,
            "timestamp": datetime.now(),
            "user_message": message,
            "ai_response": response,
            "similarity_scores": [],
            "is_pain_point": False,
            "theme": "general",
            "intensity": "low",
        }

        try:
            # Use the existing pain point detector
            pain_points = self.detector.detect_pain_points(
                session_id=self.user_id, threshold=0.6, min_occurrences=2, time_window_days=7
            )

            # Calculate similarities with previous messages
            max_similarity = 0.0
            most_similar_id = None

            for prev_msg in self.messages:
                try:
                    similarity = calculate_similarity(message, prev_msg["user_message"])
                    message_data["similarity_scores"].append(
                        {"message_id": prev_msg["id"], "similarity": round(similarity, 3)}
                    )

                    if similarity > max_similarity:
                        max_similarity = similarity
                        most_similar_id = prev_msg["id"]
                except Exception as e:
                    logger.debug(f"Similarity calculation error: {e}")
                    continue

            message_data["max_similarity"] = round(max_similarity, 3)
            message_data["most_similar_to"] = most_similar_id

            # Check if this is a pain point (similarity > 0.6 and has previous messages)
            if max_similarity >= 0.6 and len(self.messages) > 0:
                message_data["is_pain_point"] = True

                # Extract theme using simple keyword matching
                theme = "general"
                for theme_name, keywords in {
                    "workplace_anxiety": ["work", "job", "boss", "colleague", "deadline", "stress", "office"],
                    "relationship_issues": ["relationship", "partner", "marriage", "dating", "love", "breakup"],
                    "self_esteem": ["confidence", "self-worth", "insecure", "doubt", "worthless"],
                    "general_anxiety": ["anxiety", "worry", "nervous", "panic", "fear"],
                    "depression": ["sad", "depressed", "hopeless", "empty", "lonely"],
                    "family_issues": ["family", "parent", "child", "sibling", "mother", "father"],
                }.items():
                    if any(keyword in message.lower() for keyword in keywords):
                        theme = theme_name
                        break
                message_data["theme"] = theme

                # Assess intensity based on content
                if any(word in message.lower() for word in ["constantly", "always", "never", "overwhelming"]):
                    message_data["intensity"] = "high"
                elif any(word in message.lower() for word in ["often", "frequently", "really"]):
                    message_data["intensity"] = "moderate"

                # Record the pain point
                pain_point = {
                    "id": len(self.pain_points) + 1,
                    "message_id": message_data["id"],
                    "theme": theme,
                    "similarity": max_similarity,
                    "intensity": message_data["intensity"],
                    "message": message,
                    "timestamp": datetime.now(),
                    "similar_to_message": most_similar_id,
                }

                self.pain_points.append(pain_point)
                self.analytics["pain_points_detected"] += 1

                logger.info(
                    f"🚨 Pain point detected for user {self.user_id}: {theme} (similarity: {max_similarity:.3f})"
                )

        except Exception as e:
            logger.error(f"Error in pain point analysis: {e}")
            # Continue without pain point detection

        # Update analytics
        self.messages.append(message_data)
        self.analytics["total_interactions"] += 1
        self.analytics["themes"][message_data["theme"]] += 1
        max_similarity_value = message_data.get("max_similarity", 0)
        if isinstance(max_similarity_value, (int, float)) and max_similarity_value > 0:
            self.analytics["similarity_scores"].append(max_similarity_value)
        self.analytics["last_activity"] = datetime.now()

        return message_data

    def get_session_summary(self) -> dict:
        """Get comprehensive session analytics."""
        duration = (datetime.now() - self.session_start).total_seconds() / 60

        return {
            "user_id": self.user_id,
            "session_start": self.session_start.isoformat(),
            "duration_minutes": round(duration, 1),
            "total_messages": len(self.messages),
            "pain_points_detected": len(self.pain_points),
            "pain_point_rate": round((len(self.pain_points) / max(1, len(self.messages))) * 100, 1),
            "avg_similarity": round(
                sum(self.analytics["similarity_scores"]) / max(1, len(self.analytics["similarity_scores"])), 3
            ),
            "dominant_themes": dict(self.analytics["themes"].most_common(3)),
            "recent_pain_points": self.pain_points[-3:],  # Last 3 pain points
            "recommendations": self._generate_recommendations(),
        }

    def _generate_recommendations(self) -> list:
        """Generate therapeutic recommendations based on detected patterns."""
        recommendations = []

        if len(self.pain_points) == 0:
            recommendations.append("✅ No recurring patterns detected - continue supportive dialogue")
        elif len(self.pain_points) <= 2:
            recommendations.append("⚠️ Some repetitive patterns emerging - monitor for development")
        else:
            recommendations.append(f"🚨 {len(self.pain_points)} pain points detected - consider focused intervention")

        # Theme-based recommendations
        if self.analytics["themes"]:
            dominant_theme = self.analytics["themes"].most_common(1)[0][0]

            theme_recommendations = {
                "workplace_anxiety": "💼 Consider CBT techniques for work stress and time management strategies",
                "relationship_issues": "❤️ Explore communication patterns and attachment styles",
                "self_esteem": "💝 Focus on self-compassion exercises and cognitive restructuring",
                "general_anxiety": "🧘 Introduce mindfulness practices and grounding techniques",
                "depression": "🌱 Consider mood tracking and activity scheduling",
                "family_issues": "👨‍👩‍👧‍👦 Explore family dynamics and boundary setting",
            }

            if dominant_theme in theme_recommendations:
                recommendations.append(theme_recommendations[dominant_theme])

        return recommendations[:4]  # Limit to 4 recommendations


def ensure_spacy_model() -> None:
    """Ensure that the spaCy model is available, downloading if necessary."""
    try:
        # Try to load the spaCy model
        spacy.load("en_core_web_sm")
        logger.info("Successfully loaded spaCy model 'en_core_web_sm'")
    except OSError:
        # Model not found, attempt to download
        logger.warning("SpaCy model 'en_core_web_sm' not found. Attempting to download...")
        try:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            logger.info("Successfully downloaded spaCy model 'en_core_web_sm'")
        except Exception as e:
            logger.error("Failed to download spaCy model: %s", e)
            logger.error("Please install it manually with: python -m spacy download en_core_web_sm")
            # Don't raise an exception, application might still work without the model


# Ensure spaCy model is available at startup
ensure_spacy_model()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Move these variables to module level so they're available regardless of how the app runs
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
intelligent_processing_enabled = parse_bool_env("INTELLIGENT_PROCESS_ENABLED", True)
logger.info("Intelligent processing enabled: %s", intelligent_processing_enabled)

# Validate environment variables at module level
if not supabase_url or not supabase_key:
    logger.critical("Error: Please set SUPABASE_URL and SUPABASE_KEY environment variables.")
    # Don't exit here, as it would prevent module import


def get_device_from_config() -> str:
    """Get device based on CUDA_CONFIG settings."""
    from .config import CUDA_CONFIG

    if CUDA_CONFIG.get("force_cpu_fallback", False):
        return "cpu"

    device_selection = CUDA_CONFIG.get("device_selection", "auto")
    if device_selection == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    elif device_selection == "cpu":
        return "cpu"
    elif isinstance(device_selection, str) and device_selection.startswith("cuda"):
        return device_selection if torch.cuda.is_available() else "cpu"
    else:
        return "cuda" if torch.cuda.is_available() else "cpu"


DEVICE = get_device_from_config()

app = Flask(__name__)

CORS(app)


def initialize_app() -> None:
    """Set up the application before the first request."""
    logger.info("Setting up application...")
    logger.info("Welcome to the Therapy AI Assistant! Using model: %s on %s", TEXT_GENERATING_MODEL, DEVICE)

    # Initialize the model manager but don't load the model yet
    # This just sets up the instance which will lazy-load when needed
    get_model_manager(TEXT_GENERATING_MODEL, DEVICE)
    logger.info("Model manager initialized for %s", TEXT_GENERATING_MODEL)


# Call initialize directly
initialize_app()


def should_cleanup_memory() -> bool:
    """Determine if we should clean up GPU memory based on request count and time."""
    from .config import CUDA_CONFIG

    global REQUEST_COUNTER, last_memory_cleanup  # pylint: disable=global-statement

    REQUEST_COUNTER += 1
    current_time = time.time()
    time_since_cleanup = current_time - last_memory_cleanup

    # Clean up if:
    # 1. We've processed enough requests OR
    # 2. It's been long enough since last cleanup OR
    # 3. Randomly with configured probability (to avoid memory fragmentation)
    cleanup_threshold = cast(int, CUDA_CONFIG.get("cleanup_threshold", 10))
    cleanup_time_threshold = cast(float, CUDA_CONFIG.get("cleanup_time_threshold", 300))
    random_cleanup_probability = cast(float, CUDA_CONFIG.get("random_cleanup_probability", 0.1))

    if (
        REQUEST_COUNTER >= cleanup_threshold
        or time_since_cleanup >= cleanup_time_threshold
        or random.random() < random_cleanup_probability
    ):
        REQUEST_COUNTER = 0
        last_memory_cleanup = current_time
        return True

    return False


@app.teardown_request
def teardown_request(_exception: Optional[BaseException] = None) -> None:
    """Clean up after request if needed."""
    if DEVICE == "cuda" and should_cleanup_memory():
        cleanup_memory()


@app.before_request
def before_request() -> Optional[Union[Response, Tuple[Response, int]]]:
    """Initialize DatabaseManager before each request."""
    # Skip for preflight requests
    if request.method == "OPTIONS":
        return None  # Consistent return value

    # Skip for paths that don't need authentication
    if request.path in ["/health", "/memory_status", "/free_memory"]:
        return None  # Consistent return value

    user_id = request.headers.get("X-User-ID")
    if not user_id:
        return jsonify({"error": "User not authenticated"}), 401

    # Store user_id in Flask's 'g' object
    g.user_id = user_id

    if not supabase_url or not supabase_key:
        logger.error("Supabase URL or Key not configured for before_request.")
        # Return 500 error as the server is misconfigured
        return jsonify({"error": "Server configuration error"}), 500

    # Initialize DatabaseManager and store in 'g'
    g.db_manager = DatabaseManager(supabase_url, supabase_key, g.user_id)

    # Create user schema synchronously
    try:
        schema_created = g.db_manager.create_user_schema_sync()
        if not schema_created:
            logger.error("Failed to create user schema for user %s", user_id)
            return jsonify({"error": "Failed to initialize user session"}), 500
    except Exception as e:
        logger.error("Exception during schema creation for user %s: %s", user_id, e, exc_info=True)
        return jsonify({"error": "Failed to initialize user session due to server error"}), 500

    return None  # Consistent return value


@app.route("/health", methods=["GET"])
def health_check() -> Response:
    """Simple health check endpoint."""
    return jsonify({"status": "ok"})


@app.route("/memory_status", methods=["GET"])
def memory_status() -> Response:
    """Report memory usage status (useful for monitoring)."""
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        reserved = torch.cuda.memory_reserved(0) / 1e9  # GB
        allocated = torch.cuda.memory_allocated(0) / 1e9  # GB
        free = total - reserved

        return jsonify(
            {
                "device": torch.cuda.get_device_name(0),
                "total_memory_gb": round(total, 2),
                "reserved_memory_gb": round(reserved, 2),
                "allocated_memory_gb": round(allocated, 2),
                "free_memory_gb": round(free, 2),
                "utilization_percent": round((reserved / total) * 100, 2),
            }
        )

    return jsonify({"device": "CPU", "message": "No CUDA device available"})


@app.route("/chat", methods=["POST"])
def chat() -> Union[Response, Tuple[Response, int]]:
    """Handle chat requests with therapeutic responses and pain point detection."""
    try:
        data = request.json
        if not data or "question" not in data:
            return jsonify({"error": "Missing question parameter"}), 400

        user_id = g.user_id
        question = data["question"]
        enable_monitoring = data.get("enable_pain_point_monitoring", True)  # Default to enabled

        # Log the incoming request
        logger.info("Received chat request from user %s: %s...", user_id, question[:50])

        # Get the model manager instance and then get the generator
        model_manager = get_model_manager(TEXT_GENERATING_MODEL, DEVICE)
        generator = model_manager.get_generator()

        # Create a RAG processor using the retrieved documents
        rag_processor = RAGProcessor(g.db_manager, generator, True)

        try:
            # Generate response
            response = rag_processor.generate_response(
                user_question=question,
                session_id=user_id,
                device=DEVICE,
                question_id=0,
            )

            # Pain point analysis (if enabled)
            pain_point_analysis = None
            if enable_monitoring:
                try:
                    # Get or create pain point session for this user
                    if user_id not in PAIN_POINT_SESSIONS:
                        PAIN_POINT_SESSIONS[user_id] = PainPointSession(user_id, g.db_manager)

                    session = PAIN_POINT_SESSIONS[user_id]

                    # Analyze the message for pain points
                    message_analysis = session.analyze_message(question, response)

                    # Prepare pain point analysis for response
                    pain_point_analysis = {
                        "message_analysis": {
                            "id": message_analysis["id"],
                            "is_pain_point": message_analysis["is_pain_point"],
                            "theme": message_analysis["theme"],
                            "intensity": message_analysis["intensity"],
                            "max_similarity": message_analysis.get("max_similarity", 0),
                            "most_similar_to": message_analysis.get("most_similar_to"),
                        },
                        "session_summary": {
                            "total_messages": len(session.messages),
                            "pain_points_detected": len(session.pain_points),
                            "pain_point_rate": round(
                                (len(session.pain_points) / max(1, len(session.messages))) * 100, 1
                            ),
                            "dominant_themes": dict(session.analytics["themes"].most_common(2)),
                        },
                        "recommendations": (
                            session._generate_recommendations()[:2] if message_analysis["is_pain_point"] else []
                        ),
                    }

                    # Log pain point detection
                    if message_analysis["is_pain_point"]:
                        logger.info(
                            f"🚨 Pain point detected for user {user_id}: {message_analysis['theme']} (similarity: {message_analysis.get('max_similarity', 0):.3f})"
                        )

                except Exception as e:
                    logger.error(f"Error in pain point analysis: {e}")
                    # Continue without pain point analysis
                    pain_point_analysis = {"error": "Pain point analysis temporarily unavailable"}

        finally:
            # Always ensure we free memory for expensive operations
            if DEVICE == "cuda":
                cleanup_memory()

        # Return response with optional pain point analysis
        response_data = {"response": response}
        if pain_point_analysis:
            response_data["pain_point_analysis"] = pain_point_analysis

        return jsonify(response_data)

    except Exception as e:
        logger.error("Error in chat endpoint: %s", e)
        logger.error(traceback.format_exc())
        return (
            jsonify(
                {
                    "response": "I apologise, but I encountered an error. Could you try expressing your concern in a different way?"
                }
            ),
            500,
        )


@app.route("/optimize_vectors", methods=["POST"])
def optimize_vectors() -> Union[Response, Tuple[Response, int]]:
    """Optimize vector operations for the authenticated user."""
    try:
        # Get model manager for embedding generation
        model_manager = get_model_manager(TEXT_GENERATING_MODEL, DEVICE)

        # Use the database manager to optimize vector operations
        # First ensure all interactions have embedding column
        g.db_manager.add_embedding_column_to_interactions()

        # Find interactions without embeddings
        interactions = g.db_manager.get_interactions_without_embeddings()

        if not interactions:
            return jsonify({"message": "No interactions found that need embeddings."})

        total_interactions = len(interactions)
        batch_size = min(10, total_interactions)  # Process in smaller batches
        enriched_count = 0

        logger.info("Starting vector optimization: %d interactions to process", total_interactions)

        try:
            # Process in batches to avoid memory issues
            for i in range(0, total_interactions, batch_size):
                batch = interactions[i : i + batch_size]

                for interaction in batch:
                    try:
                        interaction_id = interaction.get("interaction_id")
                        question = interaction.get("question", "")
                        answer = interaction.get("answer", "")

                        # Generate embedding from combined text - variable removed as unused
                        text_to_embed = f"Question: {question}\nAnswer: {answer}"
                        embedding = model_manager.generate_embedding(text_to_embed)

                        if embedding:
                            # Add embedding to interaction
                            if g.db_manager.add_embedding_to_interaction(interaction_id, embedding):
                                enriched_count += 1
                    except Exception as e:
                        logger.error("Error enriching interaction: %s", e)
                        continue

                # After each batch, clean up memory
                if DEVICE == "cuda":
                    cleanup_memory()

                # Log progress
                logger.info("Processed %d/%d interactions", min(i + batch_size, total_interactions), total_interactions)
        finally:
            # Ensure memory is cleaned up after the operation
            if DEVICE == "cuda":
                cleanup_memory()

        # Ensure vector indexes exist
        g.db_manager.ensure_vector_indexes()

        return jsonify(
            {
                "message": (
                    f"Vector operations optimized. {enriched_count}/{total_interactions} "
                    f"interactions enriched with embeddings."
                )
            }
        )
    except Exception as e:
        logger.error("Error in optimize_vectors endpoint: %s", e)
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route("/free_memory", methods=["POST"])
def free_memory() -> Union[Response, Tuple[Response, int]]:
    """Explicitly free GPU memory on demand."""
    if DEVICE != "cuda":
        return jsonify({"message": "Running on CPU, no GPU memory to free"})

    try:
        cleanup_memory()
        return jsonify({"message": "GPU memory freed successfully"})
    except Exception as e:
        logger.error("Error freeing memory: %s", e)
        return jsonify({"error": f"Failed to free memory: {str(e)}"}), 500


@app.route("/pain_point_monitoring/start", methods=["POST"])
def start_pain_point_monitoring() -> Union[Response, Tuple[Response, int]]:
    """Start pain point monitoring for the current user."""
    try:
        user_id = g.user_id

        # Create or reset pain point session
        PAIN_POINT_SESSIONS[user_id] = PainPointSession(user_id, g.db_manager)

        logger.info(f"🔍 Started pain point monitoring for user {user_id}")

        return jsonify(
            {
                "status": "monitoring_started",
                "user_id": user_id,
                "session_start": PAIN_POINT_SESSIONS[user_id].session_start.isoformat(),
                "message": "Pain point monitoring activated for your therapeutic session",
            }
        )

    except Exception as e:
        logger.error(f"Error starting pain point monitoring: {e}")
        return jsonify({"error": "Failed to start monitoring"}), 500


@app.route("/pain_point_monitoring/status", methods=["GET"])
def get_pain_point_monitoring_status() -> Union[Response, Tuple[Response, int]]:
    """Get current pain point monitoring status and analytics."""
    try:
        user_id = g.user_id

        if user_id not in PAIN_POINT_SESSIONS:
            return jsonify({"monitoring_active": False, "message": "No active monitoring session"})

        session = PAIN_POINT_SESSIONS[user_id]
        summary = session.get_session_summary()

        # Add real-time metrics
        summary.update(
            {
                "monitoring_active": True,
                "last_activity": session.analytics["last_activity"].isoformat(),
                "recent_messages": [
                    {
                        "id": msg["id"],
                        "is_pain_point": msg["is_pain_point"],
                        "theme": msg["theme"],
                        "intensity": msg["intensity"],
                        "max_similarity": msg.get("max_similarity", 0),
                        "timestamp": msg["timestamp"].isoformat(),
                        "preview": (
                            msg["user_message"][:80] + "..." if len(msg["user_message"]) > 80 else msg["user_message"]
                        ),
                    }
                    for msg in session.messages[-5:]  # Last 5 messages
                ],
            }
        )

        return jsonify(summary)

    except Exception as e:
        logger.error(f"Error getting pain point status: {e}")
        return jsonify({"error": "Failed to get monitoring status"}), 500


@app.route("/pain_point_monitoring/dashboard", methods=["GET"])
def get_pain_point_dashboard() -> Union[Response, Tuple[Response, int]]:
    """Get comprehensive dashboard data for pain point visualization."""
    try:
        user_id = g.user_id

        if user_id not in PAIN_POINT_SESSIONS:
            return (
                jsonify({"error": "No active monitoring session", "suggestion": "Start a monitoring session first"}),
                404,
            )

        session = PAIN_POINT_SESSIONS[user_id]

        # Create similarity matrix
        messages = session.messages
        similarity_matrix = []

        for i, msg1 in enumerate(messages):
            row = []
            for j, msg2 in enumerate(messages):
                if i == j:
                    similarity = 1.0
                else:
                    # Find similarity from stored data
                    similarity = 0.0
                    for sim_data in msg1.get("similarity_scores", []):
                        if sim_data["message_id"] == msg2["id"]:
                            similarity = sim_data["similarity"]
                            break

                    # If not found, try reverse lookup
                    if similarity == 0.0:
                        for sim_data in msg2.get("similarity_scores", []):
                            if sim_data["message_id"] == msg1["id"]:
                                similarity = sim_data["similarity"]
                                break

                row.append(round(similarity, 3))
            similarity_matrix.append(row)

        # Prepare dashboard data
        dashboard_data = {
            "session_info": {
                "user_id": user_id,
                "session_start": session.session_start.isoformat(),
                "duration_minutes": round((datetime.now() - session.session_start).total_seconds() / 60, 1),
                "total_messages": len(session.messages),
                "monitoring_active": True,
            },
            "metrics": {
                "pain_points_detected": len(session.pain_points),
                "pain_point_rate": round((len(session.pain_points) / max(1, len(session.messages))) * 100, 1),
                "avg_similarity": round(
                    sum(session.analytics["similarity_scores"]) / max(1, len(session.analytics["similarity_scores"])), 3
                ),
                "theme_diversity": len(session.analytics["themes"]),
            },
            "timeline": [
                {
                    "message_id": msg["id"],
                    "timestamp": msg["timestamp"].isoformat(),
                    "is_pain_point": msg["is_pain_point"],
                    "theme": msg["theme"],
                    "intensity": msg["intensity"],
                    "similarity": msg.get("max_similarity", 0),
                    "preview": (
                        msg["user_message"][:60] + "..." if len(msg["user_message"]) > 60 else msg["user_message"]
                    ),
                }
                for msg in session.messages
            ],
            "pain_points": [
                {
                    "id": pp["id"],
                    "message_id": pp["message_id"],
                    "theme": pp["theme"],
                    "similarity": pp["similarity"],
                    "intensity": pp["intensity"],
                    "timestamp": pp["timestamp"].isoformat(),
                    "message_preview": pp["message"][:100] + "..." if len(pp["message"]) > 100 else pp["message"],
                    "similar_to_message": pp.get("similar_to_message"),
                }
                for pp in session.pain_points
            ],
            "themes": dict(session.analytics["themes"]),
            "similarity_matrix": similarity_matrix,
            "recommendations": session._generate_recommendations(),
            "last_updated": datetime.now().isoformat(),
        }

        return jsonify(dashboard_data)

    except Exception as e:
        logger.error(f"Error getting pain point dashboard: {e}")
        return jsonify({"error": "Failed to get dashboard data"}), 500


@app.route("/pain_point_monitoring/stop", methods=["POST"])
def stop_pain_point_monitoring() -> Union[Response, Tuple[Response, int]]:
    """Stop pain point monitoring and return final summary."""
    try:
        user_id = g.user_id

        if user_id not in PAIN_POINT_SESSIONS:
            return jsonify({"message": "No active monitoring session to stop"})

        session = PAIN_POINT_SESSIONS[user_id]
        final_summary = session.get_session_summary()

        # Archive the session (you could store this in database)
        logger.info(
            f"📊 Stopping pain point monitoring for user {user_id}. Final summary: {len(session.pain_points)} pain points detected"
        )

        # Remove from active sessions
        del PAIN_POINT_SESSIONS[user_id]

        return jsonify(
            {
                "status": "monitoring_stopped",
                "final_summary": final_summary,
                "message": "Pain point monitoring session completed",
            }
        )

    except Exception as e:
        logger.error(f"Error stopping pain point monitoring: {e}")
        return jsonify({"error": "Failed to stop monitoring"}), 500


if __name__ == "__main__":
    logger = get_package_logger(__name__)
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")

    app.run(debug=False, host="0.0.0.0", port=5000)
