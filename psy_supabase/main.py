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
from typing import Optional, Tuple, Union

import spacy
import torch

# Third-party imports
from flask import Flask, Response, g, jsonify, request
from typeguard import install_import_hook

from psy_supabase.config import TEXT_GENERATING_MODEL
from psy_supabase.core.database import DatabaseManager
from psy_supabase.core.model_manager import get_model_manager
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
CLEANUP_THRESHOLD = 10  # Clean up after 10 requests
CLEANUP_TIME_THRESHOLD = 300  # Clean up after 5 minutes


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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__)


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
    global REQUEST_COUNTER, last_memory_cleanup  # pylint: disable=global-statement

    REQUEST_COUNTER += 1
    current_time = time.time()
    time_since_cleanup = current_time - last_memory_cleanup

    # Clean up if:
    # 1. We've processed enough requests OR
    # 2. It's been long enough since last cleanup OR
    # 3. Randomly with low probability (to avoid memory fragmentation)
    if (
        REQUEST_COUNTER >= CLEANUP_THRESHOLD or time_since_cleanup >= CLEANUP_TIME_THRESHOLD or random.random() < 0.05
    ):  # 5% chance to clean up

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
    """Handle chat requests with therapeutic responses."""
    try:
        data = request.json
        if not data or "question" not in data:
            return jsonify({"error": "Missing question parameter"}), 400

        user_id = g.user_id  # request.headers.get('X-User-ID', 'default_user')
        question = data["question"]

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

        finally:
            # Always ensure we free memory for expensive operations
            # Chat is the most memory-intensive operation, so we clean up explicitly
            if DEVICE == "cuda":
                cleanup_memory()

        return jsonify({"response": response})
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


if __name__ == "__main__":
    logger = get_package_logger(__name__)
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")

    app.run(debug=False, port=5008)
