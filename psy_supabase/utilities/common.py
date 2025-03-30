"""
Common utility functions shared across modules.
This module should NOT import from model_manager or text_generator.
"""

import os
import torch
import logging
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)

def get_project_root():
    """Return the absolute path to the project root directory."""
    return Path(__file__).parent.parent.parent.absolute()

def ensure_dir_exists(directory_path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        logger.info(f"Created directory: {directory_path}")
    return directory_path

def get_models_dir():
    """Get the models directory path."""
    root_dir = get_project_root()
    models_dir = os.path.join(root_dir, "models")
    return ensure_dir_exists(models_dir)

def is_github_actions():
    """Check if running in GitHub Actions environment."""
    return os.environ.get('GITHUB_ACTIONS') == 'true'

def load_toxicity_model(logger_instance=None):
    """
    Load toxicity detection model with local caching support.

    This shared function is used by both TextGenerator and ModelManager
    to prevent code duplication.

    Args:
        logger_instance: Logger instance to use (falls back to module logger)

    Returns:
        tuple: (model, tokenizer) for toxicity detection
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    # Use provided logger or fall back to module logger
    log = logger_instance or logger

    # Define toxicity model name
    toxicity_model_name = "facebook/roberta-hate-speech-dynabench-r4-target"

    # Get local model path
    model_folder = toxicity_model_name.split('/')[-1]
    models_dir = get_models_dir()
    local_path = os.path.join(models_dir, model_folder)

    # Create models dir if it doesn't exist
    ensure_dir_exists(models_dir)

    # Check if model exists locally
    if os.path.exists(local_path) and os.path.isdir(local_path) and len(os.listdir(local_path)) > 0:
        # Use local model
        log.info(f"Loading toxicity model from local path: {local_path}")
        toxicity_tokenizer = AutoTokenizer.from_pretrained(local_path)
        toxicity_model = AutoModelForSequenceClassification.from_pretrained(
            local_path,
            torch_dtype=torch.float32
        )
    else:
        # Download model and save locally
        log.info(f"Downloading toxicity model to {local_path}")
        ensure_dir_exists(local_path)

        # Download and save tokenizer
        toxicity_tokenizer = AutoTokenizer.from_pretrained(toxicity_model_name)
        toxicity_tokenizer.save_pretrained(local_path)

        # Download and save model
        toxicity_model = AutoModelForSequenceClassification.from_pretrained(
            toxicity_model_name,
            torch_dtype=torch.float32
        )
        toxicity_model.save_pretrained(local_path)
        log.info(f"Toxicity model saved to {local_path}")

    # Always keep toxicity model on CPU for efficiency
    toxicity_model = toxicity_model.to("cpu")
    toxicity_model.eval()

    return toxicity_model, toxicity_tokenizer
