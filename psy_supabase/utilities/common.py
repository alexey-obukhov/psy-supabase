"""
Common utility functions shared across modules.
This module should NOT import from model_manager or text_generator.
"""

import os
import traceback
from logging import Logger
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from prismalog.log import get_logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Configure logger
logger = get_logger(__name__)


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).parent.parent.parent.absolute()


def ensure_dir_exists(directory_path: str) -> str:
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        logger.info("Created directory: %s", directory_path)
    return directory_path


def get_models_dir() -> str:
    """Get the models directory path."""
    root_dir = get_project_root()
    models_dir = os.path.join(root_dir, "models")
    return ensure_dir_exists(models_dir)


def is_github_actions() -> bool:
    """Check if running in GitHub Actions environment."""
    return os.environ.get("GITHUB_ACTIONS") == "true"


def load_toxicity_model(
    logger_instance: Optional[Logger] = None,
) -> Tuple[Optional[AutoModelForSequenceClassification], Optional[AutoTokenizer]]:
    """
    Load toxicity detection model with local caching support.

    This shared function is used by both TextGenerator and ModelManager
    to prevent code duplication.

    Args:
        logger_instance: Logger instance to use (falls back to module logger)

    Returns:
        tuple: (model, tokenizer) where tokenizer can be AutoTokenizer or AutoTokenizer
    """

    log = logger_instance or logger
    toxicity_model_name = "facebook/roberta-hate-speech-dynabench-r4-target"
    model_folder = toxicity_model_name.split("/")[-1]
    models_dir = get_models_dir()
    local_path = os.path.join(models_dir, model_folder)
    ensure_dir_exists(models_dir)

    toxicity_model: Optional[AutoModelForSequenceClassification] = None
    toxicity_tokenizer: Optional[AutoTokenizer] = None

    try:
        if os.path.exists(local_path) and os.path.isdir(local_path) and len(os.listdir(local_path)) > 0:
            log.info("Loading toxicity model from local path: %s", local_path)
            toxicity_tokenizer = AutoTokenizer.from_pretrained(local_path)
            toxicity_model = AutoModelForSequenceClassification.from_pretrained(local_path, torch_dtype=torch.float32)
        else:
            log.info("Downloading toxicity model '%s' to %s", toxicity_model_name, local_path)
            ensure_dir_exists(local_path)
            toxicity_tokenizer = AutoTokenizer.from_pretrained(toxicity_model_name)
            toxicity_tokenizer.save_pretrained(local_path)
            toxicity_model = AutoModelForSequenceClassification.from_pretrained(
                toxicity_model_name, torch_dtype=torch.float32
            )
            toxicity_model.save_pretrained(local_path)
            log.info("Toxicity model saved to %s", local_path)

        # Explicit Type Check for Model
        if toxicity_model is None or not isinstance(toxicity_model, AutoModelForSequenceClassification):
            log.error(
                "Loaded toxicity_model is None or not a AutoModelForSequenceClassification instance. Type: %s",
                type(toxicity_model),
            )
            return None, None

        # Modified Type Check for Tokenizer
        if toxicity_tokenizer is None or not isinstance(toxicity_tokenizer, AutoTokenizer):
            log.error(
                "Loaded toxicity_tokenizer is None or not a AutoTokenizer/AutoTokenizer instance. Type: %s",
                type(toxicity_tokenizer),
            )
            try:
                log.debug("Inheritance chain for tokenizer: %s", type(toxicity_tokenizer).__mro__)
            except Exception:
                pass
            return None, None

        # Move to CPU and set to eval mode
        toxicity_model = toxicity_model.to("cpu")
        toxicity_model.eval()
        log.info("Toxicity model and tokenizer loaded successfully and moved to CPU.")

    except Exception as e:
        log.error("Failed during toxicity model loading/downloading for '%s': %s", toxicity_model_name, e)
        log.error(traceback.format_exc())
        return None, None

    # Return the loaded model and tokenizer directly
    return toxicity_model, toxicity_tokenizer
