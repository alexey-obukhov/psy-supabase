"""
NLP utilities for the psy_supabase package.
Provides unified management of language models and resources.

Features:
- SpacyModelManager: Singleton for efficient spaCy model management
- NLTK resource management
- text2emotion compatibility checking
- Utility functions for text processing
"""

import os
import re
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

import spacy
from prismalog.log import get_logger

# Setup logging
logger = get_logger(__name__)

# Import NLTK with error handling
try:
    import nltk

    NLTK_AVAILABLE = True
except ImportError:
    logger.warning("NLTK not available. Some features will be disabled.")
    NLTK_AVAILABLE = False

# Import text2emotion with error handling
try:
    import text2emotion as te

    TEXT2EMOTION_AVAILABLE = True
except ImportError:
    logger.warning("text2emotion not available. Emotional analysis will be disabled.")
    TEXT2EMOTION_AVAILABLE = False

# Define a type variable for the singleton pattern
TSpacyModelManager = TypeVar("TSpacyModelManager", bound="SpacyModelManager")


class SpacyModelManager(Generic[TSpacyModelManager]):
    """
    Singleton manager for spaCy models to ensure a single instance is loaded and shared.
    """

    _instance: Optional[TSpacyModelManager] = None
    _models: Dict[str, spacy.language.Language] = {}

    @classmethod
    def get_instance(cls: Type[TSpacyModelManager]) -> TSpacyModelManager:
        """Get the singleton instance of SpacyModelManager."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_model(self, model_name: str = "en_core_web_sm") -> Optional[spacy.language.Language]:
        """
        Get the requested spaCy model, loading it if necessary.

        Args:
            model_name: Name of the spaCy model to load

        Returns:
            The loaded spaCy model or None if it couldn't be loaded
        """
        if model_name not in self._models:
            try:
                loaded_model: spacy.language.Language = spacy.load(model_name)
                self._models[model_name] = loaded_model
                logger.info("Loaded spaCy model '%s'", model_name)
            except OSError:
                logger.warning("Model '%s' not found. Attempting to download...", model_name)
                try:
                    from spacy.cli import download

                    download(model_name)
                    downloaded_model: spacy.language.Language = spacy.load(model_name)
                    self._models[model_name] = downloaded_model
                    logger.info("Successfully downloaded and loaded model '%s'", model_name)
                except Exception as e:
                    logger.error("Failed to download spaCy model: %s", e)
                    logger.error("Please run: python -m spacy download %s", model_name)
                    return None
        return self._models.get(model_name)


# NLTK Resource Management
def download_nltk_resources() -> bool:
    """Download required NLTK resources if they don't exist."""
    if not NLTK_AVAILABLE:
        logger.warning("NLTK not installed, skipping resource download")
        return False

    resources: List[str] = ["punkt", "stopwords", "wordnet"]

    for resource in resources:
        try:
            # Example: nltk.data.find(f"tokenizers/{resource}") for punkt
            # Example: nltk.data.find(f"corpora/{resource}") for stopwords/wordnet
            nltk.data.find(f"corpora/{resource}")  # Assuming corpora for wordnet/stopwords
            if resource == "punkt":
                nltk.data.find(f"tokenizers/{resource}")  # Specific check for punkt
            logger.debug("NLTK resource '%s' already downloaded", resource)
        except LookupError:
            logger.info("Downloading NLTK resource: %s", resource)
            try:
                nltk.download(resource, quiet=True)
            except Exception as e:
                logger.error("Failed to download NLTK resource '%s': %s", resource, e)
                return False

    return True


def is_text2emotion_ready() -> bool:
    """Check if text2emotion library is properly installed."""
    if not TEXT2EMOTION_AVAILABLE:
        return False

    try:
        # Simple test to ensure text2emotion works
        test: Dict[str, float] = te.get_emotion("This is a test")
        return isinstance(test, dict)
    except Exception as e:
        logger.error("text2emotion error: %s", e)
        return False


def analyze_emotion(text: str) -> Dict[str, float]:
    """
    Analyze emotions in text using text2emotion.

    Args:
        text: Text to analyse

    Returns:
        Dictionary of emotion scores or empty dict if analysis fails
    """
    if not TEXT2EMOTION_AVAILABLE or not text:
        return {}

    try:
        emotions: Dict[str, float] = te.get_emotion(text)
        return emotions
    except Exception as e:
        logger.error("Error analysing emotion: %s", e)
        return {}


# Convenience functions
def get_spacy_model(model_name: str = "en_core_web_sm") -> Optional[spacy.language.Language]:
    """Get the shared spaCy model instance."""
    manager: SpacyModelManager = SpacyModelManager.get_instance()
    return manager.get_model(model_name)


def extract_entities(text: str, model_name: str = "en_core_web_sm") -> List[Dict[str, Any]]:
    """
    Extract named entities from text.

    Args:
        text: Text to analyse
        model_name: Name of spaCy model to use

    Returns:
        List of entity dictionaries with text, type, and position
    """
    model: Optional[spacy.language.Language] = get_spacy_model(model_name)
    if not model or not text:
        return []

    doc: spacy.tokens.Doc = model(text)

    entities: List[Dict[str, Any]] = [
        {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char} for ent in doc.ents
    ]
    return entities


def tokenize_text(text: str, model_name: str = "en_core_web_sm") -> List[str]:
    """
    Tokenize text using spaCy.

    Args:
        text: Text to tokenize
        model_name: Name of spaCy model to use

    Returns:
        List of tokens
    """
    model: Optional[spacy.language.Language] = get_spacy_model(model_name)
    if not model or not text:
        return []

    doc: spacy.tokens.Doc = model(text)
    tokens: List[str] = [token.text for token in doc]
    return tokens


def clean_text(text: str) -> str:
    """
    Clean and normalize text.

    Args:
        text: Text to clean

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    import html

    cleaned_text: str = html.unescape(text)
    cleaned_text = re.sub(r"<.*?>", "", cleaned_text)
    # URL pattern (consider simplifying if too complex or causing issues)
    url_pattern = (
        r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»"
        "'']))"
    )
    cleaned_text = re.sub(url_pattern, "", cleaned_text)
    cleaned_text = re.sub(r"\u2019", "'", cleaned_text)
    cleaned_text = re.sub(r"\u2014", "-", cleaned_text)
    cleaned_text = re.sub(r"\u201c", '"', cleaned_text)
    cleaned_text = re.sub(r"\u201d", '"', cleaned_text)
    cleaned_text = re.sub(r"\u2026", "...", cleaned_text)

    # Other cleaning as needed
    return cleaned_text


# Initialize resources when module is imported
if os.environ.get("INITIALIZE_NLP_RESOURCES", "true").lower() == "true":
    download_nltk_resources()
