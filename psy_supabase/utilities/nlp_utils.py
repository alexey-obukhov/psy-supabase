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
import spacy
from typing import Optional, Dict, List, Any
from school_logging.log import ColoredLogger

# Setup logging
logger = ColoredLogger(__name__)

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

class SpacyModelManager:
    """
    Singleton manager for spaCy models to ensure a single instance is loaded and shared.
    """
    _instance = None
    _models = {}  # Support multiple models (e.g., en_core_web_sm, en_core_web_md)

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of SpacyModelManager."""
        if cls._instance is None:
            cls._instance = SpacyModelManager()
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
                self._models[model_name] = spacy.load(model_name)
                logger.info(f"Loaded spaCy model '{model_name}'")
            except OSError:
                logger.warning(f"Model '{model_name}' not found. Attempting to download...")
                try:
                    from spacy.cli import download
                    download(model_name)
                    self._models[model_name] = spacy.load(model_name)
                    logger.info(f"Successfully downloaded and loaded model '{model_name}'")
                except Exception as e:
                    logger.error(f"Failed to download spaCy model: {e}")
                    logger.error(f"Please run: python -m spacy download {model_name}")
                    return None
        return self._models[model_name]

# NLTK Resource Management
def download_nltk_resources():
    """Download required NLTK resources if they don't exist."""
    if not NLTK_AVAILABLE:
        logger.warning("NLTK not installed, skipping resource download")
        return False

    resources = ['punkt', 'stopwords', 'wordnet']

    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
            logger.debug(f"NLTK resource '{resource}' already downloaded")
        except LookupError:
            logger.info(f"Downloading NLTK resource: {resource}")
            try:
                nltk.download(resource, quiet=True)
            except Exception as e:
                logger.error(f"Failed to download NLTK resource '{resource}': {e}")
                return False

    return True

def is_text2emotion_ready():
    """Check if text2emotion library is properly installed."""
    if not TEXT2EMOTION_AVAILABLE:
        return False

    try:
        # Simple test to ensure text2emotion works
        test = te.get_emotion("This is a test")
        return isinstance(test, dict)
    except Exception as e:
        logger.error(f"text2emotion error: {e}")
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
        return te.get_emotion(text)
    except Exception as e:
        logger.error(f"Error analysing emotion: {e}")
        return {}

# Convenience functions
def get_spacy_model(model_name: str = "en_core_web_sm"):
    """Get the shared spaCy model instance."""
    return SpacyModelManager.get_instance().get_model(model_name)

def extract_entities(text: str, model_name: str = "en_core_web_sm") -> List[Dict[str, Any]]:
    """
    Extract named entities from text.

    Args:
        text: Text to analyse
        model_name: Name of spaCy model to use

    Returns:
        List of entity dictionaries with text, type, and position
    """
    model = get_spacy_model(model_name)
    if not model or not text:
        return []

    doc = model(text)

    return [
        {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        }
        for ent in doc.ents
    ]

def tokenize_text(text: str, model_name: str = "en_core_web_sm") -> List[str]:
    """
    Tokenize text using spaCy.

    Args:
        text: Text to tokenize
        model_name: Name of spaCy model to use

    Returns:
        List of tokens
    """
    model = get_spacy_model(model_name)
    if not model or not text:
        return []

    doc = model(text)
    return [token.text for token in doc]

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

    # Fix common Unicode issues
    import html
    text = html.unescape(text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»""'']))", "", text)
    text = re.sub(r"\u2019", "'", text)
    text = re.sub(r"\u2014", "-", text)
    text = re.sub(r"\u201c", '"', text)
    text = re.sub(r"\u201d", '"', text)
    text = re.sub(r"\u2026", "...", text)

    # Other cleaning as needed
    return text

# Initialize resources when module is imported
if os.environ.get('INITIALIZE_NLP_RESOURCES', 'true').lower() == 'true':
    download_nltk_resources()
