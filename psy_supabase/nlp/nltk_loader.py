"""
NLTK resources loader module.

This module handles the downloading and initialization of NLTK resources
and text2emotion library required for natural language processing tasks.
"""

from school_logging.log import ColoredLogger

# Initialize logger
logger = ColoredLogger(__name__)

try:
    import nltk
    import text2emotion as te
except ImportError as e:
    logger.error("Required NLP libraries not installed: %s", e)
    logger.error("Please install with: pip install nltk text2emotion")
    raise

def download_nltk_resources():
    """Download required NLTK resources if they don't exist."""
    resources = ['punkt', 'stopwords', 'wordnet']

    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
            logger.info(f"NLTK resource '{resource}' already downloaded")
        except LookupError:
            logger.info(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, quiet=True)

def is_text2emotion_ready():
    """Check if text2emotion library is properly installed."""
    try:
        # Simple test to ensure text2emotion works
        test = te.get_emotion("This is a test")
        return isinstance(test, dict)
    except Exception as e:
        logger.error(f"text2emotion error: {e}")
        return False

# Initialize resources when module is imported
download_nltk_resources()
