"""
NLP utilities for the psy_supabase package.
Provides a SpacyModelManager singleton for efficient spaCy model management.
"""
import logging
import spacy
from typing import Optional


logger = logging.getLogger(__name__)

class SpacyModelManager:
    """
    Singleton manager for spaCy models to ensure a single instance is loaded and shared.
    """
    _instance = None
    _model = None
    
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
        if self._model is None:
            try:
                self._model = spacy.load(model_name)
                logger.info(f"Loaded spaCy model '{model_name}'")
            except OSError:
                logger.warning(f"Model '{model_name}' not found. Attempting to download...")
                try:
                    from spacy.cli import download
                    download(model_name)
                    self._model = spacy.load(model_name)
                    logger.info(f"Successfully downloaded and loaded model '{model_name}'")
                except Exception as e:
                    logger.error(f"Failed to download spaCy model: {e}")
                    logger.error(f"Please run: python -m spacy download {model_name}")
                    return None
        return self._model

# Create a convenience function to access the model
def get_spacy_model(model_name: str = "en_core_web_sm"):
    """Get the shared spaCy model instance."""
    return SpacyModelManager.get_instance().get_model(model_name)
