"""
Model Manager Module

This module provides a `ModelManager` class to handle the loading, caching, and management of machine learning models.
It supports both text generation and toxicity detection models, as well as embedding generation using sentence-transformers.

The module also includes utility functions and adapter classes for compatibility with other interfaces.
"""

import gc
import os
import traceback
from typing import List, Optional, Dict, ClassVar
from typeguard import typechecked
import torch
import torch.cuda
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from school_logging.log import ColoredLogger
from psy_supabase.utilities.common import get_models_dir, ensure_dir_exists
from psy_supabase.core.text_generator import TextGenerator

# Create a model manager class to handle loading/unloading
class ModelManager:
    """
    ModelManager Class

    This class manages the loading, caching, and unloading of machine learning models. It supports:
    - Text generation models (e.g., causal language models)
    - Toxicity detection models
    - Embedding generation using sentence-transformers or main models

    The class ensures that models are downloaded and cached locally for reuse, and it provides methods to free memory
    and switch devices (e.g., between CPU and GPU).
    """

    # Class variable to store instances (no global variables)
    _instances: ClassVar[Dict[str, 'ModelManager']] = {}
    # Add models directory path
    MODELS_DIR = get_models_dir()

    @classmethod
    @typechecked
    def get_instance(cls, model_name: str = "microsoft/phi-1_5", device: Optional[str] = None, quantize: bool = False) -> 'ModelManager':
        """
        Get or create a ModelManager instance.

        Args:
            model_name: Model name to use
            device: Device to use (None for auto-detection)
            quantize: Whether to use 8-bit quantization for large models

        Returns:
            ModelManager instance
        """
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Use model_name as key (include quantization setting)
        instance_key = f"{model_name}_{device}_quant={quantize}"

        # Create if doesn't exist
        if instance_key not in cls._instances:
            cls._instances[instance_key] = cls(model_name, device, quantize)

        # Get the instance
        instance = cls._instances[instance_key]

        # Update device if different
        if instance.preferred_device != device:
            instance.preferred_device = device

        return instance

    def __init__(self, model_name, device="cpu", quantize=False):
        """
        Initialize the ModelManager instance.

        Args:
            model_name: Name of the model to manage
            device: Device to use (e.g., "cpu" or "cuda")
            quantize: Whether to use quantization for the model
        """
        self.model_name = model_name
        self.preferred_device = device
        self.quantize = quantize  # New parameter
        self.generator = None
        self.embedding_model = None
        self.toxicity_model = None
        self.sentence_transformer = None
        self.current_device = None
        self.logger = ColoredLogger(__name__)

        # Preload toxicity model
        self.get_toxicity_model()

        self.logger.info(f"ModelManager initialized with model: {model_name}, device: {device}, quantize: {quantize}")

    def get_local_model_path(self):
        """Get the local path for the model"""
        # Use just the model name without organization prefix for folder
        model_folder = self.model_name.split('/')[-1] if '/' in self.model_name else self.model_name
        return os.path.join(self.MODELS_DIR, model_folder)

    def is_model_downloaded(self):
        """Check if the model is already downloaded locally"""
        model_path = self.get_local_model_path()

        # Check if folder exists and is not empty
        if os.path.exists(model_path) and os.path.isdir(model_path):
            # Check if directory has any files
            return len(os.listdir(model_path)) > 0

        return False

    def get_generator(self):
        """Get or initialize text generator with local model caching."""
        # Initialize on first use
        if self.generator is None:
            # Create models dir if it doesn't exist
            os.makedirs(self.MODELS_DIR, exist_ok=True)

            # Get local model path
            local_path = self.get_local_model_path()

            # Check if model exists locally
            if self.is_model_downloaded():
                # Use local model
                self.logger.info(f"Loading model from local path: {local_path}")
                self.generator = TextGenerator(local_path, self.preferred_device, quantize=self.quantize)
            else:
                # We need to download the model regardless of quantization
                self.logger.info(f"Model not found locally. Downloading {self.model_name} to {local_path}")

                # Create directory
                os.makedirs(local_path, exist_ok=True)

                # Download and save tokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                tokenizer.save_pretrained(local_path)
                self.logger.info(f"Tokenizer saved to {local_path}")

                try:
                    # Download and save full model (without quantization)
                    self.logger.info(f"Downloading full model weights to {local_path}")
                    model = AutoModelForCausalLM.from_pretrained(self.model_name)
                    model.save_pretrained(local_path)
                    self.logger.info(f"Full model saved to {local_path}")

                    # Now create the generator, using quantization if requested
                    self.generator = TextGenerator(local_path, self.preferred_device, quantize=self.quantize)
                except Exception as e:
                    self.logger.error(f"Error downloading full model: {e}")
                    # If full model download fails, try direct initialization
                    self.generator = TextGenerator(self.model_name, self.preferred_device, quantize=self.quantize)

            self.current_device = self.preferred_device
        else:
            # Make sure model is fully on the right device
            self.logger.info(f"Moving existing model to {self.preferred_device}")
            if hasattr(self.generator, 'model'):
                # Only try to move model if not using device_map='auto'
                if not (self.quantize and hasattr(self.generator, 'using_device_map') and self.generator.using_device_map):
                    self.generator.model = self.generator.model.to(self.preferred_device)
                self.generator.device = self.preferred_device
                self.current_device = self.preferred_device
        return self.generator

    @typechecked
    def get_toxicity_model(self, toxicity_model_name="facebook/roberta-hate-speech-dynabench-r4-target"):
        """
        Get or initialize the toxicity detection model with local model caching.

        Returns:
            Tuple of (toxicity_model, toxicity_tokenizer)
        """
        if not hasattr(self, 'toxicity_model') or self.toxicity_model is None:

            # Get local model path
            model_folder = toxicity_model_name.rsplit('/', maxsplit=1)[-1]
            local_path = os.path.join(self.MODELS_DIR, model_folder)

            # Create models dir if it doesn't exist
            ensure_dir_exists(self.MODELS_DIR)

            # Check if model exists locally
            if os.path.exists(local_path) and os.path.isdir(local_path) and len(os.listdir(local_path)) > 0:
                # Use local model
                self.logger.info(f"Loading toxicity model from local path: {local_path}")
                self.toxicity_tokenizer = AutoTokenizer.from_pretrained(local_path)
                self.toxicity_model = AutoModelForSequenceClassification.from_pretrained(
                    local_path,
                    torch_dtype=torch.float32
                )
            else:
                # Download model and save locally
                self.logger.info(f"Downloading toxicity model to {local_path}")
                os.makedirs(local_path, exist_ok=True)

                # Download and save tokenizer
                self.toxicity_tokenizer = AutoTokenizer.from_pretrained(toxicity_model_name)
                self.toxicity_tokenizer.save_pretrained(local_path)

                # Download and save model
                self.toxicity_model = AutoModelForSequenceClassification.from_pretrained(
                    toxicity_model_name,
                    torch_dtype=torch.float32
                )
                self.toxicity_model.save_pretrained(local_path)
                self.logger.info(f"Toxicity model saved to {local_path}")

            # Always keep toxicity model on CPU for efficiency
            self.toxicity_model = self.toxicity_model.to("cpu")
            self.toxicity_model.eval()

        return self.toxicity_model, self.toxicity_tokenizer

    def free_memory(self):
        """Free up GPU memory by moving model to CPU and releasing CUDA memory."""
        if self.generator and hasattr(self.generator, 'model'):
            self.logger.info("Freeing GPU memory - moving model to CPU")
            # Explicitly move model to CPU
            self.generator.model = self.generator.model.to('cpu')
            self.generator.device = 'cpu'
            self.current_device = 'cpu'

            # Release CUDA memory
            torch.cuda.empty_cache()
            # Run garbage collector
            gc.collect()
            self.logger.info("GPU memory freed")

        # Also free the embedding model if it exists
        if self.embedding_model is not None:
            self.logger.info("Freeing embedding model memory")
            self.embedding_model = self.embedding_model.to('cpu')
            torch.cuda.empty_cache()
            gc.collect()

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding vector for text using the main model.
        Uses the loaded model's hidden states for embedding generation.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the embedding vector or None if failed
        """
        if not text or not text.strip():
            # Return zero vector of appropriate dimension
            self.logger.warning("Empty text provided for embedding")
            return [0.0] * 2048  # Default dimension, will be adjusted for actual models

        try:
            # Use TextGenerator's embedding function if it exists
            generator = self.get_generator()
            if hasattr(generator, 'get_embedding'):
                self.logger.info("Using TextGenerator for embedding generation")
                embedding_tensor = generator.get_embedding(text)
                if embedding_tensor is not None:
                    return embedding_tensor.squeeze().cpu().tolist()

            # Fallback: Generate embedding directly using the model
            self.logger.info("Generating embedding directly from model hidden states")
            if hasattr(generator, 'model') and hasattr(generator, 'tokenizer'):
                inputs = generator.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(generator.device)

                with torch.no_grad():
                    outputs = generator.model(**inputs, output_hidden_states=True)
                    # Use last hidden state
                    hidden_states = outputs.hidden_states[-1]
                    # Mean pooling
                    embedding = hidden_states.mean(dim=1)
                    return embedding.squeeze().cpu().tolist()

            # If we got here, we couldn't use the generator for embeddings
            raise ValueError("Model doesn't support embedding generation")

        except Exception as e:
            self.logger.error(f"Error generating embedding with main model: {e}")
            self.logger.error(traceback.format_exc())

            # Try with sentence transformer as fallback
            return self._generate_embedding_with_sentence_transformer(text)

    def _generate_embedding_with_sentence_transformer(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding using sentence-transformers as a fallback.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the embedding vector or None if failed
        """
        try:
            # Initialize sentence transformer if needed
            if self.sentence_transformer is None:
                self.logger.info("Initializing SentenceTransformer for fallback embeddings")
                try:
                    # Use a reliable, small model for embeddings
                    model_name = "sentence-transformers/all-mpnet-base-v2"
                    self.sentence_transformer = SentenceTransformer(model_name)
                    if self.preferred_device == "cuda" and torch.cuda.is_available():
                        self.sentence_transformer = self.sentence_transformer.to(self.preferred_device)
                except ImportError:
                    self.logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
                    return None

            # Generate embedding
            embedding = self.sentence_transformer.encode(text)
            return embedding.tolist()

        except Exception as e:
            self.logger.error(f"Error generating embedding with sentence transformer: {e}")
            self.logger.error(traceback.format_exc())
            return None

    def batch_generate_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for a batch of texts.
        For large batches, uses sentence-transformers which is more efficient.
        For small batches, uses the main model for better quality.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # For larger batches, use sentence-transformers (more efficient)
        if len(texts) > 5:
            try:
                # Initialize sentence transformer if needed
                if self.sentence_transformer is None:
                    # Get path to models directory
                    os.makedirs(self.MODELS_DIR, exist_ok=True)

                    # Define the model name and local path
                    st_model_name = "all-mpnet-base-v2"
                    local_path = os.path.join(self.MODELS_DIR, "sentence-transformers_" + st_model_name)

                    # Check if model exists locally
                    if os.path.exists(local_path) and os.path.isdir(local_path) and len(os.listdir(local_path)) > 0:
                        # Use local model
                        self.logger.info(f"Loading SentenceTransformer from local path: {local_path}")
                        self.sentence_transformer = SentenceTransformer(local_path)
                    else:
                        # Download model and save locally
                        self.logger.info(f"Downloading SentenceTransformer to {local_path}")
                        os.makedirs(local_path, exist_ok=True)

                        # Download and save model
                        self.sentence_transformer = SentenceTransformer("sentence-transformers/" + st_model_name)
                        self.sentence_transformer.save(local_path)
                        self.logger.info(f"SentenceTransformer saved to {local_path}")

                    # Move to the right device
                    if self.preferred_device == "cuda" and torch.cuda.is_available():
                        self.sentence_transformer = self.sentence_transformer.to(self.preferred_device)

                # Get embedding dimension
                embedding_dim = self.sentence_transformer.get_sentence_embedding_dimension()

                # Process texts (replace empty with spaces to avoid errors)
                processed_texts = [text if text and text.strip() else " " for text in texts]

                # Generate embeddings in batch
                embeddings = self.sentence_transformer.encode(processed_texts)

                # Format results
                results = []
                for i, text in enumerate(texts):
                    if not text or not text.strip():
                        results.append([0.0] * embedding_dim)  # Zero vector
                    else:
                        results.append(embeddings[i].tolist())

                return results

            except Exception as e:
                self.logger.error(f"Error in batch embedding: {e}")
                self.logger.error(traceback.format_exc())

        # For smaller batches or if sentence-transformers failed, use the main model
        return [self.generate_embedding(text) for text in texts]


@typechecked
def get_model_manager(model_name: str = "microsoft/phi-1_5", device: Optional[str] = None, quantize: bool = False) -> ModelManager:
    """
    Get a ModelManager instance.
    
    Args:
        model_name: Model name to use
        device: Device to use (None for auto-detection)
        quantize: Whether to use 8-bit quantization for large models
        
    Returns:
        ModelManager instance
    """
    return ModelManager.get_instance(model_name, device, quantize)


# Adapter class for compatibility with ai_providers.py interface
class EmbeddingProviderAdapter:
    """
    EmbeddingProviderAdapter Class

    This class provides an adapter interface for generating embeddings using the ModelManager.
    It is compatible with the `ai_providers.py` interface and supports batch embedding generation.
    """

    def __init__(self, model_name: str = "microsoft/phi-1_5", quantize: bool = False):
        """
        Initialize the EmbeddingProviderAdapter.

        Args:
            model_name: Name of the model to use for embedding generation
            quantize: Whether to use quantization for the model
        """
        self.model_name = model_name
        self.manager = get_model_manager(model_name, quantize=quantize)

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to generate embedding for

        Returns:
            List of floats representing the embedding vector
        """
        return self.manager.generate_embedding(text)

    def batch_generate_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to generate embeddings for

        Returns:
            List of embedding vectors
        """
        return self.manager.batch_generate_embeddings(texts)

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this adapter.

        Returns:
            Integer representing the embedding dimension
        """
        if "microsoft/phi-1_5" in self.model_name:
            return 2048
        return 768  # Default dimension for other models

    def set_device(self, device: Optional[str]) -> None:
        """
        Set the device for the embedding provider.
        
        Args:
            device: Device to use (cuda or cpu)
        """
        if device is not None and self.manager:
            self.manager.preferred_device = device
            # If we have a sentence transformer, update its device too
            if hasattr(self.manager, 'sentence_transformer') and self.manager.sentence_transformer is not None:
                self.manager.sentence_transformer = self.manager.sentence_transformer.to(device)


@typechecked
def get_embedding_provider(model_name: Optional[str] = None) -> EmbeddingProviderAdapter:
    """
    Get embedding provider compatible with ai_providers.py interface.
    This provides a bridge to the ModelManager for code that expects
    the ai_providers.py interface.
    
    Args:
        model_name: Optional model name
        
    Returns:
        Object with generate_embedding method
    """
    # Use model name if provided, otherwise use default
    model = model_name if model_name else "microsoft/phi-1_5"

    # Create adapter
    return EmbeddingProviderAdapter(model)
