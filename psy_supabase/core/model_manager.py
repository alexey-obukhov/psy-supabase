"""
Model Manager Module

This module provides a `ModelManager` class that manages multiple AI models used in the PSY
therapeutic assistant:

1. **Text Generation**: Microsoft Phi-1.5 model for generating therapeutic responses
2. **Embedding Generation**: SentenceTransformer model for creating vector representations of text
3. **Content Moderation**: Toxicity detection model for ensuring safe conversations

The `ModelManager` handles:
- Model downloading and local caching to reduce startup time
- Device management (CPU/GPU switching)
- Memory optimization with quantization options
- Automatic fallbacks when primary models encounter issues

Key components:
- `ModelManager`: Core class managing model instances and memory
- `EmbeddingProviderAdapter`: Standardized interface for embedding generation
- `get_model_manager()`: Factory function to get model instances
- `get_embedding_provider()`: Creates a simple embedding interface

This design centralizes model management for efficiency and provides a clean API
for other components to access AI capabilities without handling the underlying
complexity of model loading and memory management.
"""

import gc
import os
import traceback
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

import torch
import torch.cuda
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from typeguard import typechecked

from psy_supabase import get_package_logger
from psy_supabase.config import DEFAULT_EMBEDDING_MODEL, TEXT_GENERATING_MODEL, TOXIC_CLASSIFICATION_MODEL
from psy_supabase.core.text_generator import TextGenerator
from psy_supabase.utilities.common import get_models_dir
from psy_supabase.utilities.common import load_toxicity_model as common_load_toxicity_model
from psy_supabase.utilities.utils import download_and_store_model


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

    _instance = None
    _model = None
    # Class variable to store instances (no global variables)
    _instances: ClassVar[Dict[str, "ModelManager"]] = {}

    embedding_model: Optional[SentenceTransformer] = None
    sentence_transformer: Optional[SentenceTransformer] = None
    generator: Optional[TextGenerator] = None
    toxicity_model: Optional[Union[AutoModelForCausalLM, AutoModelForSequenceClassification]] = None
    toxicity_tokenizer: Optional[AutoTokenizer] = None

    @classmethod
    @typechecked
    def get_instance(
        cls, model_name: str = TEXT_GENERATING_MODEL, device: Optional[str] = None, quantize: bool = False
    ) -> "ModelManager":
        """
        Get or create a ModelManager instance.

        Args:
            model_name: Model name to use
            device: Device to use (None for auto-detection)
            quantize: Whether to use 8-bit quantization for large models

        Returns:
            ModelManager instance
        """
        if cls._instance is None:
            cls._instance = cls(model_name, device, quantize)
        return cls._instance

    def __init__(self, model_name: str, device: Optional[str] = None, quantize: bool = False) -> None:
        self.model_name = model_name
        self.logger = get_package_logger(__name__)
        self.MODELS_DIR = get_models_dir()
        self.preferred_device = self._get_preferred_device(device)
        self.quantize = quantize

        # Download and store main generation model
        self.generation_model_path = self._download_if_needed(model_name, AutoModelForCausalLM, "generation")

        # Download and store embedding model (sentence-transformers)
        self.embedding_model_path = self._download_if_needed(DEFAULT_EMBEDDING_MODEL, SentenceTransformer, "embedding")

        # Download and store toxicity model (customize as needed)
        self.toxicity_model_path = self._download_if_needed(
            TOXIC_CLASSIFICATION_MODEL, AutoModelForCausalLM, "toxicity"
        )

        # Now load models from local paths
        self.generator = TextGenerator(self.generation_model_path, self.preferred_device, quantize=self.quantize)
        self.sentence_transformer = SentenceTransformer(self.embedding_model_path)
        self.toxicity_model = AutoModelForCausalLM.from_pretrained(self.toxicity_model_path)
        self.toxicity_tokenizer = AutoTokenizer.from_pretrained(self.toxicity_model_path)

    def _get_preferred_device(self, device: Optional[str]) -> str:
        """Get preferred device using CUDA_CONFIG."""
        from ..config import CUDA_CONFIG

        if device is not None:
            return device

        if CUDA_CONFIG.get("force_cpu_fallback", False):
            return "cpu"

        device_selection = CUDA_CONFIG.get("device_selection", "auto")
        if device_selection == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device_selection == "cpu":
            return "cpu"
        if isinstance(device_selection, str) and device_selection.startswith("cuda"):
            return device_selection if torch.cuda.is_available() else "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _download_if_needed(self, model_name: str, model_class: Any, model_type: str) -> str:
        """
        Download model if not present locally.
        """
        local_path = os.path.join(self.MODELS_DIR, model_name.replace("/", "_"))
        if not (os.path.exists(local_path) and os.listdir(local_path)):
            self.logger.info(f"Downloading model {model_name} to {local_path}")
            os.makedirs(local_path, exist_ok=True)
            if model_class is SentenceTransformer:
                # Download and save SentenceTransformer model
                model = SentenceTransformer(model_name)
                model.save(local_path)
            else:
                # Download and save Hugging Face model and tokenizer
                model = model_class.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model.save_pretrained(local_path)
                tokenizer.save_pretrained(local_path)
            self.logger.info(f"{model_type.capitalize()} model saved to {local_path}")
        else:
            self.logger.info(f"{model_type.capitalize()} model found locally at {local_path}")
        return local_path

    def get_local_model_path(self) -> str:
        """Get the local path for the model"""
        # Use just the model name without organisation prefix for folder
        model_folder = self.model_name.split("/")[-1] if "/" in self.model_name else self.model_name
        return os.path.join(self.MODELS_DIR, model_folder)

    def is_model_downloaded(self) -> bool:
        """Check if the model is already downloaded locally"""
        model_path = self.get_local_model_path()

        # Check if folder exists and is not empty
        if os.path.exists(model_path) and os.path.isdir(model_path):
            # Check if directory has any files
            return len(os.listdir(model_path)) > 0

        return False

    @typechecked
    def get_generator(self) -> TextGenerator:
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
                self.logger.info("Loading model from local path: %s", local_path)
                self.generator = TextGenerator(local_path, self.preferred_device, quantize=self.quantize)
            else:
                # We need to download the model regardless of quantization
                self.logger.info("Model not found locally. Downloading %s to %s", self.model_name, local_path)

                # Create directory
                os.makedirs(local_path, exist_ok=True)

                # Download and save tokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                tokenizer.save_pretrained(local_path)
                self.logger.info("Tokenizer saved to %s", local_path)

                try:
                    # Download and save full model (without quantization)
                    self.logger.info("Downloading full model weights to %s", local_path)
                    model = AutoModelForCausalLM.from_pretrained(self.model_name)
                    model.save_pretrained(local_path)
                    self.logger.info("Full model saved to %s", local_path)

                    # Now create the generator, using quantization if requested
                    self.generator = TextGenerator(local_path, self.preferred_device, quantize=self.quantize)
                except Exception as e:
                    self.logger.error("Error downloading full model: %s", e)
                    # If full model download fails, try direct initialization
                    self.generator = TextGenerator(self.model_name, self.preferred_device, quantize=self.quantize)

            self.current_device = self.preferred_device
        else:
            # Make sure model is fully on the right device
            self.logger.info("Ensuring generator model is on device: %s", self.preferred_device)
            if self.generator is not None and hasattr(self.generator, "model") and self.generator.model is not None:
                # Only try to move model if not using device_map='auto'
                if not (
                    self.quantize and hasattr(self.generator, "using_device_map") and self.generator.using_device_map
                ):
                    self.generator.model = self.generator.model.to(self.preferred_device)
                if hasattr(self.generator, "device"):
                    self.generator.device = self.preferred_device
                    self.current_device = self.preferred_device

        return self.generator

    @typechecked
    def load_toxicity_model(
        self,
    ) -> Tuple[Union[AutoModelForCausalLM, AutoModelForSequenceClassification], AutoTokenizer]:
        """Load the toxicity detection model using the common utility."""
        try:
            # Call the common function
            model, tokenizer = common_load_toxicity_model(self.logger)

            if model is None or tokenizer is None:
                raise RuntimeError("Toxicity model or tokenizer failed to load.")

            if not isinstance(tokenizer, AutoTokenizer):
                # This should ideally not happen based on previous logs, but good safety check
                self.logger.error("Tokenizer loaded by common function is not AutoTokenizer. Type: %s", type(tokenizer))
                raise TypeError(f"Expected AutoTokenizer, got {type(tokenizer)}")

            # Assign to attributes
            self.toxicity_model = model
            self.toxicity_tokenizer = tokenizer  # Assign the Fast tokenizer

            return self.toxicity_model, self.toxicity_tokenizer

        except Exception as e:
            self.logger.error("Error loading toxicity model: %s", e)
            self.toxicity_model = None
            self.toxicity_tokenizer = None
            raise

    def free_memory(self) -> None:
        """Free up GPU memory by moving model to CPU and releasing CUDA memory."""
        if self.generator is not None and hasattr(self.generator, "model") and self.generator.model is not None:
            self.logger.info("Freeing GPU memory - moving generator model to CPU")
            self.generator.model = self.generator.model.to("cpu")
            if hasattr(self.generator, "device"):
                self.generator.device = "cpu"
            self.current_device = "cpu"
            self.logger.info("Generator model moved to CPU.")

            # Release CUDA memory
            torch.cuda.empty_cache()
            # Run garbage collector
            gc.collect()
            self.logger.info("GPU memory freed")

        # Also free the embedding model if it exists
        if self.embedding_model is not None:
            self.logger.info("Freeing embedding model memory")
            self.embedding_model = self.embedding_model.to("cpu")
            torch.cuda.empty_cache()
            gc.collect()

    def move_to_cpu(self) -> None:
        """Move model to CPU and clear CUDA cache properly"""
        # Check if generator exists and has a model attribute
        if (
            self.generator is not None
            and hasattr(self.generator, "model")
            and self.generator.model is not None
            and hasattr(self.generator, "device")
            and self.generator.device == "cuda"
        ):
            # First move model to CPU
            self.generator.model = self.generator.model.to("cpu")
            self.generator.device = "cpu"
            self.current_device = "cpu"

            # Explicitly delete any CUDA tensor caches
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for CUDA operations to finish

            # Log actual memory state
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                self.logger.info("After moving to CPU: %.2fGB allocated, %.2fGB reserved", allocated, reserved)

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
            generator: TextGenerator = self.get_generator()
            if generator is not None:
                if hasattr(generator, "get_embedding"):
                    self.logger.info("Using TextGenerator for embedding generation")
                    embedding_tensor = generator.get_embedding(text)
                    if embedding_tensor is not None:
                        return embedding_tensor.squeeze().cpu().tolist()

                # Fallback: Generate embedding directly using the model
                self.logger.info("Generating embedding directly from model hidden states")
                if hasattr(generator, "model") and hasattr(generator, "tokenizer") and generator.tokenizer is not None:
                    inputs = generator.tokenizer(
                        text, return_tensors="pt", padding=True, truncation=True, max_length=512
                    )
                    if hasattr(inputs, "to") and callable(inputs.to):
                        inputs = inputs.to(generator.device)
                    else:
                        # Move each tensor in the dict to the device
                        inputs = {k: v.to(generator.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        if generator.model is not None:
                            outputs = generator.model(**inputs, output_hidden_states=True)
                            # Use last hidden state
                            hidden_states = outputs.hidden_states[-1]
                            # Mean pooling
                            embedding = hidden_states.mean(dim=1)
                            return embedding.squeeze().cpu().tolist()
                        else:
                            self.logger.error("Generator model is None, cannot generate embedding")
                            return None

                # If we got here, we couldn't use the generator for embeddings
                raise ValueError("Model doesn't support embedding generation")

        except Exception as e:
            self.logger.error("Error generating embedding with main model: %s", e)
            self.logger.error(traceback.format_exc())

            # Try with sentence transformer as fallback
            return self.generate_embedding_with_sentence_transformer(text)

    def generate_embedding_with_sentence_transformer(self, text: str) -> Optional[List[float]]:
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
                    # Use config model
                    self.sentence_transformer = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
                    if self.preferred_device == "cuda" and torch.cuda.is_available():
                        self.sentence_transformer = self.sentence_transformer.to(self.preferred_device)
                except ImportError:
                    self.logger.error(
                        "sentence-transformers not installed. Install with: pip install sentence-transformers"
                    )
                    return None

            # Generate embedding
            embedding = self.sentence_transformer.encode(text)
            return embedding.tolist()

        except Exception as e:
            self.logger.error("Error generating embedding with sentence transformer: %s", e)
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
                    os.makedirs(self.MODELS_DIR, exist_ok=True)

                    # Use config model name
                    st_model_name = DEFAULT_EMBEDDING_MODEL.split("/")[-1]
                    local_path = os.path.join(self.MODELS_DIR, f"sentence-transformers_{st_model_name}")

                    if os.path.exists(local_path) and os.path.isdir(local_path) and len(os.listdir(local_path)) > 0:
                        self.logger.info("Loading SentenceTransformer from local path: %s", local_path)
                        self.sentence_transformer = SentenceTransformer(local_path)
                    else:
                        self.logger.info("Downloading SentenceTransformer to %s", local_path)
                        os.makedirs(local_path, exist_ok=True)

                        # Use config model
                        self.sentence_transformer = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
                        self.sentence_transformer.save(local_path)
                        self.logger.info("SentenceTransformer saved to %s", local_path)

                # Move to the right device
                if self.preferred_device == "cuda" and torch.cuda.is_available():
                    self.sentence_transformer = self.sentence_transformer.to(self.preferred_device)

                # Get embedding dimension
                embedding_dim = self.sentence_transformer.get_sentence_embedding_dimension()
                if embedding_dim is None:
                    embedding_dim = 1536

                # Process texts (replace empty with spaces to avoid errors)
                processed_texts = [text if text and text.strip() else " " for text in texts]

                # Generate embeddings in batch
                embeddings = self.sentence_transformer.encode(processed_texts)

                # Format results
                results: List[Optional[List[float]]] = []
                for i, text in enumerate(texts):
                    if not text or not text.strip():
                        results.append([0.0] * embedding_dim)  # Zero vector
                    else:
                        results.append(embeddings[i].tolist())

                return results

            except Exception as e:
                self.logger.error("Error in batch embedding: %s", e)
                self.logger.error(traceback.format_exc())

        # For smaller batches or if sentence-transformers failed, use the main model
        return [self.generate_embedding(text) for text in texts]


@typechecked
def get_model_manager(
    model_name: str = TEXT_GENERATING_MODEL, device: Optional[str] = None, quantize: bool = False
) -> ModelManager:
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


class EmbeddingProviderAdapter:
    """
    EmbeddingProviderAdapter Class

    This class provides an adapter interface for generating embeddings using the ModelManager.
    It is compatible with the `ai_providers.py` interface and supports batch embedding generation.
    """

    def __init__(self, provider_type: str = "local", model_name: str = TEXT_GENERATING_MODEL):
        """Initialize the embedding provider."""
        self.provider_type = provider_type
        self.model_name = model_name
        self._provider = None
        self.logger = get_package_logger(__name__)

    def get_embedding_dimension(self) -> int:
        """
        Return the dimension of embeddings based on the model.

        Returns:
            The embedding dimension (eg. 2048 for phi-1.5, 768 for facebook models)
        """
        embedding_model = DEFAULT_EMBEDDING_MODEL.lower()
        if "all-minilm-l6-v2" in embedding_model:
            return 384
        if "all-mpnet-base-v2" in embedding_model:
            return 768
        if "phi" in self.model_name.lower():
            return 2048
        if "facebook" in self.model_name.lower() or "fb" in self.model_name.lower():
            return 768
        # Default
        return 1536

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text.
        Uses ModelManager to leverage existing functionality.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding
        """
        if not text or not text.strip():
            return [0.0] * self.get_embedding_dimension()

        try:
            # Use the existing ModelManager directly - no circular import needed
            model_manager = get_model_manager(self.model_name)

            # Generate embedding using existing functionality
            embedding = model_manager.generate_embedding(text)

            # If embedding failed, use fallback
            if embedding is None:
                embedding = model_manager.generate_embedding_with_sentence_transformer(text)

            # If still None, return zeros
            if embedding is None:
                self.logger.warning("Embedding generation failed, returning zeros")
                return [0.0] * self.get_embedding_dimension()

            return embedding

        except Exception as e:
            self.logger.error("Error generating embedding: %s", e)
            return [0.0] * self.get_embedding_dimension()

    def _initialize_provider(self) -> None:
        """
        Initialize the provider instance.

        This method is kept for backward compatibility, but the actual initialization
        is now handled directly in the generate_embedding method.
        """
        if self.provider_type == "local":
            try:
                # Use the existing model manager directly - no circular import needed
                self._provider = get_model_manager(self.model_name)
                self.logger.info("Initialized local embedding provider with model: %s", self.model_name)
            except Exception as e:
                self.logger.error("Failed to initialize embedding provider: %s", e)
        else:
            # For other provider types
            self.logger.warning("Provider type %s initialization not implemented", self.provider_type)
            self._provider = None

    def batch_generate_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (or None for failed embeddings)
        """
        if not texts:
            return []

        try:
            # Get the model manager
            model_manager = get_model_manager(self.model_name)

            # Use the batch function from ModelManager
            results = model_manager.batch_generate_embeddings(texts)  # This returns List[Optional[List[float]]]

            # Handle potential None return from model_manager.batch_generate_embeddings itself
            if results is None:
                self.logger.error("Batch embedding generation failed unexpectedly, returning zeros.")
                return [[0.0] * self.get_embedding_dimension() for _ in texts]

            return results

        except Exception as e:
            self.logger.error("Error in batch embedding: %s", e)
            # Return list of None values matching the expected type
            return [None for _ in texts]  # Or return zeros


@typechecked
def get_embedding_provider(model_name: Optional[str] = None) -> EmbeddingProviderAdapter:
    """
    Get a standardized embedding provider for vector representations of text.

    This function creates an EmbeddingProviderAdapter that provides a simplified
    interface for generating embeddings in both single and batch operations.
    It abstracts away the complexity of the underlying ModelManager while ensuring
    consistent handling of embedding generation throughout the application.

    Args:
        model_name: Optional model name to use for embeddings (default: "microsoft/phi-1.5")

    Returns:
        EmbeddingProviderAdapter with consistent generate_embedding methods
    """
    # Use model name if provided, otherwise use default
    model = model_name if model_name else TEXT_GENERATING_MODEL

    # Create adapter
    return EmbeddingProviderAdapter(model)
