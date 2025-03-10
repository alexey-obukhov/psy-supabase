import gc
import torch
import torch.cuda
import traceback
from typing import List, Optional, Dict, Any, ClassVar
from school_logging.log import ColoredLogger
from psy_supabase.core.text_generator import TextGenerator

# Create a model manager class to handle loading/unloading
class ModelManager:
    # Class variable to store instances (no global variables)
    _instances: ClassVar[Dict[str, 'ModelManager']] = {}
    
    @classmethod
    def get_instance(cls, model_name: str = "microsoft/phi-1_5", device: str = None) -> 'ModelManager':
        """
        Get or create a ModelManager instance.
        
        Args:
            model_name: Model name to use
            device: Device to use (None for auto-detection)
            
        Returns:
            ModelManager instance
        """
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # Use model_name as key
        instance_key = f"{model_name}_{device}"
        
        # Create if doesn't exist
        if instance_key not in cls._instances:
            cls._instances[instance_key] = cls(model_name, device)
            
        # Get the instance
        instance = cls._instances[instance_key]
        
        # Update device if different
        if instance.preferred_device != device:
            instance.preferred_device = device
            
        return instance

    def __init__(self, model_name, device="cpu"):
        self.model_name = model_name
        self.preferred_device = device
        self.generator = None
        self.embedding_model = None
        self.sentence_transformer = None
        self.logger = ColoredLogger("ModelManager")
        self.logger.info(f"ModelManager initialized with model: {model_name} and device: {device}")
        
    def get_generator(self):
        # Initialize on first use
        if self.generator is None:
            self.logger.info(f"Creating new TextGenerator on {self.preferred_device}")
            self.generator = TextGenerator(self.model_name, self.preferred_device)
            self.current_device = self.preferred_device
        else:
            # Make sure model is fully on the right device
            self.logger.info(f"Moving existing model to {self.preferred_device}")
            if hasattr(self.generator, 'model'):
                self.generator.model = self.generator.model.to(self.preferred_device)
                self.generator.device = self.preferred_device
                self.current_device = self.preferred_device
        return self.generator
        
    def free_memory(self):
        if self.generator and hasattr(self.generator, 'model'):
            self.logger.info(f"Freeing GPU memory - moving model to CPU")
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
                    from sentence_transformers import SentenceTransformer
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
                    from sentence_transformers import SentenceTransformer
                    model_name = "sentence-transformers/all-mpnet-base-v2"
                    self.sentence_transformer = SentenceTransformer(model_name)
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


def get_model_manager(model_name: str = "microsoft/phi-1_5", device: str = None) -> ModelManager:
    """
    Get a ModelManager instance.
    
    Args:
        model_name: Model name to use
        device: Device to use (None for auto-detection)
        
    Returns:
        ModelManager instance
    """
    return ModelManager.get_instance(model_name, device)


# Adapter class for compatibility with ai_providers.py interface
class EmbeddingProviderAdapter:
    """
    Adapter class that provides the same interface as ai_providers.py.
    Uses ModelManager internally.
    """
    def __init__(self, model_name: str = "microsoft/phi-1_5"):
        self.model_name = model_name
        self.manager = get_model_manager(model_name)
        
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text"""
        return self.manager.generate_embedding(text)
        
    def batch_generate_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for batch of texts"""
        return self.manager.batch_generate_embeddings(texts)


def get_embedding_provider(model_name: str = None):
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
