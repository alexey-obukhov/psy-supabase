import gc
import torch.cuda
from school_logging.log import ColoredLogger
from text_generator import TextGenerator

# Create a model manager class to handle loading/unloading
class ModelManager:
    def __init__(self, model_name, device="cpu"):
        self.model_name = model_name
        self.preferred_device = device
        self.generator = None
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
