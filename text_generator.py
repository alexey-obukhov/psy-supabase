from typing import Dict
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import logging
import traceback
from typing import Dict
from utilities.therapeutic_promt import prompt_templates
from utilities.text_utils import clean_text

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextGenerator:
    def __init__(self, model_name: str, device: str, use_bfloat16: bool = False):
        self.device = device
        self.model_name = model_name
        self.use_bfloat16 = use_bfloat16
        self.tokenizer = None
        self.model = None
        self.toxic_tokenizer = None
        self.toxic_model = None
        self.prompt_templates = prompt_templates

    def _load_language_model(self):
        """Loads the language model and tokenizer."""
        logger.info(f"Loading language model: {self.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            model_kwargs = {"trust_remote_code": True}
            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"  # Use automatic device mapping
                if self.use_bfloat16 and torch.cuda.is_bf16_supported():
                    model_kwargs["torch_dtype"] = torch.bfloat16
                else:
                    model_kwargs["torch_dtype"] = torch.float16  # Use float16 for Phi-1.5
            elif self.device == "cpu":
                model_kwargs["torch_dtype"] = torch.float32 #Prevent error with phi-2

            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            self.model.eval()  # Put the model in evaluation mode

        except Exception as e:
            logger.error(f"Error loading language model {self.model_name}: {e}\n{traceback.format_exc()}")
            raise

    def _unload_language_model(self):
        """Unloads the language model and tokenizer from memory."""
        logger.info(f"Unloading language model: {self.model_name}")
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.device == "cuda":
            torch.cuda.empty_cache()  # Clear GPU cache

    def _load_toxicity_model(self):
        """Loads the toxicity model and tokenizer."""
        logger.info("Loading toxicity model: facebook/roberta-hate-speech-dynabench-r4-target")
        try:
            self.toxic_tokenizer = AutoTokenizer.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")
            self.toxic_model = AutoModelForSequenceClassification.from_pretrained(
                "facebook/roberta-hate-speech-dynabench-r4-target",
                 torch_dtype=torch.float32, # Use float32 for CPU
            )
            # ALWAYS keep the toxicity model on CPU
            self.toxic_model.to("cpu")  # Explicitly on CPU
            self.toxic_model.eval()
        except Exception as e:
            logger.error(f"Error loading toxicity model: {e}\n{traceback.format_exc()}")
            raise

    def _unload_toxicity_model(self):
        """Unloads the toxicity model and tokenizer from memory."""
        logger.info("Unloading toxicity model")
        if self.toxic_model is not None:
            del self.toxic_model
            self.toxic_model = None
        if self.toxic_tokenizer is not None:
            del self.toxic_tokenizer
            self.toxic_tokenizer = None
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def generate_text(self, prompt: str, max_length: int = 200, temperature: float = 0.7,
                      top_p: float = 0.95, no_repeat_ngram_size: int = 2) -> str:
        """
        Generates therapeutic responses using best practices from similar projects.
        This implementation uses token-based extraction for reliability.
        """
        self._load_language_model()
        try:
            logger.info(f"Generating text for prompt of length {len(prompt)}")
            
            # 1. First, encode the prompt to tokens
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            input_length = input_ids.shape[1]
            
            # 2. Generate continuation
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=True
                )
            
            # 3. Extract only the new tokens (not including input)
            new_tokens = output_ids[0][input_length:]
            
            # 4. Decode only the new tokens
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # 5. Apply specialized cleaning for therapeutic responses
            response = self._clean_therapeutic_response(response)
            
            # 6. Check for empty response and provide appropriate fallback
            if not response.strip():
                logger.warning("Empty response after cleaning")
                return self._get_supportive_fallback_response()
            
            return response
            
        except Exception as e:
            logger.error(f"Error in text generation: {e}\n{traceback.format_exc()}")
            return "I apologize for the technical difficulty. I'm here to support you - would you like to share more about what you're experiencing?"
        finally:
            self._unload_language_model()
        
    def _clean_therapeutic_response(self, text: str) -> str:
        """
        Specialized cleaning for therapeutic responses based on best practices.
        Removes artifacts while preserving therapeutic content.
        """
        import re
        
        # Stop at any marker indicating a new speaker
        stop_markers = ["USER:", "THERAPIST:", "PATIENT:", "CLIENT:", "DOCTOR:"]
        for marker in stop_markers:
            if marker in text:
                text = text.split(marker)[0]
        
        # Fix inconsistent role markers that might appear
        text = re.sub(r'\b(TR:|THERPST:|THERAP:|THERAPY:)', '', text, flags=re.IGNORECASE)
        
        # Remove any bracketed metadata that might be generated
        text = re.sub(r'\[.*?\]', '', text)
        
        # Remove lines that are just single-word responses
        lines = text.split('\n')
        lines = [line for line in lines if len(line.strip().split()) > 1]
        text = '\n'.join(lines)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove empty parentheses that sometimes get generated
        text = re.sub(r'\(\s*\)', '', text)
        
        return text

    def _get_supportive_fallback_response(self) -> str:
        """
        Provides context-appropriate therapeutic responses when generation fails.
        Uses varied responses to prevent repetitive fallbacks.
        """
        import random
        fallbacks = [
            "I'm here to listen and support you. Could you tell me more about what you're experiencing?",
            "It sounds like you're going through a difficult time. I'm here to help you work through these feelings.",
            "Thank you for sharing that with me. Would you like to explore these thoughts a bit more?",
            "Your feelings are valid, and I'm here to support you. How else have you been coping with this?",
            "I appreciate you opening up. Let's work together to understand what you're going through."
        ]
        return random.choice(fallbacks)

    def is_toxic(self, text: str) -> bool:
        """Checks if text is toxic, loading the toxicity model if needed."""
        self._load_toxicity_model()  # Load on demand
        try:
            inputs = self.toxic_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)  # No .to(device)
            with torch.no_grad():
                outputs = self.toxic_model(**inputs)  # No .to(device)
            toxic_score = outputs.logits.softmax(dim=-1)[0, -1].item()
            return toxic_score > 0.7
        except Exception as e:
            logger.error(f"Error during toxicity check: {e}\n{traceback.format_exc()}")
            return False
        finally:
            self._unload_toxicity_model()  # Unload after use


    def get_embedding(self, text: str) -> torch.Tensor:
        """Generates embeddings, loading the language model if needed."""
        self._load_language_model() # Load on demand
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)  # CRITICAL: Add this!

            # Corrected: Access hidden states from the 'outputs' object correctly.
            hidden_states = outputs.hidden_states[-1]  # Get the last hidden state
            embeddings = hidden_states.mean(dim=1)  # Mean pooling
            return embeddings
        except Exception as e:
            logger.error(f"Error during embedding generation: {e}\n{traceback.format_exc()}")
            return None
        finally:
            self._unload_language_model()  # Unload after use

    def generate_category_info(self, question: str, topic: str = "Therapy") -> Dict[str, str]:
        """
        Generate category information for a question.
        
        Args:
            question: The user question or title
            topic: The general topic area
            
        Returns:
            Dictionary mapping categories to descriptions
        """
        if not question:
            return {cat: "" for cat in self.prompt_templates.keys()}
        
        # Simple keyword-based approach that doesn't require complex AI generation
        category_info = {}
        
        # Map keywords to therapeutic approaches
        if any(word in question.lower() for word in ["sad", "depress", "unhappy", "down", "hopeless"]):
            category_info["Empathy and Validation"] = "Validating feelings of sadness and offering emotional support"
            
        if any(word in question.lower() for word in ["anxious", "worry", "stress", "fear", "nervous"]):
            category_info["Affirmation and Reassurance"] = "Providing reassurance for anxiety and stress"
            
        if any(word in question.lower() for word in ["think", "thought", "belief", "irrational", "always", "never"]):
            category_info["Cognitive Behavioral Therapy (CBT)"] = "Addressing thought patterns that may contribute to distress"
            
        if any(word in question.lower() for word in ["breath", "relax", "calm", "ground", "meditate"]):
            category_info["Mindfulness and Relaxation"] = "Techniques for present-moment awareness and relaxation"
            
        if any(word in question.lower() for word in ["advice", "suggest", "help", "tip", "how to"]):
            category_info["Providing Suggestions"] = "Offering gentle suggestions or strategies"
            
        # Provide default categories if none matched
        if not category_info:
            category_info["Others"] = "General therapeutic support"
            
        # Fill in empty values for other categories
        for cat in self.prompt_templates.keys():
            if cat not in category_info:
                category_info[cat] = ""
                
        return category_info
