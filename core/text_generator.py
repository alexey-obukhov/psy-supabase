"""
TextGenerator Module
====================

This module provides therapeutic text generation capabilities using transformer-based language models.
It handles prompt templating, context management, response generation, and safety checks for
therapeutic AI applications.

Key Components:
--------------
1. Model Management: Loading/unloading language and toxicity detection models
2. Template Handling: Jinja2-based therapeutic prompt templates with context insertion
3. Response Generation: Optimized generation with proper parameter handling
4. Safety Systems: Multiple layers of validation and fallbacks for therapeutic safety
5. Context Enhancement: Structured formatting of conversation, knowledge, and psychological contexts

Classes:
-------
TextGenerator: Main class that handles all text generation functionality

Usage Example:
------------
```python
from psy_supabase.core.text_generator import TextGenerator

# Initialize with model path and device
generator = TextGenerator(
    model_name="microsoft/phi-1_5", 
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Generate a response using a specific template
response = generator.generate_therapeutic_response(
    user_question="I've been feeling really anxious lately",
    template_name="anxiety_support",
    context={
        "knowledge_context": "Anxiety can manifest as physical symptoms.",
        "emotional_signals": ["worry", "nervousness"]
    }
)
"""
import os
import torch
import traceback
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from psy_supabase.utilities.utils import get_dir
from psy_supabase.utilities.templates.therapeutic_prompt import prompt_templates
from psy_supabase.utilities.utils_mapping import map_approach_to_template
from jinja2 import Environment, FileSystemLoader, select_autoescape
from psy_supabase.utilities.prompt_selector import PromptSelector
from school_logging.log import ColoredLogger
from detoxify import Detoxify

if TYPE_CHECKING:
    from psy_supabase.core.dynamic_rag import DynamicRAGRetriever
    from psy_supabase.utilities.prompt_selector import PromptSelector

# Load environment variables
load_dotenv()

logger = ColoredLogger(__name__)

class TextGenerator:
    MODELS_DIR = get_dir("models")

    def __init__(self, model_name: str, device: str, use_bfloat16: bool = False, quantize: bool = False):
        self.device = device
        self.model_name = model_name
        self.use_bfloat16 = use_bfloat16
        self.quantize = quantize
        self.tokenizer = None
        self.model = None
        self.toxic_tokenizer = None
        self.toxic_model = None
        self.prompt_templates = prompt_templates
        
        # Load the model immediately on initialization
        self._load_model()

        # Set cache directory for Detoxify to use our models directory
        os.environ["TRANSFORMERS_CACHE"] = self.MODELS_DIR
        
        # Initialize Detoxify once
        self.detoxify = Detoxify("original-small")
        
        logger.info(f"TextGenerator initialized with model: {model_name} on device: {device}")

    def _load_model(self):
        """Loads the model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Set the pad token if not defined
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Configure model loading
            load_config = {}
            
            # Add quantization if requested
            if self.quantize and self.device == "cuda":
                try:
                    # Try to import bitsandbytes
                    import bitsandbytes
                    logger.info("Using 8-bit quantization with bitsandbytes")
                    load_config["load_in_8bit"] = True
                    load_config["device_map"] = "auto"
                    
                    # Only add bfloat16 if specifically requested AND the GPU supports it
                    if self.use_bfloat16 and torch.cuda.is_bf16_supported():
                        logger.info("Using bfloat16 with 8-bit quantization")
                        load_config["torch_dtype"] = torch.bfloat16
                        
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        **load_config
                    )
                    self.using_device_map = True
                except ImportError:
                    logger.warning("bitsandbytes not installed, falling back to standard loading")
                    load_config["torch_dtype"] = torch.bfloat16 if self.use_bfloat16 else torch.float32
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        **load_config
                    )
                    self.model.to(self.device)
            else:
                # Standard loading without quantization
                load_config["torch_dtype"] = torch.bfloat16 if self.use_bfloat16 else torch.float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **load_config
                )
                self.model.to(self.device)

            self.model.eval()  # Set to evaluation mode
            
            logger.info(f"Model loaded successfully: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.error(traceback.format_exc())
            raise

    def _ensure_model_loaded(self):
        """Ensures the model and tokenizer are loaded."""
        if self.tokenizer is None or self.model is None:
            logger.warning("Model or tokenizer not loaded. Reloading...")
            self._load_model()

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
        """Loads the toxicity model and tokenizer with local caching."""
        logger.info("Loading toxicity model: facebook/roberta-hate-speech-dynabench-r4-target")
        try:
            # Define model name and paths
            toxicity_model_name = "facebook/roberta-hate-speech-dynabench-r4-target"
            model_folder = toxicity_model_name.split('/')[-1]
            
            local_path = os.path.join(self.MODELS_DIR, model_folder)
            
            # Create models dir if it doesn't exist
            os.makedirs(self.MODELS_DIR, exist_ok=True)
            
            # Check if model exists locally
            if os.path.exists(local_path) and os.path.isdir(local_path) and len(os.listdir(local_path)) > 0:
                # Use local model
                logger.info(f"Loading toxicity model from local path: {local_path}")
                self.toxic_tokenizer = AutoTokenizer.from_pretrained(local_path)
                self.toxic_model = AutoModelForSequenceClassification.from_pretrained(
                    local_path,
                    torch_dtype=torch.float32  # Use float32 for CPU
                )
            else:
                # Download model and save locally
                logger.info(f"Downloading toxicity model to {local_path}")
                os.makedirs(local_path, exist_ok=True)
                
                # Download and save tokenizer
                self.toxic_tokenizer = AutoTokenizer.from_pretrained(toxicity_model_name)
                self.toxic_tokenizer.save_pretrained(local_path)
                
                # Download and save model
                self.toxic_model = AutoModelForSequenceClassification.from_pretrained(
                    toxicity_model_name,
                    torch_dtype=torch.float32  # Use float32 for CPU
                )
                self.toxic_model.save_pretrained(local_path)
                logger.info(f"Toxicity model saved to {local_path}")
                
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

    def generate_text(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Generate text using the loaded model with token management.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Generated text
        """
        try:
            if not self.model or not self.tokenizer:
                logger.error("Model or tokenizer not loaded")
                return ""
            
            # Check token count and limit if necessary
            token_count = len(self.tokenizer.encode(prompt))
            max_context_tokens = 2048  # Set your model's context window size here
            logger.info(f"Prompt token count: {token_count} (limit: {max_context_tokens})")
            
            if token_count > max_context_tokens:
                # If too long, truncate the prompt to fit within token limit
                logger.warning(f"Prompt exceeds token limit ({token_count} > {max_context_tokens})")
                
                # Truncate by re-encoding with truncation
                truncated_tokens = self.tokenizer.encode(
                    prompt, 
                    truncation=True, 
                    max_length=max_context_tokens - 50  # Leave room for generation
                )
                prompt = self.tokenizer.decode(truncated_tokens)
                logger.info(f"Prompt truncated to {len(truncated_tokens)} tokens")
                
                # Log first and last part of truncated prompt for debugging
                prompt_start = prompt[:100]
                prompt_end = prompt[-100:]
                logger.debug(f"Truncated prompt starts with: {prompt_start}...")
                logger.debug(f"Truncated prompt ends with: ...{prompt_end}")
                
            # CRITICAL FIX: Check for phi-1.5 templating pattern in prompt
            if "<|im_start|>assistant" in prompt and prompt.endswith("<|im_start|>assistant\n"):
                # Force a different response starter to avoid the template pattern
                prompt = prompt + "I understand your concern about "
                logger.info("Added prompt starter text to avoid template response pattern")
            
            # Fix any "I understand your feelings about" patterns with blank spaces
            if "I understand your feelings about" in prompt:
                prompt = prompt.replace("I understand your feelings about", "I hear your concerns about")
                logger.info("Fixed known problematic pattern in prompt")
            
            # Generate text with the prepared prompt
            logger.info(f"Generating text with prompt of length {len(prompt)} (tokens: {token_count})")
            
            with torch.no_grad():
                input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
                
                # Check if input fits on device
                try:
                    # CRITICAL FIX: Modified params for phi-1.5 to avoid templating issues
                    output = self.model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=0.6,  # Slightly lower temperature
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.2,  # Higher penalty to reduce repetition
                        no_repeat_ngram_size=3  # Specifically for phi-1.5
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning("CUDA out of memory. Trying on CPU...")
                        # Move to CPU and try again
                        input_ids = input_ids.cpu()
                        self.model = self.model.cpu()
                        self.device = "cpu"
                        
                        output = self.model.generate(
                            input_ids,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            do_sample=True if temperature > 0 else False,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    else:
                        raise e
                        
                output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                
                # Extract only the generated response, not including the prompt
                response = output_text[len(prompt):]
                
                logger.info(f"Generated text of length {len(response)}")
                logger.debug(f"Generated text: {response}")
                return response
                
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            logger.error(traceback.format_exc())
            return ""

    def _clean_therapeutic_response(self, text: str) -> str:
        """
        Specialized cleaning for therapeutic responses based on best practices.
        Removes artifacts while preserving therapeutic content.
        """
        import re

        text = re.sub(r"_{2,}", "", text)

        # Check for direct instruction patterns like "Your therapeutic response should be:"
        instruction_prefixes = [
            r"Your therapeutic response should be:\s*[\"'](.+)[\"']",
            r"Your response should be:\s*[\"'](.+)[\"']",
            r"Your response:\s*[\"'](.+)[\"']",
            r"Respond with:\s*[\"'](.+)[\"']"
        ]
        
        # Check each instruction prefix pattern
        for pattern in instruction_prefixes:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                # Extract just the content inside the quotes
                extracted_text = match.group(1).strip()
                logger.warning(f"Found instruction pattern. Extracting actual response content.")
                return extracted_text

        # First pass: Check for code exercise patterns and return emergency fallback if found
        code_exercise_patterns = [
            r"# YOUR CODE HERE",
            r"# SOLUTION:",
            r"Answer the following:",
            r"\d+\.\s+Answer",
            r"```python",
            r"```javascript",
            r"def\s+\w+\s*\(",
            r"function\s+\w+\s*\(",
            r"class\s+\w+\s*\{",
            r"Illustration paragraph:",
        ]
        
        for pattern in code_exercise_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                if pattern == r"Illustration paragraph:":
                    # Special handling for illustration paragraphs - just cut off at that point
                    logger.warning("Found 'Illustration paragraph' pattern - truncating response")
                    return text.split("Illustration paragraph:")[0].strip()
                else:
                    # For other code patterns, use emergency fallback
                    logger.error(f"Code exercise pattern detected in response: {pattern}")
                    return self._get_emergency_fallback()

        # 1. Remove complete instructional sections
        instruction_patterns = [
            r"Instructions:.*?(?=\n\n|$)",
            r"Your response should:.*?(?=\n\n|$)",
            r"Response format:.*?(?=\n\n|$)",
            r"Remember to:.*?(?=\n\n|$)"
        ]
        
        for pattern in instruction_patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL|re.IGNORECASE)

        # 2. Check for and remove titles/sections that aren't part of the response
        text = re.sub(r'\n\s*\n\s*\n.*?(Title|Introduction|Chapter|Section|CHAPTER):', '', text, flags=re.DOTALL|re.IGNORECASE)
        
        # 3. Look for multiple consecutive newlines - often signal content boundary
        parts = re.split(r'\n\s*\n\s*\n', text)
        if len(parts) > 1:
            # Keep only the first part (the actual response)
            text = parts[0].strip()
        
        # 4. Apply dialogue cleaning logic - detect and extract therapist portions
        if re.search(r'(User|Therapist|CLIENT|THERAPIST):', text, re.IGNORECASE):
            try:
                therapist_responses = re.findall(r'(?:Therapist|THERAPIST):[\s]*(.*?)(?=\n[\s]*(?:User|CLIENT)|$)', text, re.DOTALL|re.IGNORECASE)
                if therapist_responses:
                    for response in therapist_responses:
                        if len(response.strip()) > 20:
                            return response.strip()
                    return self._get_emergency_fallback()
            except Exception as e:
                logger.error(f"Error extracting therapist response: {e}")
                return self._get_emergency_fallback()
        
        # 5. Apply other cleaning to remove dialogue markers
        for marker in ["USER:", "THERAPIST:", "PATIENT:", "CLIENT:", "DOCTOR:"]:
            if marker in text:
                text = text.split(marker)[0]
        
        # 6. Remove question/answer markers that appear in training data
        text = re.sub(r'Question \d+:|Answer:|Response:', '', text)
        
        # 7. Remove exercise instructions and repetition
        text = re.sub(r'Exercise:.*?(?=\n|$)', '', text, flags=re.IGNORECASE|re.DOTALL)
        text = re.sub(r'EXERCISE:.*?(?=\n|$)', '', text, flags=re.IGNORECASE|re.DOTALL)
        
        # 8. Comprehensive patterns to remove
        patterns_to_remove = [
            r'\b(?:Exercise|EXERCISE):.*?(?=\n|$)', 
            r'\bWrite (?:a|the) (?:response|answer).*?(?=\n|$)',
            r'\bYour response should.*?(?=\n|$)',
            r'\bRespond to the user.*?(?=\n|$)',
            r'\bInstructions:.*?(?=\n|$)',
            r'USER QUESTION:.*?(?=\n|$)',
            r'THERAPEUTIC APPROACH:.*?(?=\n|$)',
            r'RESPONSE \(keep.*?(?=\n|$)',
            r'PREVIOUS CONVERSATION:.*?(?=\n\n|$)',
            r'RELEVANT KNOWLEDGE:.*?(?=\n\n|$)',
            r'Current query:.*?(?=\n\n|$)',
            r'# YOUR CODE HERE',
            r'# SOLUTION:',
            r'Answer the following:',
            r'\d+\.\s+Answer',
            r'```python',
            r'```javascript',
        ]

        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE|re.DOTALL)

        # 9. Look for remaining instruction markers and truncate
        instruction_markers = [
            "Exercise:", "Instructions:", "Your response:", "Note to AI:", 
            "USER QUESTION:", "THERAPEUTIC APPROACH:", 
            "RESPONSE (keep", "PREVIOUS CONVERSATION:", 
            "RELEVANT KNOWLEDGE:", "Current query:", "Therapeutic approach:",
            "# YOUR CODE HERE", "# SOLUTION:", "Answer the following:"
        ]
        for marker in instruction_markers:
            if marker.lower() in text.lower():
                idx = text.lower().find(marker.lower())
                if idx >= 0:
                    text = text[:idx].strip()
                    break

        # 10. Clean meta-references to roles
        role_references = [
            r"As (?:a|your) therapist,? ",
            r"In my role as (?:a|your) therapist,? ",
            r"Speaking as (?:a|your) therapist,? ",
            r"As (?:a|your) counselor,? ",
        ]
        
        for pattern in role_references:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # 11. Fix line breaks and whitespace
        lines = text.split('\n')
        lines = [line for line in lines if len(line.strip().split()) > 1]
        text = '\n'.join(lines)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\(\s*\)', '', text)
        
        # 12. Check for code pattern markers that might have been missed
        code_markers = ["# 1.", "# 2.", "# 3.", "def ", "class ", "import ", "function ", "var ", "let ", "const "]
        if any(marker in text for marker in code_markers):
            logger.error(f"Code markers detected after cleaning: {text[:100]}")
            return self._get_emergency_fallback()
        
        # 13. Check if we have a valid response
        if not text or len(text) < 20:
            return self._get_emergency_fallback()
            
        # 14. Final cleanups for common instruct artifacts
        text = re.sub(r'^\d+\.\s+', '', text)  # Remove leading numbering
        text = re.sub(r'^[-*]\s+', '', text)   # Remove leading bullets
        
        # 15. Final check for patterns that should never appear in therapeutic responses
        if "# YOUR CODE HERE" in text or "SOLUTION:" in text or "Answer the following:" in text:
            logger.error("Critical pattern detected after cleaning")
            return self._get_emergency_fallback()
        
        return text

    def _get_targeted_fallback_response(self, original_text: str) -> str:
        """
        Generate a fallback response tailored to the user's query.
        This ensures we still provide value even if the main response failed.
        """
        # Check for specific keywords to provide targeted responses
        if "depress" in original_text.lower():
            return "I understand you're feeling depressed. These feelings can be incredibly heavy and make everything seem more difficult. It's important to know that depression is a real condition that can affect anyone, and you deserve support. Would you like to share more about what you've been experiencing recently? I'm here to listen without judgment."
        
        elif "anxi" in original_text.lower():
            return "I can hear that anxiety is affecting you right now. Anxiety can feel overwhelming, with racing thoughts and physical sensations that are hard to manage. Remember that your feelings are valid, and many people experience anxiety. Would it help to talk about what triggers these feelings for you? Together we can explore some strategies that might help ease these difficult moments."
        
        elif "trauma" in original_text.lower() or "abuse" in original_text.lower():
            return "Thank you for sharing something so difficult with me. Experiences of trauma or abuse can have profound impacts on our wellbeing, and it takes courage to talk about them. Your feelings and reactions are valid responses to what you've been through. Would you feel comfortable telling me a bit more about what support you're looking for right now?"
        
        elif "relationship" in original_text.lower() or "partner" in original_text.lower():
            return "Relationship challenges can be deeply affecting and complex. The connections we form with others are so important to us, which makes difficulties in these relationships particularly painful. I'm here to listen to your experience without judgment. What aspects of your relationship situation are most concerning for you right now?"
        
        elif "work" in original_text.lower() or "job" in original_text.lower():
            return "Work-related stress and challenges can significantly impact our wellbeing, especially considering how much of our time and energy we invest in our professional lives. These feelings are completely valid. Would you like to share more about what's happening in your workplace that's troubling you?"
        
        elif "family" in original_text.lower() or "parent" in original_text.lower():
            return "Family relationships are often complex and deeply emotional. The dynamics formed in our families can affect us profoundly, and navigating challenges within them can be particularly difficult. I'm here to listen and support you. Could you tell me more about what's happening with your family situation?"
        
        elif "alone" in original_text.lower() or "lonely" in original_text.lower():
            return "Feeling lonely or isolated can be incredibly painful. As humans, we have a fundamental need for connection, and when that need isn't met, it can affect us deeply. Your feelings are completely understandable. Would you like to share more about your experience of loneliness and what it's been like for you?"
        
        elif "broke up" in original_text.lower() or "breakup" in original_text.lower() or "ex" in original_text.lower():
            return "I'm sorry to hear about your breakup. The end of a relationship can be incredibly painful and bring up many difficult emotions. It's completely natural to feel a range of emotions right now - sadness, confusion, anger, or even relief mixed with guilt. Would you like to talk more about what you're experiencing during this challenging time?"
        
        # General fallback based on length of original text
        if len(original_text.split()) < 10:
            return "I'm here to listen and support you. Could you share a bit more about what you're experiencing or what's on your mind right now? The more you can tell me, the better I can understand how to help."
        else:
            return "I understand you're going through a difficult time. Your feelings are valid, and I appreciate you sharing them with me. I'd like to understand more about your situation so I can offer better support. Could you tell me more about what you've been experiencing and how it's affecting you?"

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

    def get_toxicity_model(self):
        """Get or initialize toxicity detection model with local model caching."""
        if not hasattr(self, 'toxicity_model') or self.toxicity_model is None:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            # Define toxicity model name
            toxicity_model_name = "facebook/roberta-hate-speech-dynabench-r4-target"
            
            # Get local model path
            model_folder = toxicity_model_name.split('/')[-1]
            local_path = os.path.join(self.MODELS_DIR, model_folder)
            
            # Create models dir if it doesn't exist
            os.makedirs(self.MODELS_DIR, exist_ok=True)
            
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

    def is_toxic(self, text: str) -> bool:
        """Checks if text is toxic."""
        try:
            # Use the pre-initialized Detoxify instance
            results = self.detoxify.predict(text)
            toxic_score = results["toxicity"]
            logger.info(f"Toxicity score: {toxic_score} - Text: {text[:50]}...")
            return toxic_score > 0.8

        except Exception as e:
            logger.error(f"Error during toxicity check: {e}\n{traceback.format_exc()}")
            # Don't block the response on toxicity check failure
            return False

    def get_embedding(self, text: str) -> torch.Tensor:
        """Generates embeddings, loading the language model if needed."""
        # Don't unload the model if we'll need it later
        model_was_loaded = (self.model is not None and self.tokenizer is not None)
        
        if not model_was_loaded:
            self._load_model()  # Only load if not already loaded
            
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            hidden_states = outputs.hidden_states[-1]
            embeddings = hidden_states.mean(dim=1)
            return embeddings
        except Exception as e:
            logger.error(f"Error during embedding generation: {e}")
            return None

    def _final_response_validation(self, response: str) -> str:
        """
        Final validation to catch invalid response patterns before sending to user.
        """
        import re
        
        # Check for code-related patterns that would never be appropriate
        code_patterns = [
            r"# YOUR CODE HERE",
            r"# SOLUTION:",
            r"def [a-z_]+\(",
            r"```python",
            r"function [a-z_]+\(",
            r"@app\.route",
        ]
        
        # Check for instruction leakage patterns
        instruction_patterns = [
            r"Answer the following:",
            r"\d+\.\s+Answer",
            r"Write your response",
            r"In your response",
            r"Please provide",
        ]
        
        # Check for inappropriate patterns
        for pattern in code_patterns + instruction_patterns:
            if re.search(pattern, response):
                logger.error(f"Invalid response detected with pattern: {pattern}")
                return self._get_emergency_fallback()
                
        # Check for extremely short responses
        if len(response.split()) < 10:
            logger.error(f"Response too short: {response}")
            return self._get_emergency_fallback()
            
        return response

    def _get_emergency_fallback(self, question=None) -> str:
        """
        Emergency fallback response based on the question content.
        This ensures users always get an appropriate response even when generation fails.
        
        Args:
            question: The user's original question (if available)
        """
        # If we have the original question, check for critical content
        if question:
            # Check for suicidal ideation and crisis keywords
            crisis_keywords = [
                "kill myself", "suicide", "suicidal", 
                "don't want to live", "dont want to live", 
                "end my life", "ending my life",
                "life is over", "rather be dead", 
                "want to die", "hurt myself", 
                "harm myself", "self harm",
                "no reason to live"
            ]
            
            # Check for breakup with crisis
            breakup_crisis = any(kw in question.lower() for kw in crisis_keywords) and any(
                term in question.lower() for term in ["broke up", "breakup", "left me", "ex", "girlfriend", "boyfriend"]
            )
            
            # Crisis with suicidal thoughts
            if any(kw in question.lower() for kw in crisis_keywords):
                if breakup_crisis:
                    return (
                        "I'm deeply concerned about what you're sharing regarding your breakup and your thoughts about not wanting to live. "
                        "This pain is real, but it's temporary, even though it doesn't feel that way right now. "
                        "\n\n"
                        "Please reach out for immediate support:\n"
                        "• Call the 988 Suicide and Crisis Lifeline (US): Call or text 988\n"
                        "• Text HOME to 741741 to reach the Crisis Text Line\n"
                        "• Call emergency services at 911 (US) or your local emergency number\n"
                        "\n"
                        "These trained professionals can help you through this difficult time. "
                        "You deserve support, and help is available 24/7. Would you consider reaching out to one of these resources right now?"
                    )
                else:
                    return (
                        "I'm very concerned about what you've shared. Your life matters, and the pain you're experiencing right now can be addressed with the right support. "
                        "\n\n"
                        "Please reach out for immediate help:\n"
                        "• Call or text 988 to reach the Suicide and Crisis Lifeline (US)\n"
                        "• Text HOME to 741741 for the Crisis Text Line\n"
                        "• Call emergency services (911 in US) or go to your nearest emergency room\n"
                        "\n"
                        "Trained professionals are available 24/7 who can help you through this difficult time. "
                        "Would you be willing to contact one of these resources right now? You don't have to face this alone."
                    )
            
            # Handle relationship breakups (non-crisis)
            elif any(term in question.lower() for term in ["broke up", "breakup", "left me", "ex", "girlfriend", "boyfriend"]):
                return (
                    "I'm sorry to hear about your breakup. Ending relationships can bring intense emotions - sadness, anger, confusion, and grief. "
                    "These feelings are a natural response to loss, and it's important to acknowledge them. "
                    "While it might not feel like it now, these feelings will gradually change over time. "
                    "\n\n"
                    "Would you like to share more about what you're going through? I'm here to listen and support you through this difficult time."
                )
            
            # Handle depression/sadness
            elif any(term in question.lower() for term in ["depress", "sad", "down", "hopeless", "empty"]):
                return (
                    "I can hear that you're feeling down right now. Depression and sadness can feel overwhelming and make everything seem more difficult. "
                    "Your feelings are valid, and many people experience similar struggles. "
                    "\n\n"
                    "Would you like to talk more about what you've been experiencing? I'm here to listen without judgment, and together we can explore ways to help you feel better."
                )
        
        # Default supportive response if we can't determine the content or don't have the original question
        return (
            "I'm here to support you. It sounds like you might be going through a challenging time, and I want you to know that "
            "your feelings are valid. Would you feel comfortable sharing more about what's on your mind? I'm here to listen and help."
        )

    def generate_therapeutic_response(self, user_question: str, template_name: str, 
                                      context: Dict[str, Any], 
                                      conversation_history: Optional[List[Dict]] = None) -> str:
        """
        Generate a therapeutic response using a specific template.
        
        Args:
            user_question: The user's question
            template_name: Name of the template to use
            context: Context for generation
            conversation_history: Optional conversation history
            
        Returns:
            str: Generated response
        """
        try:
            # Load model if not already loaded
            if not hasattr(self, 'model') or self.model is None:
                self._load_model()
            
            # Get the template using the existing _load_template method
            template = self._load_template(template_name)
            
            # Add user_question to context if not present
            if 'user_question' not in context:
                context['user_question'] = user_question
                
            # Add conversation history to context if available
            if conversation_history and 'conversation_history' not in context:
                context['conversation_history'] = conversation_history
            
            # Create enhanced_context structure if not present
            if 'enhanced_context' not in context:
                enhanced_context = {
                    "has_knowledge": False,
                    "knowledge_context": "",
                    "has_conversation": False,
                    "conversation_context": "",
                    "psychological_context": {
                        "emotional_signals": [],
                        "pain_point": None
                    }
                }
                
                # Add knowledge context if available
                if 'knowledge_context' in context:
                    enhanced_context["has_knowledge"] = True
                    enhanced_context["knowledge_context"] = context.get("knowledge_context", "")
                    
                # Add conversation context if available
                if conversation_history:
                    enhanced_context["has_conversation"] = True
                    # Format conversation history as text
                    conv_text = ""
                    for item in conversation_history[-3:]:  # Last 3 exchanges
                        if isinstance(item, dict):
                            q = item.get("questionText", item.get("question", ""))
                            a = item.get("answerText", item.get("answer", ""))
                            if q and a:
                                conv_text += f"User: {q}\nAssistant: {a}\n\n"
                    enhanced_context["conversation_context"] = conv_text.strip()
                    
                # Add psychological context if available
                if 'emotional_signals' in context:
                    enhanced_context["psychological_context"]["emotional_signals"] = context.get("emotional_signals", [])
                    
                if 'pain_point' in context:
                    enhanced_context["psychological_context"]["pain_point"] = context.get("pain_point")
                    
                # Add enhanced_context to the main context
                context['enhanced_context'] = enhanced_context
                
            # Render the template
            prompt = template.render(**context)
            logger.info(len(prompt))
            logger.info(f"Template rendered prompt length: {len(prompt)} chars")
            
            # UPDATED: Increased the character limit to 2048 from 1000
            if len(prompt) > 2048:
                logger.warning(f"Prompt exceeds 2048 chars, truncating")
                # Truncate the prompt but preserve important parts
                prompt_parts = prompt.split("\n\n")
                essential_parts = []
                current_length = 0
                
                # Always include the first part (instructions) and the user question
                essential_parts.append(prompt_parts[0])  # Instructions
                current_length += len(prompt_parts[0])
                
                # Find and include the user's question
                for part in prompt_parts:
                    if "USER'S CURRENT MESSAGE:" in part:
                        essential_parts.append(part)
                        current_length += len(part)
                        break
                
                # Add remaining parts until we approach the limit
                for part in prompt_parts[1:]:
                    if "USER'S CURRENT MESSAGE:" in part:
                        continue  # Already added
                        
                    if current_length + len(part) + 10 < 2048:  # Leave a small buffer
                        essential_parts.append(part)
                        current_length += len(part) + 2  # +2 for the newlines
                    else:
                        # We're out of space
                        break
                
                # Combine the essential parts back into a prompt
                prompt = "\n\n".join(essential_parts)
                logger.info(f"Truncated prompt length: {len(prompt)} chars")
                
            # Generate response
            logger.info(f"Generating response for prompt with {len(prompt)} characters")
            
            # Convert the prompt to tokens for a more accurate length assessment
            try:
                input_tokens = len(self.tokenizer.encode(prompt))
                logger.info(f"Tokenized input length: {input_tokens} tokens")
            except Exception as e:
                logger.warning(f"Could not determine token length: {e}")
            
            # Sampling produces more natural and varied responses, which is preferable for therapeutic
            # conversations where slight variations in phrasing can have significant impact.
            generation_kwargs = {
                'max_new_tokens': 512,
                'temperature': 0.7,
                'top_p': 0.9,
                'repetition_penalty': 1.15,
                'do_sample': True  # Enable sampling to use temperature and top_p
            }
            
            # Add GPU memory optimizations if using CUDA
            if self.device == "cuda":
                # These options help with limited GPU memory (6GB)
                generation_kwargs.update({
                    # Use fp16 for faster generation with less memory
                    'torch_dtype': torch.float16,
                    # Efficiently reuse key/value cache for attention
                    'use_cache': True,
                    # Don't keep unnecessary activations in memory
                    'no_repeat_ngram_size': 3,
                    # Aggressive memory cleanup during generation
                    'clean_up_tokenization_spaces': True,
                    # Reduce memory usage at expense of slightly slower processing
                    'max_length': input_tokens + 512  # Limit total sequence length
                })
                
                # For very limited GPU memory, add this line:
                if torch.cuda.get_device_properties(0).total_memory < 8e9:  # Less than 8GB
                    logger.info("Using low memory optimizations for small GPU")
                    # Force garbage collection between generations
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
            
            # Generate response
            outputs = self.model.generate(
                self.tokenizer.encode(prompt, return_tensors="pt").to(self.device),
                **generation_kwargs
            )
            
            # Decode the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Post-process to extract just the generated part
            response = response[len(prompt):].strip()
            
            # Validate the response
            if len(response) < 10:
                logger.warning("Generated response too short, retrying...")
                # Try again with different parameters
                outputs = self.model.generate(
                    self.tokenizer.encode(prompt, return_tensors="pt").to(self.device),
                    max_new_tokens=512,
                    temperature=0.8,
                    repetition_penalty=1.2,
                    top_k=40,
                    do_sample=True
                )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
            
            # Second validation
            if len(response) > 50:
                logger.info("Generated valid response with appropriate length")
            
            # MEMORY OPTIMIZATION: Always unload model after generation to free memory for database operations
            if self.device == "cuda" and torch.cuda.is_available():
                # Check GPU memory usage - for logging only
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved(0) / 1024**3    # GB
                
                logger.info(f"Memory usage: allocated={memory_allocated:.2f}GB, reserved={memory_reserved:.2f}GB")
                
                # We're done with the model for now, so unload it regardless of memory usage
                # This will free memory for subsequent database operations and new template loading
                logger.info("Unloading model after generation to free memory for next operations")
                self._unload_model()
            
            return response
        except Exception as e:
            logger.error(f"Error generating therapeutic response: {e}")
            logger.error(traceback.format_exc())
            
            # MEMORY RECOVERY: Attempt to recover memory on error
            if hasattr(self, 'model') and self.model is not None:
                logger.info("Unloading model to recover memory after error")
                self._unload_model()
                
            return "I apologize, but I encountered an error while generating a response. Could you try rephrasing your question?"

    def _get_breakup_recovery_steps(self) -> str:
        """Provides concrete steps for breakup recovery."""
        return (
            "I understand how difficult breakups can be. Here are some concrete steps that might help you begin healing:\n\n"
            
            "1. Allow yourself to feel: Give yourself permission to experience all your emotions without judgment. Crying, journaling, or talking with trusted friends can help process these feelings.\n\n"
            
            "2. Establish healthy boundaries: Consider limiting contact with your ex for a while to give yourself space to heal. This might include muting social media or asking mutual friends not to share updates.\n\n"
            
            "3. Create a self-care routine: Focus on basic needs like regular sleep, nutritious meals, and physical movement. Even light exercise can boost your mood through endorphin release.\n\n"
            
            "4. Reconnect with yourself: Breakups can be an opportunity to rediscover parts of yourself. Try revisiting old hobbies or exploring new interests that bring you joy.\n\n"
            
            "5. Seek support: Connect with friends, family, or consider talking with a therapist who can provide professional guidance tailored to your situation.\n\n"
            
            "Remember that healing isn't linear - some days will be better than others. What aspects of these suggestions feel most helpful for your situation right now?"
        )

    def _get_crisis_response(self, question: str = "") -> str:
        """
        Returns a crisis response when a true emergency is detected.
        We keep this hardcoded for safety.
        """
        # This is one of the few places where hardcoded responses make sense
        return (
            "I'm concerned about what you've shared. If you're having thoughts of harming yourself, "
            "please reach out for immediate support from trained professionals who can help:\n\n"
            "• Call or text 988 to reach the Suicide and Crisis Lifeline (US)\n"
            "• Text HOME to 741741 for the Crisis Text Line\n"
            "• Call emergency services (911 in US) or go to your nearest emergency room\n\n"
            "Your life matters, and these difficult feelings can improve with proper support. "
            "Would you be willing to reach out to one of these resources right now?"
        )

    def _get_fallback_response(self) -> str:
        """
        A simple general fallback when we can't generate a proper response.
        """
        import random
        
        # More natural variations to avoid repetition
        fallbacks = [
            "I'm here to listen and support you. Could you tell me more about what's on your mind?",
            "Thank you for sharing that with me. I'd like to understand more about what you're experiencing.",
            "I appreciate you opening up. Would it help to explore these feelings a bit more?",
            "Your experiences and feelings are important. I'm here to listen if you'd like to share more.",
            "I'd like to understand better what you're going through. Would you feel comfortable elaborating?"
        ]
        return random.choice(fallbacks)

    def _is_crisis_situation(self, text: str) -> bool:
        """
        Simple check to detect if a user message contains suicidal or serious crisis content.
        
        Args:
            text: The user's message
            
        Returns:
            bool: True if crisis is detected, False otherwise
        """
        # Core crisis keywords - kept deliberately focused
        crisis_keywords = [
            "kill myself", "suicide", "suicidal", 
            "don't want to live", "dont want to live", 
            "end my life", "ending my life",
            "life is over", "rather be dead", 
            "want to die", "hurt myself", 
            "harm myself"
        ]
        
        # Simple check if any keyword appears in the user's message
        return any(keyword in text.lower() for keyword in crisis_keywords)

    def _final_validation(self, response, question=None):
        """
        A last-chance validation to catch any instructions or inappropriate content
        before it reaches the user.
        """
        # Check for instructional patterns that should NEVER be sent to users
        instruction_indicators = [
            "remember that", "the key is to", "your goal is to", 
            "make sure to", "don't forget to", "when responding", 
            "your task is", "important to note", "the approach here"
        ]
        
        if any(indicator in response.lower() for indicator in instruction_indicators):
            logger.critical("INSTRUCTION LEAK DETECTED in final response!")
            if question and any(kw in question.lower() for kw in ["suicide", "kill myself", "dont want to live"]):
                return self._get_crisis_response(question)
            return self._get_fallback_response()
        
        # If no instructions found, return the original
        return response

    def _clean_response(self, response: str, question: Optional[str] = None) -> str:
        """
        Less aggressive cleaning method to preserve valid therapeutic responses.
        """
        import re
        try:
            # DEBUGGING - log raw response to understand what's being generated
            logger.debug(f"Raw response before cleaning: {response[:100]}...")
            
            # STEP 1: CHECK FOR SERIOUS ISSUES REQUIRING IMMEDIATE FALLBACK
            critical_patterns = [
                "# YOUR CODE HERE", "# SOLUTION:", "```python", "```javascript", "def ", "class ", "function "
            ]
            
            if any(pattern in response for pattern in critical_patterns):
                logger.critical(f"Critical pattern detected in response")
                return self._get_emergency_fallback(question)
                
            # STEP 1.5: CHECK FOR EDUCATIONAL/LECTURE CONTENT LEAKAGE
            educational_markers = [
                "Title:", "Chapter:", "Section:", "Introduction:", "Welcome to",
                "In this section", "we will explore", "we will delve into",
                "Let's embark on", "course", "module"
            ]

            if any(marker in response for marker in educational_markers):
                logger.critical("EDUCATIONAL TEMPLATE DETECTED")
                return self._get_targeted_fallback_response(question if question else "")
            
            # STEP 2: EXTRACT DIRECT RESPONSES IF QUOTED
            instruction_prefixes = [
                r"Your therapeutic response should be:\s*[\"'](.+)[\"']",
                r"Your response should be:\s*[\"'](.+)[\"']",
                r"Your response:\s*[\"'](.+)[\"']",
                r"Respond with:\s*[\"'](.+)[\"']",
                r"Here's a helpful response:\s*[\"']?(.+?)[\"']?(?=\n\n|$)"
            ]
            
            for pattern in instruction_prefixes:
                match = re.search(pattern, response, re.DOTALL)
                if match:
                    extracted_text = match.group(1).strip()
                    logger.info(f"Found direct instruction pattern, extracting content")
                    response = extracted_text
                    break
            
            # STEP 3: REMOVE COMMON INSTRUCTION SECTIONS - LESS AGGRESSIVE
            sections_to_remove = [
                r"Instructions:.*?(?=\n\n|$)",
                r"USER QUESTION:.*?(?=\n|$)",
                r"THERAPEUTIC APPROACH:.*?(?=\n|$)",
                r"RESPONSE \(keep.*?(?=\n|$)",
                r"PREVIOUS CONVERSATION:.*?(?=\n\n|$)",
                r"RELEVANT KNOWLEDGE:.*?(?=\n\n|$)",
            ]
            
            for pattern in sections_to_remove:
                response = re.sub(pattern, '', response, flags=re.DOTALL|re.IGNORECASE)
            
            # STEP 4: HANDLE DIALOGUE FORMAT - EXTRACT THERAPEUTIC CONTENT
            dialogue_patterns = [
                r'(?:Therapist|Assistant|Counselor): "?([^"]+)"?', 
            ]
            
            for pattern in dialogue_patterns:
                therapist_responses = re.findall(pattern, response, re.IGNORECASE)
                if therapist_responses and len(therapist_responses[-1]) > 30:
                    response = therapist_responses[-1].strip()
                    logger.info(f"Extracted therapist response from dialogue")
                    break
            
            # STEP 5: CLEAN REMAINING STRUCTURAL ELEMENTS
            # Remove role references but be less aggressive
            response = re.sub(r"^As (?:a|your) therapist,?\s+", "", response, flags=re.IGNORECASE)
            response = re.sub(r"^In my role as (?:a|your) therapist,?\s+", "", response, flags=re.IGNORECASE)
            
            # CRITICAL FIX: Less aggressive cleaning for numbered lists and bullet points
            # We want to keep these helpful structures
            
            # Remove leading quote marks
            response = response.strip('"\'')
            
            # STEP 6: FINAL CHECKS
            # If we still have a valid substantial response, use it
            if len(response.strip()) >= 30:
                logger.info("Generated valid response with appropriate length")
                return response.strip()
            
            # If response is too short, use targeted fallback
            logger.warning(f"Response too short after cleaning: {len(response)} chars")
            return self._get_targeted_fallback_response(question if question else "")
            
        except Exception as e:
            logger.error(f"Error cleaning response: {e}")
            return self._get_fallback_response()

    def _load_template(self, template_name):
        """
        Load a template from the templates directory with proper autoescaping.
        
        Args:
            template_name: Name of the template to load
        
        Returns:
            Jinja2 Template object
        """
        try:
            # Set up Jinja environment with expanded template search paths
            template_paths = [
                # Check main templates directory
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates"),
                # Also check parent package directory for templates
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "templates"),
            ]
            
            # Convert template name to lowercase with underscores for filename
            template_filename = template_name.lower().replace(' ', '_')
            logger.info(f"Looking for template '{template_filename}' in paths: {', '.join(template_paths)}")
            
            # Create environment with multiple search paths
            env = Environment(
                loader=FileSystemLoader(template_paths),
                autoescape=select_autoescape(['html', 'xml'])
            )
            
            # Try multiple naming conventions if the default fails
            template = None
            try:
                # Try with exact lowercase name first
                template = env.get_template(f"{template_filename}")
            except:
                try:
                    # Try with .j2 extension
                    template = env.get_template(f"{template_filename}.j2")
                except Exception as e:
                    logger.error(f"Could not find template '{template_filename}': {e}")
                    raise ValueError(f"Could not find template '{template_filename}' in any format")
            
            if template:
                logger.info(f"Successfully loaded template '{template_filename}'")
                return template
            
        except Exception as e:
            logger.error(f"Error loading template {template_name}: {e}")
            # Return a simple default template string as fallback
            from jinja2 import Template
            logger.warning(f"Using fallback template for '{template_name}'")
            fallback_template = Template("You are a compassionate AI therapist. The user said: '{{user_question}}'. Provide a supportive response.")
            return fallback_template

    def _unload_model(self):
        """Unload the model to free memory."""
        logger.info("Unloading model to free memory")
        try:
            # Delete model
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
                
            # Delete tokenizer too
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
                
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Run garbage collection
            import gc
            gc.collect()
            
            logger.info("Model unloaded successfully")
        except Exception as e:
            logger.error(f"Error unloading model: {e}")

    def generate_therapeutic_response_with_dynamic_retrieval(self, user_question: str, 
                                                             template_name: str, 
                                                             context: Dict[str, Any], 
                                                             conversation_history: List[Dict] = None
                                                             ) -> str:
        """Generate a therapeutic response with dynamic database retrieval during generation."""
        try:
            logger.info(f"Starting dynamic RAG generation with template: {template_name}")
            
            # First check if we have dynamic retrieval capability
            if not context.get('use_dynamic_retrieval') or 'dynamic_retriever' not in context:
                logger.warning("Dynamic retrieval requested but not properly configured")
                # Fall back to standard generation
                return self.generate_therapeutic_response(user_question, template_name, context, conversation_history)
            
            # Get the retriever object
            retriever: DynamicRAGRetriever = context['dynamic_retriever']
            
            # Use PromptSelector to analyze the question and extract psychological topics
            # This gives us better topic detection than hardcoded keyword matching
            prompt_selector = PromptSelector(generator=self)
            
            # Get detailed analysis of the question
            question_analysis = prompt_selector._analyze_question(user_question) 
            detected_topic = question_analysis.get('topic', 'general')
            emotion = question_analysis.get('emotion')
            
            # Generate categories from the user's question
            category_info = prompt_selector.generate_category_info(user_question)
            category_names = list(category_info.keys()) if category_info else []
            
            logger.info(f"PromptSelector analysis: Topic={detected_topic}, Emotion={emotion}")
            logger.info(f"Detected categories: {category_names}")
            
            # Extract specific psychological topics for dynamic retrieval
            extracted_topics = []
            
            # Primary topic from question analysis
            if detected_topic and detected_topic != "general":
                extracted_topics.append(detected_topic.replace('_', ' '))
                
            # Add topics from categories (up to 2 total)
            for category in category_names[:2]:
                # Map category names to search terms
                if category == "Empathy and Validation":
                    if "depression" not in extracted_topics:
                        extracted_topics.append("depression")
                elif category == "Affirmation and Reassurance":
                    if "anxiety" not in extracted_topics:
                        extracted_topics.append("anxiety")
                elif category == "Trauma":
                    if "trauma" not in extracted_topics:
                        extracted_topics.append("trauma")
                elif "CBT" in category:
                    if "cognitive behavioral therapy" not in extracted_topics:
                        extracted_topics.append("cognitive behavioral therapy")
            
            # Set emotion as a topic if appropriate
            if emotion and emotion not in ["confusion", "surprise"]:
                extracted_topics.append(emotion)
                    
            # Ensure we have at least one topic
            if not extracted_topics:
                # Use the detected topic or a fallback
                topic_from_text = detected_topic if detected_topic != "general" else "therapeutic support"
                extracted_topics.append(topic_from_text)
            
            # Limit to top 3 topics
            extracted_topics = extracted_topics[:3]
            logger.info(f"Extracted topics for RAG retrieval: {extracted_topics}")

            # Extract therapeutic approach from pain points if available
            therapeutic_approach = None
            if 'pain_point' in context and context.get('pain_point', {}).get('detected', False):
                # Get the approach_type from the pain point
                approach_type = context.get('pain_point', {}).get('suggested_approach', {}).get('approach_type')
                
                # Map the approach_type to a therapeutic template name using the utility function
                if approach_type:
                    therapeutic_approach = map_approach_to_template(approach_type)
                    logger.info(f"Using therapeutic approach '{therapeutic_approach}' from pain point approach type '{approach_type}'")

            # Add psychological context to the template context
            template_context = context.copy()
            template_context['extracted_topics'] = extracted_topics
            template_context['psychological_context'] = {
                'topic': detected_topic,
                'emotion': emotion,
                'categories': category_names
            }

            # Add the therapeutic_approach if we have one
            if therapeutic_approach:
                template_context['therapeutic_approach'] = therapeutic_approach
            
            # Load the template
            try:
                template = self._load_template(template_name)
                logger.info(f"Template '{template_name}' loaded successfully")
            except Exception as template_error:
                logger.error(f"Error loading template '{template_name}': {template_error}")
                # Fall back to a basic template
                import jinja2
                template = jinja2.Template("You are a therapeutic AI assistant. USER QUESTION: {{ user_question }}")
                
            # Define dynamic retrieval functions with topic validation
            def query_knowledge(topic_query):
                try:
                    # Validate topic_query against "topic" placeholder
                    if topic_query.lower() in ["topic", "specific topic", "the topic"]:
                        logger.warning(f"Detected placeholder 'topic' - replacing with extracted topic")
                        # Use our pre-extracted topics instead of placeholder
                        if extracted_topics:
                            topic_query = extracted_topics[0]
                        else:
                            return "\nPlease specify a concrete psychological concept to search for.\n"
                    
                    logger.info(f"Dynamic knowledge retrieval for: {topic_query}")
                    result = retriever.get_knowledge_by_query(topic_query, limit=2)
                    return f"\nRelevant knowledge about '{topic_query}':\n{result if result else 'No specific information found.'}\n"
                except Exception as e:
                    logger.error(f"Error in query_knowledge: {e}")
                    return f"\nAttempted to retrieve knowledge about '{topic_query}', but encountered an error.\n"
                    
            def query_history(topic_query):
                try:
                    # Validate topic_query against "topic" placeholder
                    if topic_query.lower() in ["topic", "specific topic", "the topic"]:
                        logger.warning(f"Detected placeholder 'topic' - replacing with extracted topic")
                        # Use our pre-extracted topics instead of placeholder
                        if extracted_topics:
                            topic_query = extracted_topics[0]
                        else:
                            return "\nPlease specify a concrete conversation topic to search for.\n"
                            
                    logger.info(f"Dynamic history retrieval for: {topic_query}")
                    result = retriever.get_past_interactions(topic_query)
                    return f"\nRelevant conversation history about '{topic_query}':\n{result if result else 'No past conversations on this topic.'}\n"
                except Exception as e:
                    logger.error(f"Error in query_history: {e}")
                    return f"\nAttempted to retrieve conversation history about '{topic_query}', but encountered an error.\n"
            
            def get_pain_point():
                try:
                    logger.info(f"Dynamic pain point retrieval")
                    result = retriever.get_pain_point()
                    if result and result.get('pain_point'):
                        return f"\nDetected recurring theme: {result.get('pain_point')}\n"
                    return "\nNo specific recurring themes detected.\n"
                except Exception as e:
                    logger.error(f"Error in get_pain_point: {e}")
                    return "\nAttempted to retrieve pain points, but encountered an error.\n"
            
            # Add the functions to the template context
            template_context['query_knowledge'] = query_knowledge
            template_context['query_history'] = query_history
            template_context['get_pain_point'] = get_pain_point
            
            # Add empty conversation history if none provided
            if 'conversation_history' not in template_context and conversation_history:
                template_context['conversation_history'] = conversation_history
                
            # Pre-fill the template with examples using the detected topics
            if extracted_topics:
                # Create a pre-retrieved section
                pre_retrieved_info = {}
                
                # Get the knowledge for the first topic
                if extracted_topics:
                    try:
                        first_topic = extracted_topics[0]
                        kb_info = retriever.get_knowledge_by_query(first_topic, limit=2)
                        if kb_info and len(kb_info) > 20:
                            pre_retrieved_info[first_topic] = kb_info
                    except Exception as e:
                        logger.warning(f"Error pre-retrieving knowledge: {e}")
                
                # Add the pre-retrieved info to the context
                template_context['pre_retrieved_info'] = pre_retrieved_info
                logger.info(f"Added pre-retrieved info for topics: {list(pre_retrieved_info.keys())}")
                
            # Render the template
            try:
                logger.info("Rendering template with context")
                logger.debug(f"Template context keys: {list(template_context.keys())}")
                prompt = template.render(**template_context)
                
                # Add token count check after rendering
                token_count = len(self.tokenizer.encode(prompt))
                max_context_tokens = 2048  # Set model's context window size
                
                logger.info(f"Template rendered successfully, length: {len(prompt)} chars ({token_count} tokens)")
                
                # Check if prompt exceeds token limit
                if token_count > max_context_tokens:
                    logger.warning(f"Prompt exceeds token limit ({token_count} > {max_context_tokens})")
                    # Truncate the prompt but preserve important parts
                    prompt_parts = prompt.split("\n\n")
                    essential_parts = []
                    current_length = 0
                    
                    # Always include the first part (instructions) and the user question
                    essential_parts.append(prompt_parts[0])  # Instructions
                    current_length += len(prompt_parts[0])
                    
                    # Find and include the user's question
                    for part in prompt_parts:
                        if "USER'S CURRENT MESSAGE:" in part:
                            essential_parts.append(part)
                            current_length += len(part)
                            break
                    
                    # Add remaining parts until we approach the limit
                    for part in prompt_parts[1:]:
                        if "USER'S CURRENT MESSAGE:" in part:
                            continue  # Already added
                            
                        if current_length + len(part) + 10 < max_context_tokens:  # Leave a small buffer
                            essential_parts.append(part)
                            current_length += len(part) + 2  # +2 for the newlines
                        else:
                            # We're out of space
                            break
                    
                    # Combine the essential parts back into a prompt
                    prompt = "\n\n".join(essential_parts)
                    logger.info(f"Truncated prompt length: {len(prompt)} chars")
                    
            except Exception as render_error:
                logger.error(f"Error rendering template: {render_error}")
                # Fall back to a basic prompt
                prompt = f"You are a therapeutic AI assistant. The user asks: {user_question}"
                
            # Generate text with the rendered prompt
            try:
                logger.info("Generating text with rendered prompt")
                response = self.generate_text(prompt, max_new_tokens=512, temperature=0.7, top_p=0.9)
                if not response:
                    logger.error("Text generator returned empty response!")
                    # Provide a fallback response based on the detected topic
                    return f"I understand that {extracted_topics[0] if extracted_topics else 'your concern'} can be challenging. Could you tell me more about what you're experiencing?"
                else:
                    logger.info(f"Generated response of length {len(response)}")
                    logger.debug(f"First 100 chars of response: {response[:100]}")
            except Exception as gen_error:
                logger.error(f"Error generating text: {gen_error}")
                return "I apologize, but I'm having trouble generating a response right now."
            
            # Clean the response
            try:
                cleaned_response = self._clean_therapeutic_response(response)
                logger.info(f"Cleaned response, final length: {len(cleaned_response)}")
                return cleaned_response
            except Exception as clean_error:
                logger.error(f"Error cleaning response: {clean_error}")
                return response  # Return uncleaned response if cleaning fails
                
        except Exception as e:
            logger.error(f"Error in dynamic RAG generation: {e}")
            logger.error(traceback.format_exc())
            return "I apologize, but I encountered an error while processing your question. Could you please try again?"
