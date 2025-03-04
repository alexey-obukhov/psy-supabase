from typing import Dict
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import logging
import traceback
from typing import Dict, List, Optional, Any
from psy_supabase.utilities.templates.therapeutic_prompt import prompt_templates
from psy_supabase.utilities.text_utils import clean_text

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
        
        # Load the model immediately on initialization
        self._load_model()
        
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
            load_config = {"torch_dtype": torch.bfloat16 if self.use_bfloat16 else torch.float32}
            
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

    def generate_text(self, prompt, max_new_tokens=256, temperature=0.7, top_p=0.9):
        """Generates text based on the provided prompt."""
        try:
            # Ensure model is loaded
            self._ensure_model_loaded()
            
            # Tokenize the input with explicit padding and attention mask
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, 
                                    max_length=1024)
            
            # Make sure inputs are on the correct device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            logger.debug(f"Input shape: {inputs['input_ids'].shape}")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,  # Now using the parameter instead of a hardcoded value
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # Decode the generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (remove the prompt)
            prompt_length = len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
            response = generated_text[prompt_length:].strip()
            
            # If the response is empty, return the full generated text
            if not response:
                response = generated_text.strip()
            
            # Clean the response
            response = self._clean_therapeutic_response(response)
            
            logger.debug(f"Generated response length: {len(response)}")
            return response
        except Exception as e:
            logger.error(f"Error generating text: {e}", exc_info=True)
            return "I apologize, but I'm having trouble processing your request right now."

    def _clean_therapeutic_response(self, text: str) -> str:
        """
        Specialized cleaning for therapeutic responses based on best practices.
        Removes artifacts while preserving therapeutic content.
        """
        import re

        # 1. Check for and remove titles/sections that aren't part of the response
        text = re.sub(r'\n\s*\n\s*\n.*?(Title|Introduction|Chapter|Section|CHAPTER):', '', text, flags=re.DOTALL|re.IGNORECASE)
        
        # 2. Look for multiple consecutive newlines - often signal content boundary
        parts = re.split(r'\n\s*\n\s*\n', text)
        if len(parts) > 1:
            # Keep only the first part (the actual response)
            text = parts[0].strip()
        
        # 3. Apply your existing dialogue cleaning logic
        if re.search(r'(User|Therapist|CLIENT|THERAPIST):', text, re.IGNORECASE):
            try:
                therapist_responses = re.findall(r'(?:Therapist|THERAPIST):\s*(.*?)(?=\n\s*(?:User|CLIENT)|$)', text, re.DOTALL|re.IGNORECASE)
                if therapist_responses:
                    for response in therapist_responses:
                        if len(response.strip()) > 20:
                            return response.strip()
                return self._get_targeted_fallback_response(text)
            except:
                return self._get_targeted_fallback_response(text)
        
        # 4. Apply other existing cleaning
        for marker in ["USER:", "THERAPIST:", "PATIENT:", "CLIENT:", "DOCTOR:"]:
            if marker in text:
                text = text.split(marker)[0]
        
        # 5. Remove question/answer markers that appear in training data
        text = re.sub(r'Question \d+:|Answer:|Response:', '', text)
        
        # New step: Remove exercise instructions and repetition
        text = re.sub(r'Exercise:.*?(?=\n|$)', '', text, flags=re.IGNORECASE|re.DOTALL)
        text = re.sub(r'EXERCISE:.*?(?=\n|$)', '', text, flags=re.IGNORECASE|re.DOTALL)
        
        # Remove duplicated sentences (this handles repetition)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        unique_sentences = []
        for sentence in sentences:
            if sentence.strip() and sentence not in unique_sentences:
                unique_sentences.append(sentence)
        text = ' '.join(unique_sentences)
        
        # 6. Additional cleaning for specific formats
        text = re.sub(r'\b(TR:|THERPST:|THERAP:|THERAPY:)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[.*?\]', '', text)
        
        # 7. Fix line breaks and whitespace
        lines = text.split('\n')
        lines = [line for line in lines if len(line.strip().split()) > 1]
        text = '\n'.join(lines)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\(\s*\)', '', text)
        
        # Add this after your existing steps in _clean_therapeutic_response
        # Simple but effective pattern matching for common artifacts
        patterns_to_remove = [
            r'\b(?:Exercise|EXERCISE):.+?(?=\n|$)', 
            r'\bWrite (?:a|the) (?:response|answer).+?(?=\n|$)',
            r'\bYour response should.+?(?=\n|$)',
            r'\bRespond to the user.+?(?=\n|$)',
            r'\bInstructions:.+?(?=\n|$)',
            r'USER QUESTION:.+?(?=\n|$)',
            r'THERAPEUTIC APPROACH:.+?(?=\n|$)',
            r'RESPONSE \(keep.+?(?=\n|$)'
        ]

        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE|re.DOTALL)

        # Look for remaining instruction markers and truncate
        instruction_markers = [
            "Exercise:", "Instructions:", "Your response:", "Note to AI:",
            "USER QUESTION:", "THERAPEUTIC APPROACH:", 
            "RESPONSE (keep", "PREVIOUS CONVERSATION:", 
            "RELEVANT KNOWLEDGE:"
        ]
        for marker in instruction_markers:
            if marker.lower() in text.lower():
                idx = text.lower().find(marker.lower())
                if idx >= 0:
                    text = text[:idx].strip()
                    break

        # 8. Check if we have a valid response
        if not text or len(text) < 15:
            return self._get_targeted_fallback_response(text)
            
        return text

    def _get_targeted_fallback_response(self, original_text):
        """
        Generate a fallback response tailored to the user's query.
        This ensures we still provide value even if the main response failed.
        """
        # Check for specific keywords to provide targeted responses
        if "depress" in original_text.lower():
            return "I understand you're feeling depressed. This is a challenging emotion to experience. Consider speaking with a mental health professional who can provide personalized support. In the meantime, gentle self-care activities and maintaining social connections can help support your wellbeing."
        # General fallback
        return "I understand you're going through a difficult time. Remember that your feelings are valid, and seeking support is a sign of strength. Consider speaking with a mental health professional who can provide personalized guidance tailored to your specific situation."

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
        """Checks if text is toxic."""
        try:
            # Skip toxicity check for short responses
            if len(text.split()) < 5:
                return False
                
            # Load model if not already loaded
            if self.toxic_tokenizer is None or self.toxic_model is None:
                self._load_toxicity_model()
                
            # Process the text
            inputs = self.toxic_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            
            # Run inference without gradients
            with torch.no_grad():
                # Make sure to specify we're using CPU
                outputs = self.toxic_model(**inputs)
                
            # Get probability of toxic class
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            toxic_score = probs[0, 1].item()  # Assuming index 1 is toxic class
            
            logger.debug(f"Toxicity score: {toxic_score}")
            return toxic_score > 0.7
            
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
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            hidden_states = outputs.hidden_states[-1]
            embeddings = hidden_states.mean(dim=1)
            return embeddings
        except Exception as e:
            logger.error(f"Error during embedding generation: {e}\n{traceback.format_exc()}")
            return None

    def generate_therapeutic_response(self, question: str, template_name: str, 
                                      enhanced_context: Dict[str, Any], 
                                      previous_conversation: Optional[List[Dict[str, str]]] = None) -> str:
        """Generate a therapeutic response using the selected prompt template."""
        try:
            # Get detected topic from enhanced context
            topic = enhanced_context.get("detected_topic", "emotional_support")
            
            # Create a more direct prompt that won't be echoed
            modified_template = """
You are a compassionate therapist responding to someone who mentioned: "{question}"

Topic: {topic} - Focus on this specific aspect of their concern.
Approach: Use {template_name} techniques without mentioning the technique by name.

Write ONLY your response to the person. Don't include instructions, labels, or markers in your response.
Keep your response warm, validating, and focused on the person's feelings.
"""
            
            # Format with the essential information
            prompt = modified_template.format(
                question=question, 
                topic=topic,
                template_name=template_name
            )
            
            # Add conversation history if available - using your existing format
            if previous_conversation and len(previous_conversation) > 0:
                # Log the exchanges we're adding for debug purposes
                for exchange in previous_conversation[-2:]:
                    logger.info(f"Exchange: {exchange}")
                    
                # Create context string using existing data format
                conversation_context = "Previous conversation:\n"
                for exchange in previous_conversation[-2:]:
                    if 'user_message' in exchange and 'ai_message' in exchange:
                        conversation_context += f"Person: {exchange.get('user_message', '')}\n"
                        conversation_context += f"You: {exchange.get('ai_message', '')}\n"
                
                # Add to prompt
                prompt = conversation_context + "\n" + prompt
            
            # Generate response with existing parameters
            response = self.generate_text(
                prompt, 
                max_new_tokens=300,
                temperature=0.75,
                top_p=0.92
            )
            
            # Clean with your existing method
            cleaned_response = self._clean_therapeutic_response(response)
            
            # Extra check: If the response contains clear template instructions, regenerate
            if "**depression**:" in cleaned_response.lower() or "since the topic is" in cleaned_response.lower():
                logger.warning("Detected template echoing, regenerating response")
                # Simplified emergency prompt
                emergency_prompt = f"As a therapist, respond to someone who said: '{question}' " + \
                                  f"They're dealing with {topic}. Be compassionate and warm."
                response = self.generate_text(emergency_prompt, max_new_tokens=200)
                cleaned_response = self._clean_therapeutic_response(response)
            
            return cleaned_response
            
        except Exception as e:
            logger.error(f"Error generating therapeutic response: {e}")
            return f"I hear that you're feeling sad. Depression can make everything seem gray and hopeless. Would you like to talk more about what's been happening?"
