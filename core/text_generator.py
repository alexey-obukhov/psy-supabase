from typing import Dict
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import logging
import traceback
from typing import Dict, List, Optional, Any
from psy_supabase.utilities.templates.therapeutic_prompt import prompt_templates
from psy_supabase.utilities.text_utils import clean_text
import os
from jinja2 import Environment, FileSystemLoader

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
        """Generates text based on the provided prompt with enhanced error handling."""
        try:
            # Ensure model is loaded
            self._ensure_model_loaded()
            
            # Tokenize the input with explicit padding and attention mask
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, 
                                    max_length=1024)

            # Make sure inputs are on the correct device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Log token count to track context usage
            token_count = inputs['input_ids'].shape[1]
            logger.info(f"Tokenized input length: {token_count} tokens")
            
            if token_count >= 1000:  # Almost at limit
                logger.warning(f"Input approaching token limit: {token_count}/1024")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
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
            
            # Pre-clean validation - check for code patterns before regular cleaning
            code_patterns = [
                "# YOUR CODE HERE",
                "# SOLUTION",
                "Answer the following:",
                "```python",
                "```javascript",
                "def ",
                "class ",
                "function "
            ]
            
            if any(pattern in response for pattern in code_patterns):
                logger.error(f"Code pattern detected in raw response: {response[:100]}...")
                return self._get_emergency_fallback()
            
            # Clean the response using the existing method
            response = self._clean_therapeutic_response(response)
            
            # Final validation to prevent inappropriate responses
            if not response or len(response.strip()) < 10:
                logger.error("Response too short after cleaning")
                return self._get_emergency_fallback()
            
            # Double-check for any remaining code patterns that might have survived cleaning
            if any(pattern in response for pattern in code_patterns):
                logger.error(f"Code pattern still present after cleaning: {response[:100]}...")
                return self._get_emergency_fallback()
            
            logger.debug(f"Generated response length: {len(response)}")
            return response
            
        except Exception as e:
            logger.error(f"Error generating text: {e}", exc_info=True)
            return self._get_emergency_fallback()

    def _clean_therapeutic_response(self, text: str) -> str:
        """
        Specialized cleaning for therapeutic responses based on best practices.
        Removes artifacts while preserving therapeutic content.
        """
        import re

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
        ]
        
        for pattern in code_exercise_patterns:
            if re.search(pattern, text, re.IGNORECASE):
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
            inputs = self.toxic_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=1024)
            
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
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            hidden_states = outputs.hidden_states[-1]
            embeddings = hidden_states.mean(dim=1)
            return embeddings
        except Exception as e:
            logger.error(f"Error during embedding generation: {e}\n{traceback.format_exc()}")
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

    def generate_therapeutic_response(self, question: str, template_name: str, 
                        enhanced_context: Dict[str, Any], 
                        previous_conversation: Optional[List[Dict[str, str]]] = None) -> str:
        """Generate a therapeutic response using the appropriate template."""
        try:
            # Load the template
            template = self._load_template("therapeutic_response")
            
            # Fix conversation history formatting if not already in enhanced_context
            if 'has_conversation' not in enhanced_context or not enhanced_context.get('has_conversation'):
                # Format previous_conversation properly
                if previous_conversation and len(previous_conversation) > 0:
                    # Format the previous conversation for the template
                    MAX_CONVERSATION_EXCHANGES = 2  # Only keep most recent exchanges
                    recent_exchanges = previous_conversation[-MAX_CONVERSATION_EXCHANGES:]
                    
                    conversation_parts = []
                    for exchange in recent_exchanges:
                        user_msg = exchange.get('questionText', exchange.get('question', ''))
                        ai_msg = exchange.get('answerText', exchange.get('answer', ''))
                        if user_msg and ai_msg:
                            # Truncate if too long
                            user_short = user_msg[:100] + ("..." if len(user_msg) > 100 else "")
                            ai_short = ai_msg[:150] + ("..." if len(ai_msg) > 150 else "")
                            conversation_parts.append(f"User: {user_short}")
                            conversation_parts.append(f"Assistant: {ai_short}")
                    
                    conversation_context = "\n".join(conversation_parts)
                    
                    # Update the enhanced_context with this conversation
                    enhanced_context['conversation_context'] = conversation_context
                    enhanced_context['has_conversation'] = bool(conversation_parts)
                    
                    logger.info(f"Added {len(recent_exchanges)} conversation exchanges from previous_conversation parameter")
            
            # Enforce hard limits on context size for template
            MAX_TEMPLATE_CONTEXT = 600
            MAX_CONVERSATION_CONTEXT = 300
            
            if enhanced_context.get('knowledge_context', '') and len(enhanced_context['knowledge_context']) > MAX_TEMPLATE_CONTEXT:
                enhanced_context['knowledge_context'] = enhanced_context['knowledge_context'][:MAX_TEMPLATE_CONTEXT] + "..."
                
            if enhanced_context.get('conversation_context', '') and len(enhanced_context['conversation_context']) > MAX_CONVERSATION_CONTEXT:
                enhanced_context['conversation_context'] = enhanced_context['conversation_context'][:MAX_CONVERSATION_CONTEXT] + "..."
            
            # Prepare the template variables
            template_vars = {
                'user_question': question,
                'enhanced_context': enhanced_context
            }
            
            # Render the template
            prompt = template.render(**template_vars)
            
            # Log prompt length
            prompt_length = len(prompt)
            logger.info(f"Template rendered prompt length: {prompt_length} chars")
            
            # Enforce absolute maximum prompt length
            MAX_PROMPT_LENGTH = 1000
            if prompt_length > MAX_PROMPT_LENGTH:
                logger.warning(f"Prompt exceeds {MAX_PROMPT_LENGTH} chars, truncating")
                # Find a good breakpoint to truncate
                end_idx = prompt.rfind('\n\n', 0, MAX_PROMPT_LENGTH)
                if end_idx == -1:
                    end_idx = MAX_PROMPT_LENGTH
                    
                prompt = prompt[:end_idx] + "\n\nIMPORTANT: Provide a compassionate response to the user's current question."
            
            # Generate response with enhanced monitoring
            logger.info(f"Generating response for prompt with {prompt_length} characters")
            
            # Generate response
            response = self.generate_text(
                prompt,
                max_new_tokens=350,  # Standardized token limit
                temperature=0.7
            )
            
            # Clean the response
            cleaned_response = self._clean_response(response, question)
            
            return cleaned_response
        except Exception as e:
            logger.error(f"Error generating therapeutic response: {e}")
            return self._get_fallback_response()

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

    def _clean_response(self, response: str, question: str = None) -> str:
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
        """Load a template from the templates directory."""
        try:
            # Set up Jinja environment
            template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
            env = Environment(loader=FileSystemLoader(template_dir))
            
            # Load the specified template
            template = env.get_template(f"{template_name}.j2")
            return template
        except Exception as e:
            logger.error(f"Error loading template {template_name}: {e}")
            # Return a simple default template as fallback
            return "You are a compassionate AI therapist. The user said: '{{user_question}}'. Provide a supportive response."
