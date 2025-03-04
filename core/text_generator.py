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
        """Generates text based on the provided prompt with enhanced error handling."""
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
                    do_sample=False,
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

    def _get_emergency_fallback(self) -> str:
        """
        Emergency fallback response when all else fails.
        This ensures users always get something helpful even in worst-case scenarios.
        """
        return ("I understand you're having difficulty with asking questions and feel stuck when you need to inquire about "
            "something. Many people struggle with this too. It can feel uncomfortable to put ourselves out there or worry "
            "about how our questions might be received. Would you like to explore what makes asking questions challenging "
            "for you? I'm here to listen and support you.")

    def generate_therapeutic_response(self, question: str, template_name: str, 
                                enhanced_context: Dict[str, Any], 
                                previous_conversation: Optional[List[Dict[str, str]]] = None) -> str:
        try:
            # Extract rich information from enhanced context
            topic = enhanced_context.get("detected_topic", "emotional_support")
            confidence = enhanced_context.get("confidence", 0.5)
            urgency = enhanced_context.get("urgency_level", "normal")
            emotional_tone = enhanced_context.get("emotional_tone", ["neutral"])
            keywords_matched = enhanced_context.get("keywords_matched", [])
            
            # Log the exact template being used
            if template_name not in self.prompt_templates:
                logger.error(f"Template {template_name} not found in prompt_templates")
                template_name = "Empathy and Validation"  # Default fallback
                
            template = self.prompt_templates[template_name]
            
            # Log the template and formatted prompt for debugging
            logger.debug(f"Using template: {template_name}")
            logger.debug(f"Raw template: {template[:100]}...")
            
            # Log detailed generation parameters
            logger.info(f"Generating response for topic: {topic}, template: {template_name}, " 
                    f"urgency: {urgency}, confidence: {confidence:.2f}")
            
            # Adapt template based on confidence level and urgency
            if urgency == "high":
                # Crisis-oriented template for urgent situations
                modified_template = """
    You are responding to someone who may be in emotional distress or crisis. They mentioned: "{question}"

    Focus on safety, validation, and immediate support. Be calm, direct, and compassionate.
    If there are any signs of self-harm or danger, emphasize getting immediate professional help.

    Your response should be:
    1. Validating their feelings without judgment
    2. Offering immediate coping strategies if appropriate
    3. Encouraging professional support
    4. Warm, present, and human

    Write ONLY your therapeutic response without preamble, instructions, or metadiscussion.
    """
            elif confidence < 0.4:
                # More general approach for low confidence situations
                modified_template = """
    You are a compassionate therapist responding to someone who mentioned: "{question}"

    Since I'm not entirely certain about their specific needs, focus on:
    1. Validating their experience without making assumptions
    2. Offering general emotional support
    3. Asking thoughtful questions to understand more
    4. Being warm and genuine

    Topic area seems to be related to {topic}, but keep your approach flexible.

    Write ONLY your therapeutic response without labels, instructions, or markers.
    """
            else:
                # Detailed template for high-confidence situations
                modified_template = """
    You are a skilled therapist using {template_name} techniques to respond to: "{question}"

    Therapeutic approach: Focus on {topic} with these emotional tones present: {emotional_tone}.
    Context: The person appears to be discussing issues related to {keywords_context}.

    Your response should:
    1. Validate their emotions and experiences authentically
    2. Use principles from {template_name} without naming the technique
    3. Address the specific {topic} concerns they've expressed
    4. Maintain a warm, supportive tone while being genuine

    If previous conversation exists, maintain continuity with what was discussed before.

    Write ONLY your therapeutic response without any meta-commentary, instructions, or markers.
    """
            
            # Format keywords context for better prompt specificity
            keywords_context = ", ".join(keywords_matched[:5]) if keywords_matched else topic
            
            # Format with the essential information
            try:
                prompt = modified_template.format(
                    question=question, 
                    topic=topic,
                    template_name=template_name,
                    urgency=urgency,
                    emotional_tone=", ".join(emotional_tone),
                    keywords_context=keywords_context
                )
                logger.debug(f"Formatted prompt: {prompt[:100]}...")
            except KeyError as e:
                logger.error(f"Error formatting prompt: {e}")
                # Use emergency template
                prompt = f"You are a therapist. The user said: {question}. Respond with empathy."
            
            # Add relevant documents if available
            relevant_docs = enhanced_context.get("relevant_documents", [])
            if relevant_docs:
                docs_summary = "\n\nRelevant information:\n"
                for i, doc in enumerate(relevant_docs[:3]):
                    # Limit each document to 300 characters to avoid overwhelming the context
                    docs_summary += f"- {doc[:300]}{'...' if len(doc) > 300 else ''}\n"
                prompt += docs_summary
            
            # Add conversation history if available in a structured format
            if previous_conversation and len(previous_conversation) > 0:
                # Log the exchanges we're adding for debug purposes
                for exchange in previous_conversation[-2:]:
                    logger.info(f"Including conversation context: {exchange.get('user_message', '')[:50]}...")
                    
                # Create context string using existing data format
                conversation_context = "Previous conversation context:\n"
                for exchange in previous_conversation[-2:]:
                    if 'user_message' in exchange and 'ai_message' in exchange:
                        conversation_context += f"Person: {exchange.get('user_message', '')}\n"
                        conversation_context += f"You: {exchange.get('ai_message', '')}\n"
                
                # Add to prompt
                prompt = conversation_context + "\n\nCurrent query: " + question + "\n\n" + prompt
            
            # Special handling for relationship breakup - ensure empathetic approach
            if "broke up" in question.lower() or "breakup" in question.lower() or "ex " in question.lower():
                if "depress" in question.lower() or "sad" in question.lower():
                    # Add specific guidance for breakup+depression scenario
                    breakup_guidance = "\n\nImportant context: The person is dealing with both a breakup and depression. " + \
                                    "Make sure to validate their grief process and feelings of loss. Acknowledge " + \
                                    "that the relationship ending can both cause and worsen depression. " + \
                                    "Focus on empathy first, avoid clichés like 'plenty of fish in the sea'."
                    prompt += breakup_guidance
            
            # Generate response with carefully tuned parameters
            response = self.generate_text(
                prompt, 
                max_new_tokens=300,
                temperature=0.75 if confidence > 0.6 else 0.7,  # Slightly higher temp for high confidence
                top_p=0.92
            )
            
            # Validate the response isn't empty or default error text
            if not response or len(response.strip()) < 5:
                logger.error("Empty response received from model")
                return self._get_emergency_fallback()
            
            # Clean with specialized therapeutic cleaning method
            cleaned_response = self._clean_therapeutic_response(response)
            
            # Extra check for remaining template instructions or formatting issues
            if "**depression**:" in cleaned_response.lower() or "since the topic is" in cleaned_response.lower():
                logger.warning("Detected template echoing, regenerating response")
                # More direct prompt for regeneration
                emergency_prompt = f"As a therapist, respond compassionately to: '{question}' " + \
                                f"The person is dealing with {topic}. Be warm and supportive."
                response = self.generate_text(emergency_prompt, max_new_tokens=250, temperature=0.7)
                cleaned_response = self._clean_therapeutic_response(response)
            
            # Final safety check for hallucinated dialogues
            if "Person:" in cleaned_response or "Client:" in cleaned_response:
                cleaned_response = re.sub(r"(Person|Client|User):.+?(?=\n|$)", "", cleaned_response)
                cleaned_response = cleaned_response.strip()
            
            # Final verification of response quality
            if len(cleaned_response.split()) < 15 or "I apologize" in cleaned_response:
                logger.warning("Low quality response detected, using fallback")
                cleaned_response = self._get_targeted_fallback_response(question)
            
            # Final safety validation before returning to user
            validated_response = self._final_response_validation(cleaned_response)
            
            logger.debug(f"Generated response length: {len(validated_response.split())} words")
            return validated_response
            
        except Exception as e:
            logger.error(f"Error generating therapeutic response: {e}", exc_info=True)
            # Provide a thoughtful fallback based on the topic if possible
            return self._get_emergency_fallback()
