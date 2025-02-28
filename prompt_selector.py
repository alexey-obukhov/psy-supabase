"""
Prompt selector for therapeutic AI responses.
Based on Alexey Obukhov's therapeutic prompt system.
"""
import logging
import traceback
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from utilities.therapeutic_promt import prompt_templates
import json
import spacy

logger = logging.getLogger(__name__)

class PromptSelector:
    """
    Selects the most appropriate therapeutic prompt template based on user input.
    Uses semantic matching and existing prompt templates for optimal responses.
    """
    
    def __init__(self, generator):
        """Initialize the prompt selector with the text generator."""
        self.generator = generator
        self.prompt_templates = prompt_templates
        
        # Load spaCy model for text processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("Downloading en_core_web_sm model...")
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Define keyword associations with prompt templates
        self.keyword_mappings = {
            "Empathy and Validation": [
                "sad", "depressed", "down", "unhappy", "alone", "lonely", "grief", "loss", 
                "hurt", "pain", "suffering", "cry", "tears", "heartbroken"
            ],
            "Affirmation and Reassurance": [
                "anxious", "worried", "stressed", "nervous", "fear", "scared", "panic", 
                "overwhelmed", "frightened", "uneasy", "tense", "afraid"
            ],
            "Providing Suggestions": [
                "help", "advice", "tips", "suggestion", "guidance", "recommend", "strategy", 
                "solution", "fix", "resolve", "approach", "technique", "method", "cope", "handle"
            ],
            "Information": [
                "why", "explain", "understand", "how", "what", "learn", "know", "curious", 
                "information", "research", "fact", "science", "reason", "cause"
            ],
            "Question": [
                "confused", "unsure", "uncertain", "wonder", "think", "feel", "opinion"
            ],
            # Add more mappings as needed
        }
        
        # Define topic keywords
        self.topic_keywords = {
            "Depression": ["sad", "depressed", "hopeless", "empty", "unmotivated", "tired"],
            "Anxiety": ["anxious", "worry", "nervous", "panic", "fear", "stress"],
            "Trauma": ["trauma", "abuse", "flashback", "nightmare", "ptsd", "hurt"],
            "Relationships": ["relationship", "partner", "friend", "family", "marriage", "connect"],
            "Self-esteem": ["confidence", "worth", "value", "failure", "not good enough"],
            "emotional_support": []  # Default topic - changed from "General" to "emotional_support"
        }
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing unwanted characters and normalizing it."""
        import re
        import html
        text = html.unescape(text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»""'']))", "", text)
        text = re.sub(r"\u2019", "'", text)
        text = re.sub(r"\u2014", "-", text)
        text = re.sub(r"\u201c", '"', text)
        text = re.sub(r"\u201d", '"', text)
        text = re.sub(r"\u2026", "...", text)
        text = re.sub(r"[^a-zA-Z0-9\s'\":-]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def tokenize_and_lemmatize(self, text: str) -> str:
        """Tokenize and lemmatize text to extract key concepts."""
        try:
            doc = self.nlp(text)
            cleaned_tokens = [
                token.lemma_.lower() for token in doc
                if not token.is_stop and not token.is_punct and not token.is_space
            ]
            return " ".join(cleaned_tokens).strip()
        except Exception as e:
            logger.error(f"Error in tokenize_and_lemmatize: {e}")
            return text
    
    def select_prompt_template(self, user_question: str) -> Tuple[str, Dict[str, Any]]:
        """
        Select the most appropriate prompt template based on user input.
        
        Args:
            user_question: The user's question or statement
            
        Returns:
            Tuple of (template_name, enhanced_context)
        """
        # Clean and normalize the question
        cleaned_question = self.clean_text(user_question)
        
        # Default to Others template if no specific match is found
        template_name = "Others"
        
        # Check for keyword matches first (fast and effective)
        for template, keywords in self.keyword_mappings.items():
            if any(keyword in cleaned_question.lower() for keyword in keywords):
                template_name = template
                break
        
        # Generate category information using TextGenerator's method
        try:
            category_info = self.generator.generate_category_info(
                question=cleaned_question,
                topic="Therapy"  # General topic for therapeutic context
            )
        except Exception as e:
            logger.error(f"Error generating categories: {e}")
            category_info = {cat: "" for cat in self.prompt_templates.keys()}
        
        # Determine the topic
        topic = self._determine_topic(category_info, cleaned_question)
        
        # Build enhanced context for the prompt
        enhanced_context = {
            "detected_template": template_name,
            "detected_topic": topic,
            "category_info": category_info
        }
        
        logger.info(f"Selected template '{template_name}' for topic '{topic}'")
        return template_name, enhanced_context
    
    def _determine_topic(self, category_info: Dict[str, str], question: str) -> str:
        """Determine the most relevant therapeutic topic."""
        # Check if the question contains keywords for specific topics
        question_lower = question.lower()
        
        # Look for direct topic mentions
        for topic, keywords in self.topic_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                return topic
        
        # Check if CBT category has meaningful content
        if len(category_info.get("Cognitive Behavioral Therapy (CBT)", "")) > 20:
            if any(k in question_lower for k in ["think", "thought", "belief"]):
                return "CBT"
        
        # Default to emotional_support instead of General
        return "emotional_support"
    
    def analyze_response_effectiveness(self, question: str, response: str, template_used: str) -> Dict[str, Any]:
        """
        Analyze how effective a response seems to be based on the prompt used.
        
        Args:
            question: The user's question
            response: The AI-generated response
            template_used: The prompt template that was used
        
        Returns:
            Dictionary with analysis metrics
        """
        analysis = {
            "template": template_used,
            "metrics": {}
        }
        
        try:
            # Length appropriateness
            response_words = len(response.split())
            analysis["metrics"]["response_length"] = response_words
            
            # Check if response is too short or too long
            if response_words < 20:
                analysis["metrics"]["length_quality"] = "too_short"
            elif response_words > 500:
                analysis["metrics"]["length_quality"] = "too_long"
            else:
                analysis["metrics"]["length_quality"] = "appropriate"
                
            # Check if response addresses the question (basic check)
            question_tokens = self.tokenize_and_lemmatize(question)
            response_tokens = self.tokenize_and_lemmatize(response)
            
            # Skip if tokenization failed
            if question_tokens and response_tokens:
                question_terms = set(question_tokens.split())
                response_terms = set(response_tokens.split())
                
                # Calculate overlap
                term_overlap = len(question_terms.intersection(response_terms))
                if len(question_terms) > 0:
                    analysis["metrics"]["term_overlap"] = term_overlap / len(question_terms)
                else:
                    analysis["metrics"]["term_overlap"] = 0
                    
            # Template-specific checks
            if template_used == "Question" and "?" in response:
                analysis["metrics"]["template_adherence"] = "high"
            elif template_used == "Empathy and Validation" and any(term in response.lower() 
                                                                for term in ["understand", "feel", "valid"]):
                analysis["metrics"]["template_adherence"] = "high"
            else:
                analysis["metrics"]["template_adherence"] = "medium"
        
        except Exception as e:
            logger.error(f"Error analyzing response effectiveness: {e}")
            analysis["metrics"]["error"] = str(e)
            
        return analysis
