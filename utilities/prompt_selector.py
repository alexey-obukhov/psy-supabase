"""
Prompt selector for therapeutic AI responses.
Based on Alexey Obukhov's therapeutic prompt system.
"""
import logging
from typing import Dict, List, Tuple, Any

from psy_supabase.utilities.text_utils import load_enhanced_mental_health_taxonomy
from psy_supabase.utilities.templates.therapeutic_prompt import prompt_templates
from psy_supabase.utilities.nlp_utils import get_spacy_model

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
        
        # Use the shared spaCy model
        self.nlp = get_spacy_model()
        if self.nlp is None:
            logger.error("Failed to load spaCy model. Some functionality may be limited.")
        
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
            "Trauma": [
                "abuse", "trauma", "ptsd", "harass", "assault", "bully", "victim",
                "workplace abuse", "work abuse", "boss abuse", "manager abuse",
                "toxic workplace", "hostile", "threat", "intimidate", "humiliate",
                "mistreat", "mobbing", "gaslighting", "discrimination", "retaliate",
                "harassment"
            ],
            # Add more mappings as needed
        }

        self.topic_keywords = load_enhanced_mental_health_taxonomy()

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

    def extract_entities(self, question: str) -> List[str]:
        """ Process the question to get named entities """
        doc = self.nlp(question)
        entities = []
        for ent in doc.ents:
            entities.append(ent.text)
        return entities

    def generate_category_info(self, question: str) -> Dict[str, str]:
        """
        Generate category information for a question, focusing on meaningful categories only.
        
        Args:
            question: The user question or statement.
            
        Returns:
            A dictionary mapping categories to descriptions, without any empty categories.
        """
        if not question:
            return {"Others": "General therapeutic support"}
        
        # Clean and normalize the question text for NLP processing
        cleaned_question = self.clean_text(question).lower()
        
        # Initialize category info dictionary
        category_info = {}
        
        # HIGH PRIORITY MENTAL HEALTH CONCERNS - Direct matching for critical keywords
        # Depression - Check explicitly first due to its importance
        if any(word in cleaned_question for word in ["depress", "hopeless", "sad", "suicid", 
                                                   "end my life", "kill myself", "worthless"]):
            category_info["Empathy and Validation"] = "Supporting depression and hopelessness with validation"
            
        # Anxiety
        if any(word in cleaned_question for word in ["anxi", "worry", "panic", "fear", "stress"]):
            category_info["Affirmation and Reassurance"] = "Supporting anxiety with reassurance"
        
        # Trauma
        if any(word in cleaned_question for word in ["trauma", "abuse", "assault", "ptsd"]):
            category_info["Trauma"] = "Supporting trauma recovery"
        
        # RELATIONSHIP CATEGORIES - Using nlp entities and keywords
        entities = self.extract_entities(cleaned_question)
        for entity in entities:
            entity = entity.lower()
            if entity in ["grief", "loss", "bereavement", "death", "died"]:
                category_info["Grief"] = "Support for dealing with loss and grief"
            elif entity in ["relationship", "partner", "breakup", "divorce", "marriage"]:
                category_info["Interpersonal"] = "Supporting relationship issues or interpersonal struggles"
        
        # THERAPEUTIC APPROACHES - Add these based on question content
        if any(word in cleaned_question for word in ["help", "advice", "tip", "suggestion"]):
            category_info["Providing Suggestions"] = "Offering gentle suggestions or strategies for improvement"
        
        if any(word in cleaned_question for word in ["explain", "why", "how", "what", "reason"]):
            category_info["Information"] = "Providing relevant psychoeducational information"
        
        # TECHNIQUE-SPECIFIC CATEGORIES
        if any(word in cleaned_question for word in ["thought", "belief", "think", "pattern"]):
            category_info["Cognitive Behavioral Therapy (CBT)"] = "Addressing thought patterns"
        
        if any(word in cleaned_question for word in ["calm", "breathe", "relax", "mindful"]):
            category_info["Mindfulness and Relaxation"] = "Guiding relaxation and mindfulness practices"
        
        # ALWAYS ensure we have at least one category
        if not category_info:
            category_info["Others"] = "General therapeutic support"
        
        return {cat: description for cat, description in category_info.items() if description.strip()}

    def refine_category_info(self,
                             raw_category_info: Dict[str, str]
                             ) -> Dict[str, str]:
        """
        Ensures there is at least one valid category in the refined category info.
        
        Args:
            raw_category_info: The raw category info generated by `generate_category_info`.
            
        Returns:
            A dictionary mapping categories to descriptions.
        """
        # If no valid categories are found, default to 'Providing Suggestions'
        if not raw_category_info:
            raw_category_info['Providing Suggestions'] = 'Offering gentle suggestions or strategies'
        
        return raw_category_info


    def select_prompt_template(self,
user_question: str
) -> Tuple[str, Dict[str, Any]]:
        """
        Select the most appropriate prompt template based on user input.
        
        Args:
            user_question: The user's question or statement
        
        Returns:
            Tuple of (template_name, enhanced_context)
        """
        # Clean and normalize the question
        cleaned_question = self.clean_text(user_question)
        
        # Default to 'Others' template if no specific match is found
        template_name = "Others"
        
        # Step 1: First, check for keyword matches (fast and efficient)
        for template, keywords in self.keyword_mappings.items():
            if any(keyword in cleaned_question.lower() for keyword in keywords):
                template_name = template
                break
        
        # Step 2: Generate and refine category information from the question
        try:
            # Generate raw category info only once
            raw_category_info = self.generate_category_info(cleaned_question)
            # Refine the category info to eliminate empty or non-meaningful entries
            category_info = self.refine_category_info(raw_category_info)
            
            # Log category info only once after refinement
            logger.info(f"Refined category info: {category_info}")
        except Exception as e:
            logger.error(f"Error generating categories: {e}")
            category_info = {cat: "" for cat in self.prompt_templates.keys()}
        
        # Step 3: If no template was selected through keywords, use the category info
        if template_name == "Others":
            for category, description in category_info.items():
                if description.strip() and category in self.prompt_templates:
                    # Select a template based on refined category info
                    template_name = category
                    break
        
        # Step 4: Determine the topic based on the category info and question
        topic = self._determine_topic(category_info, cleaned_question)
        
        # Step 5: Make sure we have an appropriate template for the detected topic
        # Check if we need to map the topic to a specific template
        if topic in ["Workplace Trauma", "Heartbreak", "Relationship"]:
            # These are specialized topics that need their own templates
            if topic in self.prompt_templates:
                # If we have a dedicated template for this topic, use it
                template_name = topic
                logger.info(f"Using specialized template '{template_name}' for topic '{topic}'")
            else:
                # Map to the closest template if no exact match exists
                template_mapping = {
                    "Workplace Trauma": "Trauma",
                    "Heartbreak": "Empathy and Validation", 
                    "Relationship": "Empathy and Validation"
                }
                template_name = template_mapping.get(topic, template_name)
                logger.info(f"Mapped topic '{topic}' to closest template '{template_name}'")
        
        # Step 6: Build the enhanced context for the prompt
        enhanced_context = {
            "detected_template": template_name,
            "detected_topic": topic,
            "category_info": category_info
        }
        
        # Log the final template selection for debugging purposes
        logger.info(f"Final template selection: '{template_name}' for topic '{topic}'")
        
        # Return the final selected template and the enhanced context
        return template_name, enhanced_context


    def _determine_topic(self, category_info: Dict[str, str], question: str) -> str:
        """
        Determine the most relevant therapeutic topic using the enhanced mental health taxonomy.
        Returns 'emotional_support' as default when no specific match is found.
        """
        question_lower = question.lower()

        # Extract entities using NER (Named Entity Recognition)
        entities = self.extract_entities(question_lower)
        
        # Track matches with their scores
        topic_scores = {}

        # Special handling for workplace trauma detection
        workplace_indicators = ["at work", "my job", "my boss", "my manager", 
                            "my workplace", "my coworker", "my colleague",
                            "office", "workplace", "company", "employer", "employment"]
        
        abuse_terms = ["abuse", "bully", "harass", "toxic", "trauma", "stress", 
                    "unfair", "discriminat", "threat", "hostile", "intimidat",
                    "yell", "scream", "humiliat", "mistreat", "fired", "lay off"]
        
        # Check for workplace context
        has_workplace = any(term in question_lower for term in workplace_indicators)
        has_abuse = any(term in question_lower for term in abuse_terms)
        
        # Strong signal: If both workplace AND abuse terms are present, prioritize Workplace Trauma
        if has_workplace and has_abuse:
            logger.info("Detected workplace abuse context - prioritizing Workplace Trauma topic")
            return "Workplace Trauma"

        logger.info(f"Processing question with {len(self.topic_keywords)} possible topics")
        
        # Check each topic in our comprehensive taxonomy
        for topic, keywords in self.topic_keywords.items():
            if topic == "emotional_support":  # Skip the default topic
                continue
            
            # Count how many keywords match in the question
            matches = sum(1 for keyword in keywords if keyword in question_lower)
            entity_matches = sum(1 for entity in entities if entity in keywords)
            
            # Combine the scores from keyword matches and entity matches
            total_matches = matches + entity_matches

            # Boosting: Apply category info-based adjustments
            # The idea is to give more weight to topics suggested by category_info
            if category_info.get('Anxiety') and topic == 'Anxiety':
                total_matches += 3  # Boost Anxiety-related topics more strongly
            elif category_info.get('Stress') and topic == 'Stress':
                total_matches += 2  # Boost Stress-related topics slightly less
            elif category_info.get('Depression') and topic == 'Depression':
                total_matches += 2  # Boost Depression-related topics
            
            # You can continue to apply similar boosts for other categories as needed:
            elif category_info.get('Relationship') and topic == 'Relationship':
                total_matches += 2  # Relationship-related topics
            elif category_info.get('Self-Esteem') and topic == 'Self-Esteem':
                total_matches += 1  # Self-Esteem-related topics
            elif category_info.get('Relationship Healing') and topic == 'Relationship Healing':
                total_matches += 3  # Give higher score for relationship healing
            elif category_info.get('Breakup Recovery') and topic == 'Grief':
                total_matches += 2  # Slightly boost grief-related topics if related to breakup
            
            #todo If category_info indicates other relevant categories, apply them similarly
            
            # If total_matches is greater than 0, record the topic score
            if total_matches > 0:
                topic_scores[topic] = total_matches
        
        # If we have matches, return the topic with the highest score
        if topic_scores:
            best_topic = max(topic_scores.items(), key=lambda x: x[1])[0]
            return best_topic
        
        # Default to emotional_support if no good match found
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
