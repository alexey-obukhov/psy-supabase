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
        
        # Process with NLP for deeper understanding
        doc = None
        if self.nlp:
            doc = self.nlp(cleaned_question)
        
        # Extract emotional tone and urgency signals
        emotional_signals = []
        if doc:
            emotional_signals = [token.text for token in doc if token.pos_ == "ADJ" and token.text.lower() in 
                            ["sad", "angry", "happy", "anxious", "scared", "worried", "depressed",
                                "hopeless", "fearful", "desperate", "overwhelmed", "confused",
                                "hurt", "upset", "frustrated", "lonely", "abandoned", "rejected"]]
        
        urgency_signals = any(word in cleaned_question.lower() for word in 
                        ["immediate", "urgent", "emergency", "now", "help me", "desperate", 
                        "crisis", "suicidal", "can't take it", "end my life", "give up"])
        
        # Full keyword mappings with comprehensive coverage
        keyword_mappings = {
            "Empathy and Validation": [
                "sad", "depressed", "down", "unhappy", "alone", "lonely", "grief", "loss", 
                "hurt", "pain", "suffering", "cry", "tears", "heartbroken", "devastated",
                "miserable", "despair", "hopeless", "worthless", "meaningless", "empty",
                "numb", "isolation", "rejected", "abandoned", "betrayed", "disappointed"
            ],
            "Affirmation and Reassurance": [
                "anxious", "worried", "stressed", "nervous", "fear", "scared", "panic", 
                "overwhelmed", "frightened", "uneasy", "tense", "afraid", "apprehensive",
                "dread", "terror", "phobia", "paranoid", "restless", "concerned", 
                "overthinking", "ruminating", "obsessing", "uncertainty", "doubt"
            ],
            "Providing Suggestions": [
                "help", "advice", "tips", "suggestion", "guidance", "recommend", "strategy", 
                "solution", "fix", "resolve", "approach", "technique", "method", "cope", "handle",
                "manage", "deal with", "overcome", "improve", "change", "what should I do", 
                "how can I", "options", "alternatives", "ideas", "tools"
            ],
            "Information": [
                "why", "explain", "understand", "how", "what", "learn", "know", "curious", 
                "information", "research", "fact", "science", "reason", "cause", "meaning",
                "definition", "describe", "enlighten", "clarify", "detail", "background",
                "mechanics", "process", "function", "operation", "mechanism", "theory"
            ],
            "Question": [
                "confused", "unsure", "uncertain", "wonder", "think", "feel", "opinion",
                "perspective", "view", "insight", "reflection", "consideration", "judgment",
                "assessment", "evaluation", "analysis", "thoughts", "feedback", "response",
                "reaction", "impression", "belief", "stance", "position"
            ],
            "Trauma": [
                "abuse", "trauma", "ptsd", "harass", "assault", "bully", "victim", 
                "workplace abuse", "work abuse", "boss abuse", "manager abuse",
                "toxic workplace", "hostile", "threat", "intimidate", "humiliate",
                "mistreat", "mobbing", "gaslighting", "discrimination", "retaliate",
                "harassment", "violence", "attack", "violate", "nightmare", "flashback",
                "trigger", "memory", "incident", "event", "childhood", "molest", "rape"
            ],
            "Cognitive Behavioral Therapy (CBT)": [
                "thought", "belief", "think", "pattern", "distortion", "irrational", 
                "negative", "cognitive", "automatic", "mindset", "perspective",
                "interpretation", "assumption", "core belief", "schema", "mental filter",
                "black and white", "catastrophizing", "personalization", "mind reading",
                "should statements", "labeling", "discounting positives", "magnification"
            ],
            "Mindfulness and Relaxation": [
                "calm", "breathe", "relax", "mindful", "present", "awareness", "meditation",
                "grounding", "centering", "peace", "tranquility", "serene", "zen",
                "breathing exercise", "body scan", "progressive relaxation", "visualization",
                "guided imagery", "stress reduction", "tension", "attention", "focus",
                "consciousness", "here and now", "sensations", "observation"
            ],
            "Grief and Loss": [
                "grief", "loss", "death", "died", "passed away", "gone", "missing",
                "mourning", "bereavement", "funeral", "memorial", "deceased", "departed",
                "lost someone", "anniversary", "coping with loss", "stages of grief",
                "denial", "anger", "bargaining", "depression", "acceptance", "widow",
                "widower", "survivor", "remember", "legacy", "tribute"
            ],
            "Relationship Healing": [
                "relationship", "partner", "spouse", "marriage", "couple", "together",
                "communication", "conflict", "argument", "fight", "misunderstanding",
                "trust", "betrayal", "infidelity", "cheating", "forgiveness", "reconciliation",
                "commitment", "compromise", "boundaries", "respect", "intimacy", "connection",
                "bond", "repair", "rebuild", "strengthen", "therapy"
            ]
        }

        # Match against keyword mappings with weighted scoring
        template_scores = {}
        for template, keywords in keyword_mappings.items():
            # Score based on exact keyword matches
            keyword_score = sum(2 for keyword in keywords if keyword in cleaned_question.lower())
            
            # Add partial matches with lower weight
            keyword_score += sum(0.5 for keyword in keywords if any(word.startswith(keyword) for word in cleaned_question.lower().split()))
            
            if keyword_score > 0:
                template_scores[template] = keyword_score
        
        # Enhanced keyword weights for relationship issues
        relationship_breakup_terms = {
            "broke up": 3.0,
            "break up": 3.0,
            "breakup": 3.0,
            "ex girlfriend": 2.5,
            "ex boyfriend": 2.5,
            "ex partner": 2.5,
            "ex wife": 2.5,
            "ex husband": 2.5
        }
        
        # Look for compound terms with higher weights
        for compound_term, weight in relationship_breakup_terms.items():
            if compound_term in cleaned_question.lower():
                if "Relationship Issues" in template_scores:
                    template_scores["Relationship Issues"] += weight
                else:
                    template_scores["Relationship Issues"] = weight
                    
                if "Empathy and Validation" in template_scores:
                    template_scores["Empathy and Validation"] += weight * 0.8  # Also boost related template
                else:
                    template_scores["Empathy and Validation"] = weight * 0.8
        
        # Detect common typos and misspellings
        typo_corrections = {
            "girlfrien": "girlfriend",
            "boyfried": "boyfriend",
            "mariage": "marriage",
            "divorc": "divorce",
            "seperat": "separate",
            "breakin up": "breaking up"
        }
        
        for typo, correction in typo_corrections.items():
            if typo in cleaned_question.lower():
                # Apply the same logic but with the corrected term
                logger.info(f"Detected possible typo: '{typo}', treating as '{correction}'")
                
                # Loop through template keywords to see if the correction matches
                for template, keywords in keyword_mappings.items():
                    if any(kw in correction for kw in keywords):
                        template_scores[template] = template_scores.get(template, 0) + 2.0
        
        # Check for depression mentioned alongside relationship terms - common combination
        if "depress" in cleaned_question.lower() and any(term in cleaned_question.lower() for term in 
                                                        ["broke up", "breakup", "ex ", "relationship"]):
            # This is a strong signal for empathy and validation with relationship focus
            template_scores["Empathy and Validation"] = template_scores.get("Empathy and Validation", 0) + 4.0
            template_scores["Relationship Issues"] = template_scores.get("Relationship Issues", 0) + 3.0
            logger.info("Detected depression in relationship context - boosting relevant templates")
                
        # Default to 'Empathy and Validation' if no specific matches or if urgent
        template_name = "Empathy and Validation"
        
        # After calculating all template_scores
        if template_scores:
            # Find max score for normalization
            max_score = max(template_scores.values())
            
            # Normalize scores to 0-1 range with minimum threshold
            normalized_scores = {}
            for template, score in template_scores.items():
                # Apply sigmoid-like normalization
                normalized_score = min(0.95, max(0.3, score / (max_score * 1.2)))
                normalized_scores[template] = normalized_score
                
            # Use normalized scores
            template_scores = normalized_scores
            
            # Select highest scoring template
            template_name = max(template_scores.items(), key=lambda x: x[1])[0]
            confidence = template_scores[template_name]
        else:
            confidence = 0.3  # Default confidence when no templates match
        
        # If urgency is detected, override with appropriate supportive template
        if urgency_signals:
            template_name = "Crisis Support"
            confidence = 0.9  # High confidence for crisis
            logger.info("Urgency detected in query, using Crisis Support template with high confidence")
        
        # Generate rich category information
        raw_category_info = self.generate_category_info(cleaned_question)
        category_info = self.refine_category_info(raw_category_info)
        
        # Determine topic with all available data
        topic = self._determine_topic(category_info, cleaned_question)
        
        # Special handling for detected trauma
        if "trauma" in topic.lower() or template_name == "Trauma":
            logger.info(f"Trauma detected: {topic}")
            # For trauma topics, ensure we're using trauma-informed approach
            template_name = "Trauma"
            confidence = max(confidence, 0.7)  # Ensure high confidence for trauma
        
        # Special handling for grief
        if "grief" in topic.lower() or "loss" in topic.lower():
            template_name = "Grief and Loss"
            confidence = max(confidence, 0.7)  # Ensure high confidence for grief
        
        # Build comprehensive enhanced context
        enhanced_context = {
            "detected_template": template_name,
            "detected_topic": topic,
            "category_info": category_info,
            "urgency_level": "high" if urgency_signals else "normal",
            "emotional_tone": emotional_signals[:3] if emotional_signals else ["neutral"],
            "confidence": confidence,
            "keywords_matched": [k for t in template_scores.keys() for k in keyword_mappings.get(t, []) if k in cleaned_question.lower()],
            "input_length": len(cleaned_question.split())
        }
        
        # Log the final template selection for debugging purposes
        logger.info(f"Final template selection: '{template_name}' for topic '{topic}' with confidence {confidence:.2f}")
        
        return template_name, enhanced_context


    def _determine_topic(self, category_info: Dict[str, str], question: str) -> str:
        """
        Determine the most relevant therapeutic topic using the enhanced mental health taxonomy.
        Returns 'emotional_support' as default when no specific match is found.
        """
        import re

        question_lower = question.lower()
        
        # Special pattern detection for relationship breakups
        breakup_pattern = re.search(r"\b(?:broke\s?up|break\s?up|ex\s+(?:girl|boy|partner|husband|wife))\b", 
                                question_lower)
        if breakup_pattern:
            # Direct matching for this specific pattern
            return "Relationship Issues"
        
        # Comprehensive primary topics with expanded keywords
        primary_topics = {
            "Depression": ["depression", "sad", "hopeless", "worthless", "empty", "tired", "unmotivated", 
                        "despair", "miserable", "unhappy", "low", "down", "blue", "gloomy", "grief", 
                        "crying", "tears", "exhausted", "numb", "apathy", "disinterest", "suicidal"],
            "Anxiety": ["anxiety", "worry", "panic", "fear", "stress", "nervous", "tense", "uneasy", 
                    "restless", "afraid", "scared", "dread", "apprehension", "anxious", "overwhelmed",
                    "overthinking", "rumination", "insecure", "frightened", "on edge", "jitters"],
            "Trauma": ["trauma", "abuse", "ptsd", "trigger", "flashback", "nightmare", "assault", 
                    "violence", "accident", "shock", "violated", "frightening", "horrifying", 
                    "terrifying", "disturbing", "threatening", "danger", "victim", "survivor"],
            "Relationship Issues": ["breakup", "divorce", "cheating", "trust", "communication", "arguing", 
                                "conflict", "ex", "partner", "spouse", "boyfriend", "girlfriend", 
                                "marriage", "separated", "dating", "betrayal", "jealousy", "controlling"],
            "Self-esteem": ["confidence", "self-worth", "inadequate", "failure", "inferior", "comparison", 
                        "not good enough", "self-doubt", "insecurity", "self-image", "self-hatred", 
                        "ugly", "stupid", "incompetent", "shame", "embarrassed", "humiliated"],
            "Stress": ["overwhelmed", "burnout", "pressure", "deadline", "too much", "exhaustion", 
                    "overworked", "can't cope", "stressed", "tension", "strain", "burden", 
                    "responsibilities", "demanding", "workload"],
            "Grief": ["loss", "death", "died", "passed away", "mourning", "bereavement", "missing", 
                    "gone", "funeral", "deceased", "lost someone", "grieving", "remembrance", 
                    "anniversary of death", "coping with loss"],
            "Identity": ["who am i", "purpose", "meaning", "direction", "lost", "confused about myself", 
                        "authentic", "real self", "true self", "identity crisis", "finding myself", 
                        "self-discovery", "questioning", "uncertain about future"]
        }
        
        # Detailed secondary topics with comprehensive keywords
        secondary_topics = {
            "Workplace": ["job", "work", "career", "boss", "coworker", "office", "colleague", "workplace", 
                        "employment", "profession", "manager", "supervisor", "fired", "laid off", "promotion", 
                        "demotion", "working", "professional", "employee", "employer", "company", "business"],
            "Relationship": ["partner", "spouse", "marriage", "date", "breakup", "divorce", "boyfriend", 
                            "girlfriend", "husband", "wife", "significant other", "engaged", "dating", 
                            "romance", "intimacy", "commitment", "couple", "affair", "dating app"],
            "Family": ["parent", "child", "sibling", "mother", "father", "family", "son", "daughter", 
                    "brother", "sister", "mom", "dad", "grandparent", "relative", "aunt", "uncle", 
                    "cousin", "in-law", "stepfamily", "adopted", "household"],
            "Social": ["friend", "friendship", "acquaintance", "social", "party", "gathering", "peer", 
                    "social media", "loneliness", "rejection", "belonging", "inclusion", "excluded", 
                    "outsider", "social anxiety", "social skills", "social life"],
            "Academic": ["school", "college", "university", "student", "study", "exam", "professor", 
                        "teacher", "class", "course", "degree", "education", "grades", "academic", 
                        "assignment", "thesis", "dissertation", "learning", "academic pressure"],
            "Health": ["illness", "disease", "diagnosis", "chronic", "pain", "symptom", "medical", 
                    "health anxiety", "hypochondria", "doctor", "hospital", "treatment", "medication", 
                    "recovery", "terminal", "disability", "condition", "health issue"],
            "Financial": ["money", "debt", "financial", "bills", "afford", "expensive", "poverty", 
                        "bankruptcy", "loan", "mortgage", "rent", "savings", "income", "unemployed", 
                        "budget", "financial stress", "economic", "finances"]
        }
        
        # Calculate scores with contextual weighting and extended matches
        primary_scores = {}
        for topic, keywords in primary_topics.items():
            # Use more sophisticated matching with context awareness
            exact_matches = sum(2 for kw in keywords if kw in question_lower)
            partial_matches = sum(1 for kw in keywords if any(word.startswith(kw) for word in question_lower.split()))
            score = exact_matches + (partial_matches * 0.5)
            if score > 0:
                primary_scores[topic] = score
                
        # Find intersection between primary and secondary topics
        composite_topics = {}
        for primary, p_score in primary_scores.items():
            for secondary, s_keywords in secondary_topics.items():
                s_score = sum(1.5 for kw in s_keywords if kw in question_lower)
                if s_score > 0:
                    # Create composite topic with combined confidence
                    composite = f"{secondary} {primary}"
                    composite_topics[composite] = p_score + s_score
        
        # Check for workplace abuse/trauma specifically with high priority
        workplace_indicators = ["at work", "my job", "my boss", "my manager", "my workplace", 
                            "my coworker", "my colleague", "office", "workplace", "company"]
        abuse_terms = ["abuse", "bully", "harass", "toxic", "trauma", "stress", "unfair", 
                    "discriminat", "threat", "hostile", "intimidat", "yell", "scream", 
                    "humiliat", "mistreat", "fired", "lay off"]
        
        if any(term in question_lower for term in workplace_indicators) and any(term in question_lower for term in abuse_terms):
            return "Workplace Trauma"
        
        # Special case for breakups and relationship issues
        relationship_terms = ["broke up", "breakup", "ex girlfriend", "ex boyfriend", "ex partner", "divorce"]
        if any(term in question_lower for term in relationship_terms):
            if "depress" in question_lower or "sad" in question_lower:
                return "Relationship Issues"  # This is a very specific and common category
        
        # Return highest scoring composite topic if available
        if composite_topics:
            return max(composite_topics.items(), key=lambda x: x[1])[0]
        
        # Return highest scoring primary topic if available
        if primary_scores:
            return max(primary_scores.items(), key=lambda x: x[1])[0]
        
        # If no clear topic is detected, extract entities and emotional content
        if self.nlp:
            doc = self.nlp(question)
            emotional_words = [token.text for token in doc if token.pos_ == "ADJ" and token.text in 
                            ["sad", "angry", "happy", "confused", "scared", "worried", "upset", 
                            "frustrated", "overwhelmed", "disappointed", "lonely"]]
            if emotional_words:
                if "sad" in emotional_words or "lonely" in emotional_words:
                    return "Depression"
                if "scared" in emotional_words or "worried" in emotional_words:
                    return "Anxiety"
                if "angry" in emotional_words or "frustrated" in emotional_words:
                    return "Emotional Regulation"
        
        return "emotional_support"  # Default fallback

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
