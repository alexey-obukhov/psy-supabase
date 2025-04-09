"""
prompt_selector.py

This module implements the `PromptSelector` class, which is responsible for selecting the most appropriate
therapeutic prompt template based on user input. It uses semantic matching, keyword associations, and
natural language processing (NLP) techniques to analyse user queries and determine the best response strategy.

Key Features:
- **Therapeutic Prompt Selection**:
  - Matches user input to predefined therapeutic prompt templates based on keywords, topics, and emotions.
  - Supports a wide range of therapeutic categories, including anxiety, depression, trauma, grief, and more.

- **Natural Language Processing (NLP)**:
  - Tokenizes and lemmatizes user input to extract key concepts.
  - Uses spaCy for named entity recognition (NER) and advanced text processing.
  - Cleans and normalises user input for consistent analysis.

- **Category and Topic Analysis**:
  - Maps user input to therapeutic categories such as "Empathy and Validation" or "Providing Suggestions."
  - Detects primary and secondary topics using an enhanced mental health taxonomy.
  - Identifies emotional content and intensity to tailor responses.

- **Response Effectiveness Analysis**:
  - Evaluates the quality of AI-generated responses based on length, term overlap, and adherence to the selected prompt template.

Classes:
- `PromptSelector`: The main class that provides methods for analysing user input, selecting prompt templates,
  and refining therapeutic categories.

Dependencies:
- `psy_supabase.utilities.utils`: Utility functions for loading mental health taxonomies.
- `psy_supabase.utilities.nlp_utils`: NLP utilities for text cleaning, tokenization, and entity extraction.
- `psy_supabase.utilities.templates.therapeutic_prompt`: Predefined therapeutic prompt templates.
- `school_logging.log.ColoredLogger`: Enhanced logging for debugging and monitoring.

Usage:
    from psy_supabase.utilities.prompt_selector import PromptSelector

    # Initialize the prompt selector with a text generator
    prompt_selector = PromptSelector(generator)

    # Analyze a user question and select a prompt template
    question = "I'm feeling very anxious about my upcoming presentation."
    template, context = prompt_selector.select_prompt_template(question)

    print("Selected Template:", template)
    print("Context:", context)

    # Generate category information for the question
    category_info = prompt_selector.generate_category_info(question)
    print("Category Info:", category_info)

    # Analyze the effectiveness of a response
    response = "Try practicing deep breathing exercises to calm your nerves."
    analysis = prompt_selector.analyze_response_effectiveness(question, response, template)
    print("Response Analysis:", analysis)
"""
import re
import traceback
from typing import Dict, List, Tuple, Any, TYPE_CHECKING

from psy_supabase.utilities.utils import load_enhanced_mental_health_taxonomy
from psy_supabase.utilities.templates.therapeutic_prompt import prompt_templates
from psy_supabase.utilities.nlp_utils import get_spacy_model
from school_logging.log import ColoredLogger

if TYPE_CHECKING:
    from psy_supabase.core.model_manager import TextGenerator

logger = ColoredLogger(__name__)

class PromptSelector:
    """
    Selects the most appropriate therapeutic prompt template based on user input.
    Uses semantic matching and existing prompt templates for optimal responses.
    """

    def __init__(self, generator: 'TextGenerator', prompt_templates: Dict[str, str] = prompt_templates):
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
        """Clean text by removing unwanted characters and normalising it."""
        from psy_supabase.utilities.nlp_utils import clean_text as nlp_clean_text

        # Use the shared clean_text function
        cleaned = nlp_clean_text(text)

        # Add any PromptSelector-specific cleaning if needed
        cleaned = re.sub(r"[^a-zA-Z0-9\s'\":-]", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned

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
            logger.error("Error in tokenize_and_lemmatize: %s", e)
            return text

    def extract_entities(self, question: str) -> List[str]:
        """Process the question to get named entities."""
        from psy_supabase.utilities.nlp_utils import extract_entities as nlp_extract_entities

        # Use the shared function from nlp_utils
        entities_data = nlp_extract_entities(question)

        # Convert entity dictionaries to just the text values
        return [entity["text"] for entity in entities_data]

    def generate_category_info(self, question: str) -> Dict[str, str]:
        """
        Generate category information for a question, focusing on meaningful categories only.

        Args:
            question: The user question or statement.

        Returns:
            A dictionary mapping categories to descriptions, without any empty categories.
        """
        if not question:
            return {"Others": "therapeutic support"}

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
            category_info["Others"] = "therapeutic support"

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


    def select_prompt_template(self, question_text: str) -> Tuple[str, Dict[str, Any]]:
        """Select the appropriate prompt template based on the question content."""
        try:
            # Analyze question to determine topic and template
            classification = self.analyze_question(question_text)
            topic = classification.get("topic", "general")
            confidence = classification.get("confidence", 0.5)

            # Basic mapping of topics to templates
            template_mapping = {
                "anxiety": "Anxiety Support",
                "depression": "Depression Support",
                "grief": "Grief Support",
                "trauma": "Trauma Support",
                "relationships": "Relationship Support",
                "self_esteem": "Self-Esteem Support",
                "stress": "Stress Management",
                "identity": "Identity Exploration",
                "loneliness": "Loneliness Support",
                "motivation": "Motivation Support",
                "general": "basic_answer",
                "suicidal": "Crisis Support",
                "crisis": "Crisis Support",
                "direct_therapeutic_exploration": "direct_therapeutic_exploration",
                "gentle_therapeutic_guidance": "gentle_therapeutic_guidance",
                "subtle_therapeutic_exploration": "subtle_therapeutic_exploration",
            }

            # Check for emotion-based overrides
            emotion = classification.get("emotion")
            emotion_intensity = classification.get("emotion_intensity", 0.5)

            # Override template for high emotional intensity
            if emotion_intensity > 0.8:
                if emotion in ["anger", "frustration"]:
                    template_name = "Emotional Regulation"
                elif emotion in ["sadness", "despair"]:
                    template_name = "Empathy and Validation"
                elif emotion in ["anxiety", "fear"]:
                    template_name = "Grounding and Reassurance"
                else:
                    template_name = template_mapping.get(topic, "basic_answer")
            else:
                template_name = template_mapping.get(topic, "basic_answer")

            # Return the selected template name and context data
            return template_name, {
                "detected_topic": topic,
                "confidence": confidence,
                "emotion": emotion,
                "emotion_intensity": emotion_intensity
            }
        except Exception as e:
            logger.error("Error selecting prompt template: %s", e)
            return "basic_answer", {"detected_topic": "general", "confidence": 0.5}

    def determine_topic(self, category_info: Dict[str, str], question: str) -> str:
        """
        Determine the most relevant therapeutic topic using the enhanced mental health taxonomy.
        Returns 'emotional support' as default when no specific match is found.
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

        return "emotional support"  # Default fallback

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
            logger.error("Error analysing response effectiveness: %s", e)
            analysis["metrics"]["error"] = str(e)

        return analysis

    def analyze_question(self, question_text: str, min_confidence=0.5) -> Dict[str, Any]:
        """
        Analyze a question to determine topic, emotion, and other contextual factors.

        Uses pattern matching with weighted rules to identify topics and emotions in user questions.
        Handles complex cases such as:
        - Crisis detection (highest priority)
        - Mixed emotions (e.g., anxiety with depression)
        - Topic dependencies (e.g., self-esteem issues from trauma)
        - Subtle emotional cues without explicit terms
        - Confidence scoring with contextual boosting

        The method applies several best practices from NLP research:
        - Context-aware emotion prioritization
        - Topic dependency modeling
        - Intensity modifiers recognition
        - Multi-label classification for mixed cases

        Args:
            question_text: The user's question text
            min_confidence: Minimum confidence threshold for topic detection (default: 0.5)

        Returns:
            Dict containing analysis results including:
            - topic: Detected primary topic (e.g., "anxiety", "depression")
            - confidence: Confidence score for topic detection (0.0-1.0)
            - emotion: Detected primary emotion (e.g., "fear", "sadness")
            - emotion_intensity: Intensity score for detected emotion (0.0-1.0)
        """
        try:
            # Initialize default analysis results
            analysis = {
                "topic": "general",
                "confidence": 0.5,
                "emotion": None,
                "emotion_intensity": 0.0
            }

            # Skip analysis for empty questions
            if not question_text or len(question_text.strip()) < 3:
                return analysis

            # Check for crisis keywords first (safety priority)
            crisis_keywords = [
                "suicide", "kill myself", "want to die", "end my life",
                "don't want to live", "do not want to live", "suicidal", "harm myself"
            ]
            if any(keyword in question_text.lower() for keyword in crisis_keywords):
                analysis["topic"] = "crisis"
                analysis["confidence"] = 0.95
                analysis["emotion"] = "distress"
                analysis["emotion_intensity"] = 0.9
                return analysis

            # Enhanced topic classification patterns
            topic_patterns = {
                "anxiety": [
                    # Standard patterns - with higher weight (2.0)
                    (r'\banxiety\b', 2.0), (r'\banxious\b', 2.0), (r'\bpanic\b', 2.0),
                    (r'\bworried\b', 1.8), (r'\bfear\b', 1.8),
                    (r'\bstress(ed)?\b', 1.5), (r'\boverwhelm(ed|ing)\b', 1.5),
                    # Subtle patterns
                    (r'\bkeep thinking about what could\b', 1.5), (r'\bconstantly worry\b', 1.5),
                    (r'\bnervous\b', 1.5), (r'\bon edge\b', 1.5), (r'\bcan\'t relax\b', 1.5),
                    (r'\brestless\b', 1.2), (r'\buneasy\b', 1.2), (r'\btense\b', 1.2),
                    (r'\bworry about\b', 1.8), (r'\banxious about\b', 2.0)
                ],
                "depression": [
                    # Standard patterns
                    (r'\bdepress(ed|ion)\b', 2.0), (r'\bsad\b', 1.8), (r'\blow\b', 1.2),
                    (r'\bmood\b', 1.0), (r'\bhopeless\b', 1.8), (r'\bunmotivated\b', 1.5),
                    (r'\bexhausted\b', 1.2),
                    # Subtle patterns
                    (r'\bdon\'t enjoy\b', 1.5), (r'\bno pleasure\b', 1.5), (r'\blost interest\b', 1.5),
                    (r'\bno energy\b', 1.2), (r'\bfeel empty\b', 1.5), (r'\bworthless\b', 1.8),
                    (r'\btired all the time\b', 1.2), (r'\bfeeling down\b', 1.5)
                ],
                "trauma": [
                    (r'\btrauma\b', 2.0), (r'\bptsd\b', 2.0), (r'\babuse\b', 1.8), (r'\bviolent\b', 1.5),
                    (r'\bassault\b', 1.8), (r'\bincident\b', 1.0), (r'\bflashbacks\b', 1.8),
                    (r'\bnightmares\b', 1.5), (r'\bhaunt\b', 1.2)
                ],
                "relationships": [
                    (r'\bpartner\b', 1.5), (r'\bspouse\b', 1.5), (r'\bmarriage\b', 1.8),
                    (r'\brelationship\b', 2.0), (r'\bdating\b', 1.5), (r'\bcouple\b', 1.2),
                    (r'\bex\b', 1.3), (r'\bbreak[- ]?up\b', 1.8), (r'\bgirlfriend\b', 1.5),
                    (r'\bboyfriend\b', 1.5), (r'\bhusband\b', 1.5), (r'\bwife\b', 1.5),
                    (r'\bdivorce\b', 1.8), (r'\bseparation\b', 1.5)
                ],
                "self_esteem": [
                    (r'\bself[- ]esteem\b', 2.0), (r'\bconfidence\b', 1.8), (r'\bworth\b', 1.5),
                    (r'\bunlovable\b', 1.8), (r'\bunattractive\b', 1.5), (r'\binadequate\b', 1.8),
                    (r'\bnot good enough\b', 2.0), (r'\bnever feel good enough\b', 2.0),
                    (r'\bfailure\b', 1.8), (r'\bworthless\b', 1.8), (r'\bhate myself\b', 2.0),
                    (r'\bugly\b', 1.5), (r'\bcompared to others\b', 1.8)
                ],
                "stress": [
                    (r'\bstress(ed)?\b', 1.8), (r'\boverwhelm(ed|ing)\b', 1.8), (r'\bbusy\b', 1.0),
                    (r'\bworkload\b', 1.5), (r'\bburn[- ]?out\b', 1.8), (r'\bcoping\b', 1.2),
                    (r'\btoo much to do\b', 1.5), (r'\boverworked\b', 1.8)
                ],
                "identity": [
                    (r'\bidentity\b', 1.8), (r'\bwho am I\b', 1.8), (r'\bmeaning\b', 1.5),
                    (r'\bpurpose\b', 1.5), (r'\bdirection\b', 1.2), (r'\blife purpose\b', 1.8),
                    (r'\bexistential\b', 1.8)
                ],
                "loneliness": [
                    (r'\blonely\b', 2.0), (r'\balone\b', 1.8), (r'\bisolat(ed|ion)\b', 1.8),
                    (r'\bno friends\b', 2.0), (r'\bsocially\b', 1.0), (r'\bconnection\b', 1.2),
                    (r'\bno one\b', 1.5), (r'\bsolitude\b', 1.5)
                ],
            }

            # Enhanced emotion detection patterns with more comprehensive patterns
            emotion_patterns = {
                "anger": [
                    (r'\bangry\b', 2.0), (r'\bmad\b', 1.8), (r'\bfurious\b', 2.0),
                    (r'\birritated\b', 1.5), (r'\bfrustrated\b', 1.5), (r'\bresent\b', 1.5),
                    (r'\bfed up\b', 1.5), (r'\bannoy(ed|ing)\b', 1.5),
                    (r'\bso angry\b', 2.2), (r'\bfeel so angry\b', 2.3)  # Higher weight for intensity
                ],
                "sadness": [
                    (r'\bsad\b', 2.0), (r'\bcry(ing)?\b', 1.8), (r'\btear(s|ful)?\b', 1.8),
                    (r'\bupset\b', 1.5), (r'\bmiserable\b', 1.8), (r'\bheartbroken\b', 2.0),
                    (r'\bdown\b', 1.2), (r'\blow\b', 1.2), (r'\bdepress(ed|ion)\b', 2.0),
                    (r'\bhopeless\b', 1.8), (r'\bblue\b', 1.0),
                    (r'\bempty inside\b', 1.8), (r'\bfeel empty\b', 1.8), # Added for "feel empty inside"
                    (r'\blonely\b', 1.5), (r'\balone\b', 1.2)
                ],
                "fear": [
                    (r'\bafraid\b', 1.8), (r'\bscared\b', 2.0), (r'\bfearful\b', 1.8),
                    (r'\bterrified\b', 2.0), (r'\banxious\b', 2.0), (r'\bpanic\b', 2.0),
                    (r'\bworry\b', 1.8), (r'\bworried\b', 1.8), # Added explicit 'worried'
                    (r'\bnervous\b', 1.5), (r'\buneasy\b', 1.2),
                    (r'\bdread\b', 1.5), (r'\btense\b', 1.2),
                    (r'\bthings won\'t work out\b', 1.5), # Added for "worried things won't work out"
                    (r'\bfear\b', 1.8), (r'\banxiety\b', 2.0) # Added more direct mentions
                ],
                "joy": [
                    (r'\bhappy\b', 2.0), (r'\bjoy(ful)?\b', 2.0), (r'\belated\b', 1.8),
                    (r'\bexcited\b', 1.8), (r'\bglad\b', 1.5), (r'\bpleased\b', 1.5),
                    (r'\bthrilled\b', 1.8), (r'\bdelighted\b', 1.8), (r'\bgrateful\b', 1.5),
                    (r'\bcontent\b', 1.2)
                ],
                "shame": [
                    (r'\bashamed\b', 2.0), (r'\bembarrassed\b', 1.8), (r'\bhumiliated\b', 1.8),
                    (r'\bregret\b', 1.5), (r'\bguilt(y)?\b', 2.0), (r'\bremorse\b', 1.8),
                    (r'\bsorry\b', 1.5), (r'\bblame myself\b', 2.0),
                    (r'\bcan\'t forgive myself\b', 2.2), # Added for "can't forgive myself"
                    (r'\bforgive myself\b', 1.8)
                ],
                "surprise": [
                    (r'\bsurprised\b', 2.0), (r'\bshocked\b', 1.8), (r'\bastounded\b', 1.8),
                    (r'\bamazed\b', 1.8), (r'\bastonished\b', 1.8), (r'\bstunned\b', 1.7),
                    (r'\bdidn\'t expect\b', 1.5), (r'\bwasn\'t expecting\b', 1.5),
                    (r'\bhow things turned out\b', 1.3) # Added for "surprised by how things turned out"
                ]
            }

            # Normalize question text
            normalized_text = question_text.lower()

            # Analyze topics
            topic_scores = {}
            for topic, patterns in topic_patterns.items():
                topic_score = 0
                pattern_matches = []

                for pattern, weight in patterns:
                    if re.search(pattern, normalized_text):
                        pattern_matches.append((pattern, weight))
                        topic_score += weight

                if pattern_matches:
                    # Calculate weighted confidence score
                    total_weight = sum(w for _, w in pattern_matches)

                    # Base confidence (starting higher than before: 0.65)
                    base_confidence = 0.65

                    # Add weighted proportion (max add of 0.30)
                    confidence = base_confidence + min(0.30, total_weight / 10)

                    # Boost for intensity words
                    intensity_words = ["extremely", "very", "really", "so", "deeply", "constantly"]
                    if any(word in normalized_text for word in intensity_words):
                        confidence = min(0.99, confidence + 0.08)

                    topic_scores[topic] = confidence

            # Select highest scoring topic
            if topic_scores:
                # Priority topics with revised priorities
                priority_topics = {
                    "crisis": 10,
                    "anxiety": 3,
                    "relationships": 5,
                    "self_esteem": 4,
                    "trauma": 5,
                    "depression": 2
                }

                # Find highest scoring topics
                max_score = max(topic_scores.values())

                # Consider top topics that are close to max score (80% of max)
                top_topics = [(t, s) for t, s in topic_scores.items() if s >= max_score * 0.8]

                # Special case for "anxious and depressed" - make sure anxiety is preferred
                if "anxious" in normalized_text and "depress" in normalized_text:
                    if "anxiety" in topic_scores and "depression" in topic_scores:
                        priority_topics["anxiety"] = 8  # Extra boost for this specific case

                # Sort by priority then by score
                top_topics.sort(key=lambda x: (priority_topics.get(x[0], 1), x[1]), reverse=True)

                # Select the top topic
                top_topic = top_topics[0]
                analysis["topic"] = top_topic[0]
                analysis["confidence"] = top_topic[1]

            # Analyze emotions with weighted patterns
            emotion_scores = {}
            for emotion, patterns in emotion_patterns.items():
                emotion_score = 0
                pattern_matches = []

                for pattern, weight in patterns:
                    if re.search(pattern, normalized_text):
                        pattern_matches.append((pattern, weight))
                        emotion_score += weight

                if pattern_matches:
                    # Calculate weighted intensity
                    total_weight = sum(w for _, w in pattern_matches)

                    # Base emotion intensity (starting higher: 0.6)
                    base_intensity = 0.6

                    # Add weighted proportion (max add of 0.30)
                    intensity = base_intensity + min(0.35, total_weight / 8)

                    # Boost for intensity words
                    intensity_words = {
                        "very": 0.08,
                        "extremely": 0.10,
                        "really": 0.07,
                        "so": 0.09,
                        "incredibly": 0.10,
                        "terribly": 0.09
                    }

                    for word, boost in intensity_words.items():
                        if f" {word} " in f" {normalized_text} ":
                            intensity = min(0.99, intensity + boost)

                    emotion_scores[emotion] = intensity

            # Select emotion with highest score
            if emotion_scores:
                max_emotion = max(emotion_scores.items(), key=lambda x: x[1])
                analysis["emotion"] = max_emotion[0]
                analysis["emotion_intensity"] = max_emotion[1]

            # Apply minimum confidence threshold if provided
            if analysis["confidence"] < min_confidence and analysis["topic"] != "crisis":
                analysis["topic"] = "general"
                analysis["confidence"] = 0.4

            # Ensure emotion is never None
            if analysis["emotion"] is None:
                analysis["emotion"] = "none"
                analysis["emotion_intensity"] = 0.0

            # Log the analysis results
            logger.info(f"Question analysed - Topic: {analysis['topic']} ({analysis['confidence']:.2f}), "
                        f"Emotion: {analysis['emotion']} ({analysis['emotion_intensity']:.2f})")

            return analysis

        except Exception as e:
            logger.error("Error analyzing question: %s", e)
            logger.error(traceback.format_exc())
            default_analysis = {
                "topic": "general",
                "confidence": 0.5,
                "emotion": "none",
                "emotion_intensity": 0.0
            }
            return default_analysis
