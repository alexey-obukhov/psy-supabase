import re
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from prismalog.log import get_logger
from transformers import AutoModel, AutoTokenizer

from psy_supabase.utilities.therapeutic_mappings import TherapeuticMappings

logger = get_logger(__name__)


class SemanticEmotionDetector:
    """
    Lightweight emotion and topic detector using a small language model
    optimized for semantic similarity.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize with a very small but effective sentence transformer model.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

            # Put model in evaluation mode and if possible on GPU
            self.model.eval()
            self.device = "cpu"
            self.model.to(self.device)

            logger.info(f"Initialized emotion detector with model {model_name} on {self.device}")

            # Initialize emotion examples with their descriptions
            self.emotion_templates = {
                # Core emotional states
                "anger": "feeling angry, frustrated, irritated, mad or furious about something that happened",
                "anxiety": "feeling worried, nervous, anxious, afraid or extremely anxious about what might happen",
                "depression": "feeling sad, empty, hopeless, depressed, or not enjoying anything anymore",
                "shame": "feeling embarrassed, humiliated, or inadequate about oneself",
                "guilt": "feeling responsible, regretful, or remorseful about something done wrong",
                "joy": "feeling happy, excited, pleased, or content about something good",
                "surprise": "feeling shocked, amazed, or caught off guard by something unexpected",
                "concern": "feeling uneasy, troubled, or bothered about a situation",
                # Additional emotional states
                "fear": "feeling scared, terrified, frightened, or panicked about a threat or danger",
                "loneliness": "feeling isolated, alone, disconnected, or lacking meaningful connections",
                "grief": "feeling deep sorrow, loss, heartache, or mourning someone or something",
                "helplessness": "feeling powerless, unable to cope, or lacking control over situations",
                "overwhelm": "feeling too much pressure, stress, or unable to handle multiple demands",
                "confusion": "feeling uncertain, unclear, or having difficulty understanding situations",
                "hope": "feeling optimistic, looking forward to possibilities, or seeing potential for positive change",
                "numbness": "feeling emotionally empty, disconnected, or unable to feel emotions",
                "exhaustion": "feeling mentally or emotionally drained, depleted, or burnt out",
                "frustration": "feeling blocked, thwarted, or unable to achieve desired goals",
                # Social emotions
                "rejection": "feeling unwanted, excluded, or pushed away by others",
                "jealousy": "feeling threatened by others' advantages or relationships",
                "trust": "feeling safe, secure, and able to rely on others",
                "gratitude": "feeling thankful, appreciative, or grateful for experiences or people",
                # Neutral states
                "neutral": "feeling calm, balanced, or emotionally steady",
                "greeting": "saying hello, hi, good morning, or checking in without expressing a specific emotion",
            }

            # Add topic templates including greeting
            self.topic_templates = {
                "anxiety": "anxiety, worried, nervous, fear, panic, stress, phobia; feeling extremely anxious",
                "depression": "depression, sad, low mood, hopelessness; don't enjoy anything anymore; lack of interest",
                "supportive_listening": "listening, support, understanding, validation, empathy",
                "trauma": "trauma, ptsd, abuse, neglect, painful experiences",
                "relationship_issues": "relationship, partner, marriage, dating, breakup, divorce, couple, romantic",
                "self-esteem": "self esteem, confidence, self worth, inadequate, not good enough",
                "grief_loss": "grief, loss, death, mourning, bereavement",
                "shame": "shame, embarrassment, humiliation, social rejection",
                "guilt": "guilt, regret, remorse, responsibility, blame",
                # Add greeting category for topic detection
                "greeting": "hello, hi, hey, good morning, good day, greetings, checking in, how are you, what's up",
            }

            # Pre-compute emotion embeddings
            self.emotion_embeddings = {
                emotion: self._get_embedding(description) for emotion, description in self.emotion_templates.items()
            }

            # Pre-compute topic embeddings
            self.topic_embeddings = {
                topic: self._get_embedding(description) for topic, description in self.topic_templates.items()
            }

        except Exception as e:
            logger.error(f"Failed to initialize emotion detector: {e}")
            self.model = None
            self.tokenizer = None

    def _get_embedding(self, text: str) -> np.ndarray:
        """Extract embeddings from the model in a memory-efficient way"""
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)
            # Use CLS token embedding as the sentence representation
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            # Normalize the embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding[0]

    def detect_emotion(self, text: str) -> Tuple[str, float]:
        """Detect emotion by comparing text embedding similarity with emotion templates"""
        if not text or not self.model or not self.tokenizer:
            return ("concern", 0.5)

        try:
            # Basic preprocessing
            text = text.lower().strip()

            # Get text embedding
            text_embedding = self._get_embedding(text)

            # Calculate similarity with each emotion
            similarities = {}
            for emotion, emotion_embedding in self.emotion_embeddings.items():
                similarity = self._cosine_similarity(text_embedding, emotion_embedding)
                similarities[emotion] = float(similarity)

            # Get the most similar emotion
            best_emotion = max(similarities.items(), key=lambda x: x[1])
            emotion, score = best_emotion

            # Rule-based boosting with improved partial matching
            text_tokens = text.split()
            if any(word in text for word in ["angry", "anger", "mad", "furious"]):
                if "anger" in similarities:
                    score = max(score, 0.85)
                    emotion = "anger"
            elif any(word.startswith("anx") for word in text_tokens) or "worr" in text:
                if "anxiety" in similarities:
                    score = max(score, 0.85)
                    emotion = "anxiety"
            elif any(word.startswith("depress") for word in text_tokens) or "sad" in text:
                if "depression" in similarities:
                    score = max(score, 0.85)
                    emotion = "depression"

            return (emotion, score)

        except Exception as e:
            logger.error(f"Error detecting emotion: {e}")
            return ("concern", 0.5)

    def detect_topic(self, text: str) -> Tuple[str, float]:
        """Detect topic by comparing text embedding similarity with topic templates"""
        if not text or not self.model or not self.tokenizer:
            return ("supportive_listening", 0.5)

        try:
            # Basic preprocessing
            text = text.lower().strip()

            # Get text embedding
            text_embedding = self._get_embedding(text)

            # Calculate similarity with each topic
            similarities = {}
            for topic, topic_embedding in self.topic_embeddings.items():
                similarity = self._cosine_similarity(text_embedding, topic_embedding)
                similarities[topic] = float(similarity)

            # Get the most similar topic
            best_topic = max(similarities.items(), key=lambda x: x[1])
            topic, score = best_topic

            return (topic, score)

        except Exception as e:
            logger.error(f"Error detecting topic: {e}")
            return ("supportive_listening", 0.5)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        return float(np.dot(vec1, vec2))

    def get_standardized_emotion(self, emotion: str) -> str:
        """Map detected emotions to standardized emotions."""
        if not emotion:
            return "neutral"

        lower_emotion = str(emotion).lower()

        # Test-aligned emotion mappings
        emotion_mapping = {
            "anxiety": "anxiety",
            "stress": "anxiety",
            "worry": "anxiety",
            "concern": "anxiety",
            "nervous": "anxiety",
            "fear": "anxiety",
            "panic": "anxiety",
            "sadness": "sadness",
            "depression": "sadness",
            "grief": "sadness",
            "hopelessness": "sadness",
            "despair": "sadness",
            "anger": "anger",
            "frustration": "anger",
            "irritation": "anger",
            "shame": "shame",
            "embarrassment": "shame",
            "humiliation": "shame",
            "guilt": "guilt",
            "regret": "guilt",
            "remorse": "guilt",
            "joy": "happiness",
            "happiness": "happiness",
            "contentment": "happiness",
            "surprise": "surprise",
            "shock": "surprise",
            "amazement": "surprise",
            "neutral": "neutral",
            "calm": "neutral",
            "greeting": "neutral",
        }

        return emotion_mapping.get(lower_emotion, "neutral")

    def get_standardized_topic(self, topic: str) -> str:
        """Map detected topics to standardized test-expected topics."""
        if not topic:
            return "supportive_listening"

        # Use the centralized mapping system
        theme = TherapeuticMappings.find_theme_for_keyword(topic)
        # Add null check to handle Optional[str] return type
        return theme if theme else "supportive_listening"

    def get_standardized_approach(self, topic: str) -> str:
        """Map topics to standardized approach types expected by tests."""
        if not topic:
            return "supportive_listening"

        # First, check if the topic is already a standard approach type
        standard_approaches = [
            "cognitive_behavioral",
            "trauma",
            "interpersonal_therapy",
            "compassion_focused_therapy",
            "behavioral_activation",
            "grief_processing",
            "supportive_listening",
            "stress_management",
            "workplace_stress",
            "stress",
        ]

        lower_topic = topic.lower()

        # Pass through if already a standard approach
        if lower_topic in standard_approaches:
            return lower_topic

        # Then proceed with regular mapping
        approach_mapping = {
            "anxiety": "cognitive_behavioral",
            "depression": "behavioral_activation",
            "trauma": "trauma",
            "ptsd": "trauma",
            "flashback": "trauma",
            "relationship": "interpersonal_therapy",
            "self_worth": "compassion_focused_therapy",
            "grief": "grief_processing",
            "shame": "compassion_focused_therapy",
            "guilt": "cognitive_behavioral",
            "general_support": "supportive_listening",
            "cbt": "cognitive_behavioral",
        }

        return approach_mapping.get(lower_topic, "supportive_listening")

    def get_emotion_mappings(self) -> Dict[str, List[str]]:
        """Return standardized emotion mappings for tests."""
        return {
            "anxiety": ["worry", "concern", "nervous", "anxious"],
            "sadness": ["sad", "depressed", "down", "hopeless", "despair"],
            "hopelessness": ["hopeless", "helpless", "despairing", "worthless", "empty"],
            "anger": ["angry", "frustrated", "irritated", "mad", "furious"],
            "shame": ["embarrassed", "humiliated", "inadequate", "ashamed"],
            "guilt": ["guilty", "remorseful", "regretful"],
            "happiness": ["happy", "joyful", "excited", "pleased"],
            "surprise": ["surprised", "shocked", "amazed", "astonished"],
            "neutral": ["neutral", "calm", "balanced"],
            "fear": ["afraid", "scared", "terrified", "fearful", "panic", "phobia"],
            "loneliness": ["isolated", "alone", "disconnected"],
            "helplessness": ["powerless", "unable to cope", "lacking control"],
            "overwhelm": ["overwhelmed", "stressed", "unable to handle"],
            "confusion": ["confused", "unclear", "uncertain"],
            "hope": ["optimistic", "looking forward", "positive change"],
            "numbness": ["disconnected", "unable to feel"],
            "exhaustion": ["drained", "burnt out", "depleted"],
            "fatigue": ["tired", "exhausted", "worn out", "fatigued", "lethargic"],
            "frustration": ["frustrated", "blocked", "thwarted", "unable to achieve"],
            "rejection": ["rejected", "unwanted", "excluded", "pushed away"],
            "tension": ["tense", "stressed", "strained"],
            "stress": ["stressed", "pressured", "tense", "overwhelmed", "strained"],
            "hurt": ["hurt", "wounded", "pained", "injured", "betrayed"],
            "control": ["control", "controlling", "powerless", "grip", "handle", "manage"],
            "inadequacy": ["inadequate", "not good enough", "inferior", "less than"],
            "self-criticism": ["self-critical", "harsh", "judgmental"],
            "longing": ["yearning", "missing", "craving", "pining", "desire"],
            "emptiness": ["empty", "void", "hollow", "numb", "nothing"],
            "grief": ["grief", "loss", "mourning", "bereavement"],
            "jealousy": ["jealous", "envious", "covetous"],
            "trust": ["trusting", "secure", "safe"],
            "gratitude": ["grateful", "thankful", "appreciative"],
        }

    def detect_emotion_standardized(self, text: str) -> Tuple[str, float]:
        """Detect emotion and standardize the output for testing compatibility."""
        raw_emotion, score = self.detect_emotion(text)
        standardized_emotion = self.get_standardized_emotion(raw_emotion)
        logger.info(f"Raw emotion: {raw_emotion} → Standardized: {standardized_emotion}")
        return (standardized_emotion, score)

    def detect_topic_standardized(self, text: str) -> Tuple[str, float]:
        """Detect topic and standardize the output for testing compatibility."""
        raw_topic, score = self.detect_topic(text)
        standardized_topic = self.get_standardized_topic(raw_topic)
        logger.info(f"Raw topic: {raw_topic} → Standardized: {standardized_topic}")
        return (standardized_topic, score)

    def recommend_approach(self, text: str) -> str:
        """Recommend a standardized therapeutic approach based on topic."""
        topic, _ = self.detect_topic(text)
        standardized_topic = self.get_standardized_topic(topic)
        approach = self.get_standardized_approach(standardized_topic)
        logger.info(f"Topic: {topic} → Standardized topic: {standardized_topic} → Approach: {approach}")
        return approach
