from typing import Tuple, Dict, List, Any
import re
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from prismalog.log import get_logger

logger = get_logger(__name__)

class SemanticEmotionDetector:
    """
    Lightweight emotion and topic detector using a small language model
    optimized for semantic similarity.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize with a very small but effective sentence transformer model.
        This model is only ~80MB vs 1.6GB+ for BART models.
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
                "anger": "feeling angry, frustrated, irritated, mad or furious about something that happened",
                "anxiety": "feeling worried, nervous, anxious, afraid or extremely anxious about what might happen",
                "depression": "feeling sad, empty, hopeless, depressed, or not enjoying anything anymore",
                "shame": "feeling embarrassed, humiliated, or inadequate about oneself",
                "guilt": "feeling responsible, regretful, or remorseful about something done wrong",
                "joy": "feeling happy, excited, pleased, or content about something good",
                "surprise": "feeling shocked, amazed, or caught off guard by something unexpected",
                "concern": "feeling uneasy, troubled, or bothered about a situation",
                # Add greeting category for emotion detection
                "greeting": "saying hello, hi, good morning, or checking in without expressing a specific emotion"
            }

            # Add topic templates including greeting
            self.topic_templates = {
                "anxiety": "anxiety, worried, nervous, fear, panic, stress, phobia; feeling extremely anxious",
                "depression": "depression, sad, low mood, hopelessness; don't enjoy anything anymore; lack of interest",
                "supportive_listening": "listening, support, understanding, validation, empathy",
                "trauma": "trauma, ptsd, abuse, neglect, painful experiences",
                "relationship_issues": "relationship, partner, marriage, dating, breakup",
                "self-esteem": "self esteem, confidence, self worth, inadequate, not good enough",
                "grief_loss": "grief, loss, death, mourning, bereavement",
                "shame": "shame, embarrassment, humiliation, social rejection",
                "guilt": "guilt, regret, remorse, responsibility, blame",
                # Add greeting category for topic detection
                "greeting": "hello, hi, hey, good morning, good day, greetings, checking in, how are you, what's up, introduction, small talk"
            }

            # Pre-compute emotion embeddings
            self.emotion_embeddings = {
                emotion: self._get_embedding(description)
                for emotion, description in self.emotion_templates.items()
            }

            # Pre-compute topic embeddings
            self.topic_embeddings = {
                topic: self._get_embedding(description)
                for topic, description in self.topic_templates.items()
            }

        except Exception as e:
            logger.error(f"Failed to initialize emotion detector: {e}")
            self.model = None
            self.tokenizer = None

    def _get_embedding(self, text: str) -> np.ndarray:
        """Extract embeddings from the model in a memory-efficient way"""
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt",
                                   padding=True, truncation=True, max_length=128)
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
