"""
Utility functions for text processing and system management.

This module provides various utility functions used throughout the PSY Supabase application
for text cleaning, natural language processing, memory management, and configuration:

Text Processing:
- clean_text: Sanitizes and normalises text by handling Unicode, HTML entities, and special characters
- tokenize_and_lemmatize: Processes text using spaCy for advanced NLP tasks

System Management:
- cleanup_memory: Frees GPU memory and performs garbage collection
- parse_bool_env: Safely parses boolean environment variables

Data Resources:
- load_enhanced_mental_health_taxonomy: Provides a comprehensive taxonomy of mental health terms
  organized by categories like Depression, Anxiety, Trauma, etc.

These utilities are designed to be reusable across different components of the application
and provide consistent text processing and system management capabilities.
"""

import re
import html
import traceback
import pandas as pd
from typing import Optional
from collections import deque
from school_logging.log import ColoredLogger
from psy_supabase.utilities.keep_words import keep_words
from psy_supabase.utilities.nlp_utils import get_spacy_model


def clean_text(text: str) -> str:
    """
    Cleans and decodes text to handle Unicode characters, unwanted symbols, and escape single quotes.

    Args:
        text (str): The text to be cleaned and decoded.

    Returns:
        str: The cleaned and decoded text.
    """
    text = text.encode('utf-8').decode('unicode_escape')  # Decode Unicode escape sequences
    text = html.unescape(text)                            # Unescape HTML entities
    text = re.sub(r"\u2019", "'", text)                   # Replace right single quotation mark with apostrophe
    text = re.sub(r"\u2014", "-", text)                   # Replace em dash with hyphen
    text = re.sub(r"\u201c", '"', text)                   # Replace left double quotation mark with double quote
    text = re.sub(r"\u201d", '"', text)                   # Replace right double quotation mark with double quote
    text = re.sub(r"\u2026", "...", text)                 # Replace ellipsis with three dots
    text = re.sub(r"[^a-zA-Z0-9\s.,?!'\":-]", "", text)   # Remove unwanted characters except ':' and '-
    text = re.sub(r"\s+", " ", text).strip()              # Remove extra whitespace
    text = re.sub(r'\n+', '\n', text)                     # Remove redundant newlines
    text = text.replace("'", "''")                        # Escape single quotes for SQL
    text = text.replace('   ', ' ')                       # Replace multiple spaces with single space
    text = text.replace('  ', ' ')                        # Replace double spaces with single space
    return text

def tokenize_and_lemmatize(text: str,
                           logger: Optional[ColoredLogger] = None
                           ) -> str:
    """
    Tokenize and lemmatize text using spaCy.

    Args:
        text (str): Text to process
        logger (Any, optional): Logger instance for debug messages

    Returns:
        str: Processed text with tokens lemmatized and filtered
    """
    nlp = get_spacy_model()
    if nlp is None:
        if logger:
            logger.error("spaCy model not available. Cannot tokenize text.")
        return text

    try:
        if logger:
            logger.debug("Tokenizing text (first 50 chars): '%s...'", text[:50])

        doc = nlp(text)
        cleaned_tokens = [
            token.lemma_.lower() for token in doc
            if (token.lemma_.lower() in keep_words) or (not token.is_stop and not token.is_punct and not token.is_space)
        ]
        cleaned_text = " ".join(cleaned_tokens)

        if logger:
            logger.debug("Lemmatized text (first 50 chars): '%s...'", cleaned_text[:50])

        return cleaned_text.strip()

    except Exception as e:
        if logger:
            logger.error("Error in tokenize_and_lemmatize: %s\n%s", str(e), traceback.format_exc())
        return text

def debug_errors(logger=None):
    """
    Decorator to debug errors in methods with detailed information.

    This enhanced decorator captures rich debugging information when exceptions occur,
    including argument types, values, and detailed stack traces. It's designed to
    help diagnose complex issues like the "'list' object has no attribute 'get'" error.

    Args:
        logger: Optional logger instance. If None, will use a default logger.

    Returns:
        Decorator function that wraps methods for detailed error reporting

    Example:
        @debug_errors(logger=my_logger)
        def analyze_emotional_response_to_interaction(self, interaction_data):
            # Method implementation
    """
    # Get a default logger if none provided
    if logger is None:
        from school_logging.log import ColoredLogger
        logger = ColoredLogger("ErrorDebug")

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                import inspect
                import traceback

                # Get more information about the arguments
                arg_info = []
                for i, arg in enumerate(args):
                    if i == 0:  # Skip 'self'
                        continue
                    arg_info.append(f"Arg {i}: {type(arg).__name__}")
                    # Print detailed info for lists and dicts
                    if isinstance(arg, list):
                        arg_info.append(f"  List length: {len(arg)}")
                        if arg:
                            arg_info.append(f"  First element type: {type(arg[0]).__name__}")
                            if isinstance(arg[0], dict):
                                arg_info.append(f"  Keys: {list(arg[0].keys())}")
                    elif isinstance(arg, dict):
                        arg_info.append(f"  Dict keys: {list(arg.keys())}")

                # Get source code around the error
                frame = inspect.currentframe()
                frames = inspect.getouterframes(frame)
                error_line = None
                error_code = None

                for f in frames:
                    if f.filename == inspect.getfile(func.__code__):
                        error_line = f.lineno
                        try:
                            lines, _ = inspect.getsourcelines(f.frame)
                            error_code = "".join(lines[:5])  # First 5 lines
                        except:
                            error_code = "Could not retrieve source"
                        break

                # Log detailed error information
                logger.error(f"Error in {func.__name__} at line {error_line}: {str(e)}")
                logger.error(f"Arguments: {', '.join(arg_info)}")
                logger.error(f"Code context: \n{error_code}")
                logger.error(f"Traceback: {traceback.format_exc()}")

                # Re-raise the exception
                raise
        return wrapper
    return decorator

def cleanup_memory():
    """Clean up GPU memory."""
    import torch
    from psy_supabase.core.model_manager import get_model_manager
    from school_logging.log import ColoredLogger
    logger = ColoredLogger("MemoryCleanup")
    # --- Use GPU if available ---
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define model name at module level for consistency
    model_name = "microsoft/phi-1_5"
    try:
        model_manager = get_model_manager(model_name, device)
        logger.info("Freeing GPU memory...")
        model_manager.free_memory()
        torch.cuda.empty_cache()
        # Also call Python's garbage collector
        import gc
        gc.collect()
        logger.info("Memory cleanup completed")
    except Exception as e:
        logger.error("Error during memory cleanup: %s", e)

def parse_bool_env(env_var, default=False):
    """Parse boolean environment variables properly."""
    import os
    value = os.environ.get(env_var, str(default)).lower()
    return value in ("true", "1", "yes", "y", "t")

def load_enhanced_mental_health_taxonomy():
    """
    Load an enhanced mental health taxonomy based on professional frameworks.
    Combines LIWC psychological dimensions with DSM-5 terminology.
    """
    return {
        "Depression": [
            'melancholy', 'fatigue', 'tired', 'unmotivated', "don't enjoy",
            'insomnia', 'exhausted', 'weight', "can't sleep", 'appetite',
            'guilt', 'concentration', 'depressed', 'no motivation', 'unhappy',
            'empty', 'suicidal', 'no energy', 'no interest', 'lost interest',
            'sad', 'hypersomnia', 'indecisive', 'numb', 'anhedonia', "can't eat",
            'psychomotor', "can't enjoy", 'worthless', 'gloomy', 'despair', 'miserable',
            'emptiness', 'meaningless', 'depression', 'hopeless'
        ],

        "Anxiety": [
            'nervous', 'on edge', 'frightened', 'catastrophizing', 'dread', 'arousal',
            'apprehensive', 'overwhelmed', 'anxious', 'racing thoughts', 'uneasy',
            'hypervigilant', 'stress', 'panic', 'worry', 'social anxiety', 'scared',
            'obsessive', 'avoidance', 'tense', 'irritable', 'fear', 'restless', 'compulsive',
            'phobia', 'worried', 'anxiety', 'overthinking', 'performance anxiety'
        ],

        "Trauma": [
            'neglect', 'avoidance', 'nightmare', 'trauma', 'violence',
            'hyperarousal', 'harass', 'ptsd', 'assaulted', 'startle',
            'horror', 'childhood trauma', 'threat', 'helpless', 'assault', 'intrusion',
            'flashback', 'emotional dysregulation', 'survivor', 'accident', 'danger',
            'numb', 'triggered', 'traumatized', 'dissociate', 'disaster',
            'victim', 'abuse', 'hypervigilant', 'abused', 'victimized', 'detached'
        ],

        # Workplace trauma and abuse
        "Workplace Trauma": [
            # Primary workplace abuse terms (stronger matches)
            "workplace abuse", "work abuse", "boss abuse", "manager abuse",
            "toxic workplace", "hostile work", "bullied at work", "harassed at work",
            "workplace harassment", "workplace bullying", "abused at work",
            "work trauma", "workplace trauma", "toxic boss", "toxic manager",
            "abusive supervisor", "boss bully", "manager bully", "workplace bully",

            # Secondary workplace terms
            "mobbing", "work stress", "threatened at work", "intimidated at work",
            "humiliated at work", "workplace retaliation", "work mistreatment",
            "gaslighting at work", "workplace injustice", "unfair treatment at work",

            # Additional workplace problem indicators
            "discriminated at work", "work discrimination", "hostile environment",
            "career sabotage", "workplace violence", "demotion", "unfair review",
            "fired unfairly", "targeted at work", "work anxiety", "job trauma",
            "toxic team", "toxic coworker", "work harassment", "work bullying",

            # Common phrases
            "hate my job", "hate my boss", "terrible workplace", "awful job",
            "hostile boss", "mean coworker", "being bullied", "being harassed",
            "work is hell", "office politics", "power abuse", "authority abuse",
            "work ptsd", "verbally abused", "yelled at", "screamed at"
        ],

        # Relationship-related topics
        "Relationship": [
            "relationship", "marriage", "partner", "boyfriend", "girlfriend", "husband",
            "wife", "spouse", "couple", "dating", "significant other", "ex", "breakup",
            "divorce", "separated", "together", "commitment", "trust", "betrayal",
            "cheating", "infidelity", "jealousy", "communication", "argument", "fight",
            "romantic", "love", "loved", "loving", "connection", "attachment"
        ],

        # Heartbreak and healing
        "Heartbreak": [
            "heartbreak", "heartbroken", "broken heart", "broken up", "dumped",
            "rejected", "betrayed", "abandoned", "alone", "lonely", "miss them",
            "missing them", "moving on", "get over", "heal", "healing", "closure",
            "broken heart", "love pain", "hurt by love", "hurt by them", "never again",
            "brake my heart", "break my heart", "no more love", "trust again",
            "never trust", "fall in love", "falling for someone", "vulnerable"
        ],
        "Interpersonal": [
            "relationship", "marriage", "partner", "spouse", "family", "friend", "colleague", "conflict",
            "intimacy", "attachment", "boundary", "communication", "trust", "abandonment", "rejection",
            "loneliness", "isolation", "connection", "breakup", "divorce", "separation", "betrayal",
            "argument", "misunderstanding"
        ],

        "Identity": [
            "self-esteem", "identity", "self-worth", "confidence", "imposter", "shame", "perfectionism",
            "failure", "inadequacy", "self-doubt", "body image", "self-criticism", "self-compassion",
            "validation", "purpose", "meaning", "values", "authentic", "true self", "gender", "sexuality",
            "culture"
        ],

        "Adjustment": [
            "grief", "loss", "bereavement", "change", "transition", "adaptation", "adjustment", "stress",
            "coping", "resilience", "life stage", "retirement", "career", "moving", "relocation",
            "major life event", "crisis", "upheaval", "uncertainty", "decision-making", "crossroads",
            "opportunity", "challenge"
        ],

        "Behavior": [
            "addiction", "substance", "alcohol", "drug", "gambling", "compulsive", "habit", "dependence",
            "withdrawal", "craving", "relapse", "recovery", "abstinence", "moderation", "harm-reduction",
            "impulse control", "self-regulation", "behavioral therapy", "reinforcement", "trigger"
        ],

        "Wellness": [
            "mindfulness", "meditation", "relaxation", "self-care", "resilience", "growth", "strength",
            "resource", "wellness", "prevention", "maintenance", "balance", "harmony", "fulfillment",
            "joy", "satisfaction", "gratitude", "meaning", "purpose", "flourishing", "thriving", "vitality"
        ],

        "Cognition": [
            "thought", "belief", "cognition", "distortion", "schema", "assumption", "automatic thought",
            "rumination", "worry", "attention", "memory", "concentration", "problem-solving", "decision-making",
            "perception", "interpretation", "reframe", "perspective", "mindset", "attribution"
        ],

        "Grief & Loss": [
            "bereavement", "loss", "mourning", "acceptance", "denial", "anger", "bargaining", "depression",
            "adaptation", "adjustment", "memorialization", "letting go", "moving on", "honoring", "memory"
        ],

        "Self-Compassion": [
            "self-kindness", "common humanity", "mindfulness", "self-criticism", "self-care", "forgiveness",
            "acceptance", "compassionate voice", "inner peace", "empathy"
        ],

        "Guilt & Shame": [
            "guilt", "shame", "self-judgment", "self-blame", "embarrassment", "regret", "wrongdoing", "redemption",
            "forgiveness", "moral distress", "humiliation", "self-forgiveness"
        ],

        "Obsessive-Compulsive Disorder (OCD)": [
            "obsession", "compulsion", "ritual", "perfectionism", "control", "anxiety", "reassurance-seeking",
            "intrusive thought", "cleaning", "checking", "counting", "hoarding"
        ],

        "Suicidality & Self-Harm": [
            "suicidal", "self-harm", "cutting", "despair", "hopelessness", "crisis", "emotional pain",
            "coping", "prevention", "life-threatening", "overwhelming"
        ],

        "emotional support": [
            "help", "support", "understand", "listen", "care", "concern"
        ],
    }
