"""
2025, Dresden Alexey Obukhov, alexey.obukhov@hotmail.com

This module provides utility functions for text cleaning and decoding.
"""

import re
import html
import traceback
import pandas as pd
from typing import Any
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
    return text


def tokenize_and_lemmatize(text: str, logger: Any = None) -> str:
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
            logger.debug(f"Tokenizing text (first 50 chars): '{text[:50]}...'")
            
        doc = nlp(text)
        cleaned_tokens = [
            token.lemma_.lower() for token in doc
            if (token.lemma_.lower() in keep_words) or (not token.is_stop and not token.is_punct and not token.is_space)
        ]
        cleaned_text = " ".join(cleaned_tokens)
        
        if logger:
            logger.debug(f"Lemmatized text (first 50 chars): '{cleaned_text[:50]}...'")
            
        return cleaned_text.strip()
        
    except Exception as e:
        if logger:
            logger.error(f"Error in tokenize_and_lemmatize: {str(e)}\n{traceback.format_exc()}")
        return text


def create_context(df: pd.DataFrame, max_context_turns: int = 3, logger: ColoredLogger = None) -> None:
    """
    Modifies the DataFrame inplace by generating structured conversation context for each turn.

    Args:
        df (pd.DataFrame): The input DataFrame containing 'questionID', 'interactionID', and text columns.
        max_context_turns (int): Maximum number of previous turns to include in context.
        logger (ColoredLogger, optional): Logger instance for debugging.

    Returns:
        None: The function modifies the DataFrame inplace.
    """
    if logger:
        logger.name = "CreateContext"
        logger.info("Creating conversation context inplace with max %d turns per context.", max_context_turns)
    else:
        print("Creating conversation context inplace with max %d turns per context." % max_context_turns)

    # Ensure data is sorted correctly for rolling context
    df.sort_values(['questionID', 'interactionID'], ascending=[True, True], inplace=True)

    # Dictionary to store deque for each questionID
    context_map = {}

    def generate_context(row):
        """Generates rolling context for each row using deque."""
        q_id = row['questionID']
        if q_id not in context_map:
            print(f"Question ID {q_id} not found in context_map. adding it...")
            context_map[q_id] = deque(maxlen=max_context_turns)

        current_turn = f"Q: {row['questionTitle']} A: {row['answerText']}"
        context = " ".join(context_map[q_id])
        context_map[q_id].append(current_turn)
        return context

    # Apply rolling context generation
    df['context'] = df.apply(generate_context, axis=1)

    if logger:
        logger.info("Context creation completed successfully (vectorized & inplace).")
    else:
        print("Context creation completed successfully (vectorized & inplace).")


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

        "emotional_support": [
            "help", "support", "understand", "listen", "care", "concern"
        ],
    }
