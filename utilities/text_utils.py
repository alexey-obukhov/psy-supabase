"""
2025, Dresden Alexey Obukhov, alexey.obukhov@hotmail.com

This module provides utility functions for text cleaning and decoding. These functions are used to handle
Unicode characters, unwanted symbols, and other text normalization tasks.

Functions:
1. clean_text: Cleans and decodes text to handle Unicode characters and unwanted symbols.

Usage:
    from utilities.text_utils import create_context, process_chunk, get_embedding, clean_text
    from utilities.text_utils import create_context, process_chunk, get_embedding, clean_text
"""

import re
import html
import spacy
import traceback
import pandas as pd
from typing import Any
from collections import deque
from school_logging.log import ColoredLogger
from utilities.keep_words import keep_words

import spacy
import traceback
import pandas as pd
from typing import Any
from collections import deque
from school_logging.log import ColoredLogger
from utilities.keep_words import keep_words


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
    #text = text.replace('"', '""')                        # Escape double quotes for SQL
    return text


def tokenize_and_lemmatize(text: str, logger: Any, nlp=spacy.load("en_core_web_sm")) -> str:
    """
    Static method to tokenize and lemmatize text.

    Args:
        text (str): Text to process
        logger (Any): Logger instance
        nlp: SpaCy model (default loads en_core_web_sm)

    Returns:
        str: Processed text
    """
    try:
        logger.debug(f"Tokenizing text (first 50 chars): '{text[:50]}...'")
        doc = nlp(text)
        cleaned_tokens = [
            token.lemma_.lower() for token in doc
            if (token.lemma_.lower() in keep_words) or (not token.is_stop and not token.is_punct and not token.is_space)
        ]
        cleaned_text = " ".join(cleaned_tokens)
        logger.debug(f"Lemmatized text (first 50 chars): '{cleaned_text[:50]}...'")
        return cleaned_text.strip()
    except Exception as e:
        logger.error(f"Error in tokenize_and_lemmatize: {str(e)}\n{traceback.format_exc()}")
        return ""


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
