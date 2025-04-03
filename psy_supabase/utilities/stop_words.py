"""
Common English stop words for text processing.

This module provides a set of common English stop words that are typically filtered
out during natural language processing tasks like keyword extraction, topic modeling,
and semantic analysis. Stop words are high-frequency words that typically don't carry
significant meaning on their own (articles, prepositions, pronouns, etc.).

Usage:
    from psy_supabase.utilities.stop_words import stop_words

    # Filter out stop words from a text
    words = [word for word in text.split() if word.lower() not in stop_words]

Note:
    This list is intentionally kept small to focus on the most common stop words.
    For more comprehensive lists, consider using NLTK or spaCy libraries.
"""

stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about',
              'as', 'of', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been',
              'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall',
              'should', 'may', 'might', 'must', 'can', 'could', 'i', 'you', 'he', 'she',
              'it', 'we', 'they', 'this', 'that', 'these', 'those'}