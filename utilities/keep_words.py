"""
2025, Dresden Alexey Obukhov, alexey.obukhov@hotmail.com

This module defines a tuple of words that should be preserved during the tokenization and lemmatization process
in the `tokenize_and_lemmatize` function of the `data_augmentation.py` module. These words include key pronouns
and verbs that are important for maintaining the context and meaning of the text data.

Purpose:
The `keep_words` tuple is used to ensure that certain important words are not removed during the tokenization
and lemmatization process, which helps in preserving the essential structure and meaning of the text.

Usage:
    from utilities.keep_words import keep_words
"""

keep_words = ("these", "shouldn't", "i'll", "isn't", "i'd",
              "you've", "hasn't", "doesn't", "or",
              "what", "she'd", "where", "won't", "cannot", "who",
              "he'd", "those", "tought", "didn't", "because",
              "that", "he's", "he'll", "had", "are", "we'll",
              "why", "i'm", "if", "they'd", "could", "hadn't",
              "you'll", "did", "when", "be", "they're", "as",
              "do", "they'll", "we're", "should", "have", "you'd",
              "a", "does", "being", "you're", "we've", "whom", "wouldn't",
              "the", "couldn't", "haven't", "which", "but", "i've", "am", "this",
              "it's", "she's", "was", "weren't", "can't", "having", "were", "been",
              "doing", "don't", "they've", "an", "is", "wasn't", "and", "we'd", "she'll",
              "mustn't", "shan't", "would", "has", "aren't", "how")
