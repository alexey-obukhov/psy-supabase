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

keep_words = ("i", "me", "my", "mine", "you", "your", "yours", "he", "him", "his", "she",
              "her", "hers", "it", "its", "we", "us", "our", "ours", "they", "them", "their",
              "theirs", "be", "am", "is", "are", "was", "were", "been", "being", "have",
              "has", "had", "having", "do", "does", "did", "doing")
