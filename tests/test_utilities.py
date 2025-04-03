"""
Tests for utility functions in the psy_supabase package.

This module contains tests for various utility functions used throughout
the psy_supabase package, primarily focusing on text analysis, embedding
utilities, and common text processing operations.

The tests here don't require database access and are designed to run quickly
as unit tests, verifying the core functionality of utility functions.
"""
from psy_supabase.utilities.embedding_utils import detect_repetition_pattern

def test_detect_repetition_pattern():
    """Test detection of repetition patterns."""
    original_question = "Why do I feel anxious all the time?"
    current_question = "Why am I always feeling anxious?"

    # Create test data with 'question' and 'similarity' keys
    similar_questions = [
        {'question': 'Why do I feel anxious?', 'similarity': 0.85},
        {'question': 'What causes my anxiety?', 'similarity': 0.75},
        {'question': 'How to stop feeling anxious?', 'similarity': 0.72}
    ]

    # Call the function
    result = detect_repetition_pattern(
        original_question,
        current_question,
        similar_questions
    )

    # Verify the structure of the result
    assert isinstance(result, dict)

    # Verify count calculation (3 questions with similarity > 0.7)
    assert 'count' in result
    assert result['count'] == 3

    # Verify recurring terms extraction
    assert 'recurring_terms' in result
    assert isinstance(result['recurring_terms'], list)
    assert 'anxious' in result['recurring_terms']
    assert 'feel' not in result['recurring_terms']  # Should be in keep_words

    # Verify fixation detection (≥3 occurrences = fixation)
    assert 'is_fixation' in result
    assert result['is_fixation'] == True

    # Verify intensity calculation
    assert 'intensity' in result
    assert 0 <= result['intensity'] <= 1.0
    assert result['intensity'] == 3/5  # 3 occurrences / 5

def test_detect_repetition_pattern_no_fixation():
    """Test repetition pattern detection with fewer occurrences (no fixation)."""
    original_question = "Why do I feel sad?"
    current_question = "Why am I feeling depressed today?"

    # Only one similar question with high similarity
    similar_questions = [
        {'question': 'What makes me feel down?', 'similarity': 0.78}
    ]

    result = detect_repetition_pattern(
        original_question,
        current_question,
        similar_questions
    )

    assert result['count'] == 1
    assert result['is_fixation'] == False  # Less than 3 occurrences
    assert result['intensity'] == 1/5      # 1 occurrence / 5

def test_detect_repetition_pattern_empty():
    """Test repetition pattern detection with no similar questions."""
    original_question = "Why do I feel sad?"
    current_question = "Why am I feeling depressed today?"

    # No similar questions
    similar_questions = []

    result = detect_repetition_pattern(
        original_question,
        current_question,
        similar_questions
    )

    assert result['count'] == 0
    assert result['is_fixation'] == False
    assert result['intensity'] == 0
    assert 'recurring_terms' in result
    assert isinstance(result['recurring_terms'], list)

def test_detect_repetition_pattern_keep_words_filtering():
    """Test that keep_words are properly filtered from recurring terms."""
    original_question = "Why do I feel so sad and depressed all the time?"
    current_question = "I feel sad and depressed so often, why is that?"

    similar_questions = [
        {'text': 'Why do I feel sad?', 'similarity': 0.85},
        {'text': 'Why am I depressed?', 'similarity': 0.80}
    ]

    # Import keep_words to verify our test logic
    from psy_supabase.utilities.keep_words import keep_words

    result = detect_repetition_pattern(
        original_question,
        current_question,
        similar_questions
    )

    # Verify recurring terms list contains meaningful terms but not common words
    assert 'sad' in result['recurring_terms']
    assert 'depressed' in result['recurring_terms']

    # Check that common words are filtered out
    common_words = ['feel', 'why', 'and', 'the', 'is', 'that', 'do', 'so', 'all', 'time', 'often']
    for word in common_words:
        # Verify each common word is either not in recurring_terms or is explicitly in keep_words
        if word in result['recurring_terms']:
            assert word not in keep_words, f"Common word '{word}' found in recurring_terms but not in keep_words"

    # Verify that all returned significant terms meet criteria
    for term in result['recurring_terms']:
        # Terms should be longer than 2 characters and not in keep_words
        assert len(term) > 2, f"Term '{term}' is too short"
        assert term not in keep_words, f"Term '{term}' is in keep_words but still included"

def test_detect_repetition_pattern_only_common_words():
    """Test behavior when questions only share common words."""
    original_question = "How do I do this?"
    current_question = "How can I do that?"

    similar_questions = [
        {'text': 'What do I do now?', 'similarity': 0.75}
    ]

    result = detect_repetition_pattern(
        original_question,
        current_question,
        similar_questions
    )

    # Since there are only common words, recurring_terms should be empty
    assert len(result['recurring_terms']) == 0

    # Other values should still be calculated correctly
    assert result['count'] == 1
    assert result['is_fixation'] == False
    assert result['intensity'] == 0.2  # 1/5
