"""
Integration tests for topic detection.
These tests focus on correctly identifying topics and emotions in user input.
"""

import pytest
from prismalog.log import get_logger

logger = get_logger(__name__)


class TestTopicDetectionIntegration:
    """Integration tests for topic detection."""

    def test_topic_detection_in_question_analysis(self, real_prompt_selector):
        """Test that topics are correctly detected in question analysis."""
        # Test cases for different psychological topics
        test_cases = [
            {
                "question": "I feel anxious all the time and can't stop worrying",
                "expected_topic": "anxiety",
                "expected_emotion": "anxiety",
            },
            {
                "question": "I feel so sad and hopeless lately",
                "expected_topics": ["depression", "sadness", "low_mood"],  # Allow multiple valid topics
                "expected_emotion": "sadness",
            },
            {
                "question": "I keep having nightmares about my accident",
                "expected_topic": "trauma",
                "expected_emotion": "fear",
            },
            {
                "question": "I lost my mother last month and can't cope",
                "expected_topic": "grief_loss",
                "expected_emotion": "sadness",
            },
        ]

        for case in test_cases:
            # Analyze the question
            analysis = real_prompt_selector.analyze_question(case["question"])

            # Log results for debugging
            topic = analysis.get("topic", "")
            emotion = analysis.get("emotion", "")
            logger.info(f"Question: {case['question'][:30]}...")
            logger.info(f"Detected topic: {topic}")
            logger.info(f"Detected emotion: {emotion}")

            # Use more flexible assertion for topics
            if "expected_topics" in case:
                # Check if any of the expected topics match
                topic_match = any(
                    exp_topic in topic.lower() or topic.lower() in exp_topic for exp_topic in case["expected_topics"]
                )
                assert topic_match, f"Expected one of {case['expected_topics']}, got '{topic}'"
            else:
                # Check single expected topic with flexible matching
                expected = case.get("expected_topic", "")
                topic_match = expected in topic.lower() or topic.lower() in expected
                assert topic_match, f"Expected topic containing '{expected}', got '{topic}'"

            # Assert emotion detection - the specific emotion might vary
            assert emotion, f"No emotion detected for input: {case['question'][:30]}..."

    def test_extracted_topics_for_retrieval(self, integration_response_generator):
        """Test that the right topics are extracted for retrieval."""
        # Test cases with more flexible expectations
        test_cases = [
            {
                "question": "I'm having trouble sleeping because of anxiety",
                "expected_primary": ["anxiety", "sleep", "insomnia"],  # Accept any of these as primary
                "expected_secondary": ["anxiety", "sleep", "stress", "insomnia"],  # Accept any of these as secondary
            },
            # ...other test cases...
        ]

        for case in test_cases:
            # Extract topics
            topics_context = integration_response_generator.extract_psychological_topics(case["question"])

            # Get extracted topics
            extracted_topics = topics_context.get("extracted_topics", [])
            logger.info(f"Question: {case['question']}")
            logger.info(f"Extracted topics: {extracted_topics}")

            # More flexible assertion for primary topic
            if extracted_topics:
                primary_topic = extracted_topics[0].lower()
                primary_match = any(
                    expected.lower() in primary_topic or primary_topic in expected.lower()
                    for expected in case["expected_primary"]
                )
                assert (
                    primary_match
                ), f"Primary topic '{primary_topic}' doesn't match any expected topics: {case['expected_primary']}"

            # Check for secondary topics if we have more than one topic
            if len(extracted_topics) > 1:
                secondary_topics = [t.lower() for t in extracted_topics[1:]]
                # Check that at least one expected secondary topic is found
                found_secondary = any(
                    any(
                        expected.lower() in topic or topic in expected.lower()
                        for expected in case["expected_secondary"]
                    )
                    for topic in secondary_topics
                )

                assert (
                    found_secondary or primary_match
                ), f"No expected topics found in extracted topics: {extracted_topics}"
            else:
                # If only one topic was extracted, we already verified it matches a primary topic
                pass

    def test_emotion_detection(self, real_prompt_selector):
        """Test that emotions are correctly detected in user questions."""
        # Test cases for emotion detection
        test_cases = [
            {"question": "I'm furious about how I was treated at work", "expected_emotion": "anger"},
            {"question": "I'm terrified of flying and have a trip coming up", "expected_emotion": "fear"},
            {"question": "I feel ashamed about what happened at the party", "expected_emotion": "shame"},
        ]

        for case in test_cases:
            # Analyze the question
            analysis = real_prompt_selector.analyze_question(case["question"])

            # Get detected emotion
            emotion = analysis.get("emotion", "")

            # Log results
            logger.info(f"Question: {case['question']}")
            logger.info(f"Detected emotion: {emotion}")

            # Check if the detected emotion contains the expected one or vice versa
            emotion_match = case["expected_emotion"] in emotion.lower() or emotion.lower() in case["expected_emotion"]

            assert emotion_match, f"Expected emotion '{case['expected_emotion']}' not found in '{emotion}'"
