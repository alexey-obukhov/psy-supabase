def test_simple_context_extraction():
    """Simple direct test of context extraction logic."""
    from tests.conftest import DEFAULT_APPROACH, DEFAULT_TOPIC, logger

    # Mock the necessary objects
    detected_topic = "anxiety"
    extracted_topics = ["stress_management"]
    approach_type = "cognitive_behavioral"
    pain_point = {"topic": "fear", "detected": True}

    # Create a simple context selection function
    def select_context():
        """Determine the most specific context based on priority."""
        if detected_topic and detected_topic != DEFAULT_TOPIC:
            return detected_topic.replace(" ", "_").lower()
        if extracted_topics and extracted_topics[0] != DEFAULT_TOPIC:
            return extracted_topics[0].replace(" ", "_").lower()
        if approach_type and approach_type != DEFAULT_APPROACH:
            return approach_type.replace(" ", "_").lower()
        if pain_point and pain_point.get("topic"):
            return pain_point["topic"].replace(" ", "_").lower()
        return "therapeutic_dialogue"  # Default fallback

    # Test each priority level
    context = select_context()
    logger.info(f"Selected context with all data: {context}")
    assert context == "anxiety"

    # Test without detected topic
    detected_topic = DEFAULT_TOPIC
    context = select_context()
    logger.info(f"Selected context without detected topic: {context}")
    assert context == "stress_management"

    # Test without extracted topics
    detected_topic = DEFAULT_TOPIC
    extracted_topics = [DEFAULT_TOPIC]
    context = select_context()
    logger.info(f"Selected context without useful topics: {context}")
    assert context == "cognitive_behavioral"

    # Test fallback to pain point
    approach_type = DEFAULT_APPROACH
    context = select_context()
    logger.info(f"Selected context from pain point: {context}")
    assert context == "fear"

    # Test fallback to default
    pain_point = {}
    context = select_context()
    logger.info(f"Selected default context: {context}")
    assert context == "therapeutic_dialogue"


def test_real_context_extraction():
    """Test the real context extraction logic in the program."""
    from unittest.mock import MagicMock

    from psy_supabase.core.response_generator import ResponseGenerator
    from tests.conftest import DEFAULT_APPROACH, DEFAULT_TOPIC, logger

    # Mock the dependencies
    text_generator = MagicMock()
    db_manager = MagicMock()
    prompt_selector = MagicMock()

    # Initialize the ResponseGenerator with mocked dependencies
    response_generator = ResponseGenerator(
        text_generator=text_generator,
        db_manager=db_manager,
        prompt_selector=prompt_selector,
    )

    # Mock the necessary inputs
    user_question = "How can I manage my anxiety?"
    topics_context = {"topic": "anxiety", "emotion": "concern"}
    pain_point_results = {"pain_point": {"topic": "fear", "detected": True}, "approach_type": "cognitive_behavioral"}
    metadata = {"session_id": "test_session"}

    # Mock the behavior of prompt_selector to simulate ML-based topic extraction
    prompt_selector.analyze_question.return_value = {
        "topic": "anxiety",
        "confidence": 0.8,
        "emotion": "concern",
        "emotion_intensity": 0.7,
        "emotion_confidence": 0.7,
    }
    prompt_selector.generate_category_info.return_value = {"anxiety": {"description": "Managing anxiety"}}
    prompt_selector.determine_topic.return_value = "anxiety"

    # Call the real method to extract the context
    context, updated_metadata = response_generator.determine_final_context(
        user_question=user_question,
        topics_context=topics_context,
        pain_point_results=pain_point_results,
        metadata=metadata,
    )

    # Log and assert the results
    logger.info(f"Extracted context: {context}")
    logger.info(f"Updated metadata: {updated_metadata}")

    # Assertions to verify the context extraction logic
    assert context == "anxiety", "Context should prioritize the detected topic."
    assert updated_metadata["pain_point_detected"] is True, "Pain point should be detected in metadata."
    assert updated_metadata["therapeutic_approach"] == "cognitive_behavioral", "Approach type should be updated."
