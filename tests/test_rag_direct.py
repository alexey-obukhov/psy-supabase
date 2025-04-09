def test_rag_direct_save():
    """Test RAGProcessor directly saves to the database."""
    from psy_supabase.core.rag_processor import RAGProcessor
    from unittest.mock import MagicMock, patch

    # Create mocks with clear return values
    mock_db = MagicMock()
    mock_generator = MagicMock()
    mock_generator.generate_text.return_value = "This is a test response"

    # Set up processor
    with patch('psy_supabase.core.rag_processor.PromptSelector') as mock_selector_cls:
        # Create a mock for the prompt selector instance
        mock_selector = MagicMock()
        mock_selector_cls.return_value = mock_selector

        # Configure the mock selector
        mock_selector.generate_prompt.return_value = "Test prompt"
        mock_selector.generate_category_info.return_value = {"Test Category": 0.9}
        mock_selector.determine_topic.return_value = "test_topic"
        mock_selector.analyze_question.return_value = {"topic": "test", "confidence": 0.9}

        # Create processor
        processor = RAGProcessor(db_manager=mock_db, generator=mock_generator)

        # Now force save_interaction to be something we can track
        save_mock = MagicMock()
        mock_db.save_interaction = save_mock

        # Override toxicity check
        processor.check_toxicity = lambda text: {"is_toxic": False, "score": 0.1}

        # Process a question
        response = processor.generate_response("This is a test question", session_id="test")

        # Verify save_interaction was called
        save_mock.assert_called()
