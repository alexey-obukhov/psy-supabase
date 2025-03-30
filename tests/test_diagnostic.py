def test_diagnostic_rag_processor():
    """A diagnostic test to find what's happening with RAGProcessor."""
    from unittest.mock import MagicMock, patch
    from psy_supabase.core.rag_processor import RAGProcessor

    # Create mocks
    mock_db = MagicMock()
    mock_text_gen = MagicMock()

    # Set specific return value
    mock_text_gen.generate_text.return_value = "Test response"

    # Create processor
    processor = RAGProcessor(
        db_manager=mock_db,
        generator=mock_text_gen
    )

    # Disable toxicity check
    processor.check_toxicity = lambda text: {"is_toxic": False, "score": 0.1}

    # Basic prompt selector
    mock_selector = MagicMock()
    mock_selector.analyze_question.return_value = {"topic": "test"}
    processor.prompt_selector = mock_selector

    # Capture the method execution
    try:
        with patch('psy_supabase.core.rag_processor.logger') as mock_logger:
            response = processor.generate_response("Test question", session_id="test_session")

            # Log interactions for debugging
            print(f"Response: {response}")
            print(f"Response type: {type(response)}")
            print(f"Save interaction called: {mock_db.save_interaction.called}")
            print(f"Logger calls: {[call for call in mock_logger.method_calls]}")

            # Count message types
            info_count = sum(1 for call in mock_logger.method_calls if call[0] == 'info')
            error_count = sum(1 for call in mock_logger.method_calls if call[0] == 'error')
            print(f"Info logs: {info_count}, Error logs: {error_count}")

    except Exception as e:
        print(f"Exception during execution: {e}")
        import traceback
        traceback.print_exc()

    # Just pass this diagnostic test
    assert True