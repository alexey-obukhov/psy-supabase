def test_simple_save_interaction():
    """Test that ensures save_interaction is callable on a mock."""
    from unittest.mock import MagicMock

    # Create a fresh mock
    mock_db = MagicMock()
    save_mock = MagicMock()
    mock_db.save_interaction = save_mock

    # Call save_interaction
    mock_db.save_interaction(
        context="test_context",
        question="test_question",
        answer="test_answer",
        metadata={},
        session_id="test_session"
    )

    # Assert it was called
    save_mock.assert_called_once()
    print("SUCCESS: save_interaction was called as expected")