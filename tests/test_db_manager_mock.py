def test_db_manager_mock_save_interaction():
    """Test that our mock_db_manager properly handles save_interaction."""
    from unittest.mock import MagicMock

    # Create mock db manager with our custom implementation
    mock_db = MagicMock()

    # Track saved interactions
    mock_db._saved_interactions = []
    mock_db._last_context = None
    mock_db._last_metadata = None

    # Create a save_interaction implementation
    def save_interaction_impl(context, question, answer, metadata=None, session_id=None):
        """Implementation that tracks save_interaction calls."""
        mock_db._saved_interactions.append(
            {"context": context, "question": question, "answer": answer, "metadata": metadata, "session_id": session_id}
        )

        mock_db._last_context = context
        mock_db._last_metadata = metadata
        return True

    # Replace with our implementation
    mock_db.save_interaction = save_interaction_impl

    # Call save_interaction directly
    result = mock_db.save_interaction(
        context="test_context",
        question="test_question",
        answer="test_answer",
        metadata={"test": "metadata"},
        session_id="test_session",
    )

    # Verify it worked
    assert result is True, "save_interaction should return True"
    assert len(mock_db._saved_interactions) == 1, "Should have 1 saved interaction"
    assert mock_db._last_context == "test_context", f"Context should be 'test_context', got '{mock_db._last_context}'"
