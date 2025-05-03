from unittest.mock import MagicMock


def test_direct_mock():
    """Test that a direct mock works correctly."""
    # Create a mock
    mock_db = MagicMock()
    mock_db.save_interaction = MagicMock()

    # Call the mock
    mock_db.save_interaction("test", "question", "answer")

    # Verify it was called
    mock_db.save_interaction.assert_called()

    # Get the arguments
    args = mock_db.save_interaction.call_args[0]
    assert args[0] == "test"
