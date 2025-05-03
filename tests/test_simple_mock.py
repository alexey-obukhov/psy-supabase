from unittest.mock import MagicMock

import pytest


def test_mock_called():
    """Verify that assert_called works properly on a MagicMock."""
    # Create a mock
    mock_func = MagicMock()

    # Call the mock
    mock_func(1, 2, 3, name="test")

    # These assertions will pass
    mock_func.assert_called()  # Check it was called
    assert mock_func.call_args is not None  # Check call_args exists

    # Extract arguments
    args, kwargs = mock_func.call_args
    assert args == (1, 2, 3)
    assert kwargs == {"name": "test"}
