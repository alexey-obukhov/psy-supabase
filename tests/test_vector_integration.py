"""
Integration tests for vector utilities with database manager.

These tests verify that the vector utilities module works correctly
with the DatabaseManager class.
"""

from unittest.mock import MagicMock, patch

import pytest

from psy_supabase.core.database import DatabaseManager


@pytest.fixture
def db_manager():
    """Create a mocked database manager for testing."""
    with patch("psy_supabase.core.database.create_client") as mock_create_client:
        # Setup supabase mock
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client

        # Create database manager with test values
        manager = DatabaseManager(
            supabase_url="https://test.supabase.co", supabase_key="test_key", user_id="test_user_123"
        )

        # Set schema name explicitly for testing
        manager.schema_name = "test_user_123"

        yield manager


def test_database_manager_uses_vector_utils_module(db_manager):
    """Test that DatabaseManager properly uses the vector_utils module."""
    # The issue: we need to patch at the correct import location
    # The DatabaseManager imports it as 'optimize_vectors', not 'optimize_vector_operations'
    with patch("psy_supabase.core.database.optimize_vectors") as mock_optimize:
        # Set up mock return value
        mock_optimize.return_value = {
            "column_added": True,
            "indexes_created": True,
            "interactions_enriched": 3,
            "statistics_updated": True,
        }

        # Call the database manager method
        result = db_manager.optimize_vector_operations()

        # Verify the vector_utils function was called with correct parameters
        mock_optimize.assert_called_once_with(db_manager, db_manager.schema_name)

        # Verify the result was correctly passed through
        assert result["column_added"] is True
        assert result["indexes_created"] is True
        assert result["interactions_enriched"] == 3
        assert result["statistics_updated"] is True


def test_database_manager_vector_operations_error_handling(db_manager):
    """Test error handling when vector operations fail."""
    # Ensure the exception has a proper string representation
    with patch("psy_supabase.core.database.optimize_vectors", side_effect=Exception("Test error")) as mock_optimize:

        # Call the method
        result = db_manager.optimize_vector_operations()

        # Verify error handling in database manager
        assert result["column_added"] is False
        assert result["indexes_created"] is False
        assert result["interactions_enriched"] == 0
        assert result["statistics_updated"] is False
        assert "error" in result
        assert isinstance(result["error"], str)
        assert "Test error" in str(result["error"])


def test_vector_utilities_module_imported(db_manager):
    """Simple test to ensure vector_utils module is being imported correctly."""
    # This would fail if the import wasn't working
    from psy_supabase.utilities import vector_utils

    # Just check that some of the expected functions exist
    assert hasattr(vector_utils, "optimize_vector_operations")
    assert hasattr(vector_utils, "batch_enrich_interactions")
    assert hasattr(vector_utils, "ensure_vector_indexes")
    assert hasattr(vector_utils, "update_table_statistics")
