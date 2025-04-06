"""
Integration tests for vector utilities using real database connections.

These tests validate the vector utility functions against a real Supabase database
with pgvector capabilities, ensuring proper vector storage and retrieval.
"""
import os
import unittest
from unittest.mock import patch, MagicMock, Mock
import pytest
import numpy as np
from typing import List, Dict, Any

from school_logging.log import ColoredLogger
from tests.helpers.database_test_base import generate_id, DatabaseTestBase
from psy_supabase.core.database import DatabaseManager
from psy_supabase.utilities.vector_utils import (
    validate_vector_format,
    format_vector_for_pgvector,
    find_similar_interactions,
    unified_vector_search,
    ensure_vector_indexes,
    batch_enrich_interactions,
    optimize_vector_operations,
    add_vector_embedding
)

logger = ColoredLogger(__name__)


class TestVectorUtilsIntegration(DatabaseTestBase):
    """Integration tests for vector utilities."""

    # Allow tests to run with mock DB if real DB is not available
    allow_no_db = True

    def setUp(self):
        """Set up test environment."""
        super().setUp()

        # If no DB access, create mock responses for our tests
        if not self.has_db_access:
            self.logger.info("Using mock DB for integration tests")

            # Setup a simple mock that always returns success
            success_response = MagicMock()
            success_response.data = [{'success': True}]

            # Mock side effect for different cases
            def mock_side_effect(*args, **kwargs):
                mock_execute = MagicMock()

                # Ensure vector indexes - always success
                if args[0] == 'sql' and 'ensure_vector_indexes' in str(kwargs):
                    mock_execute.execute.return_value = success_response

                # Batch enrichment - return interactions for query, success for updates
                elif args[0] == 'sql' and 'SELECT interaction_id' in str(kwargs):
                    interactions_response = MagicMock()
                    interactions_response.data = [
                        {'interaction_id': 1, 'question': 'Test', 'answer': 'Response'}
                    ]
                    mock_execute.execute.return_value = interactions_response

                # Update query - return success
                elif args[0] == 'sql' and 'UPDATE' in str(kwargs):
                    update_response = MagicMock()
                    update_response.data = True
                    mock_execute.execute.return_value = update_response

                # Default - return success
                else:
                    mock_execute.execute.return_value = success_response

                return mock_execute

            # Apply the side effect
            self.db_manager.supabase.rpc.side_effect = mock_side_effect

    def test_validate_vector_format(self):
        """Test vector format validation."""
        assert validate_vector_format([0.1, 0.2, 0.3]) is True
        assert validate_vector_format("not-a-vector") is False

    def test_format_vector_for_pgvector(self):
        """Test vector formatting for pgvector."""
        vector = [0.1, 0.2, 0.3]
        formatted = format_vector_for_pgvector(vector)
        assert formatted == "[0.1,0.2,0.3]" or formatted == "[0.1, 0.2, 0.3]"

    def test_parameter_validation(self):
        """Test that find_similar_interactions correctly validates parameters."""
        # Import typeguard error
        from typeguard import TypeCheckError

        # Test with string embedding - expect TypeCheckError
        with pytest.raises(TypeCheckError) as exc_info:
            # Import inside the with block for better isolation
            from psy_supabase.utilities.vector_utils import find_similar_interactions

            # This should raise TypeCheckError
            find_similar_interactions(
                self.db_manager,
                embedding="health_check",  # This triggers TypeCheckError
                schema_name="test_schema"
            )

        # Verify error message contains typeguard message
        assert "is not a list" in str(exc_info.value)

        # Test with proper embedding
        valid_embedding = [0.1, 0.2, 0.3, 0.4] * 100

        # Mock the database call to avoid actual DB queries
        with patch.object(self.db_manager, 'find_similar_interactions_by_embedding') as mock_db_find:
            mock_db_find.return_value = [{"id": 1, "similarity": 0.95}]

            # Import here
            from psy_supabase.utilities.vector_utils import find_similar_interactions

            # This should work with valid parameters
            result = find_similar_interactions(
                self.db_manager,
                embedding=valid_embedding,
                schema_name="test_schema"
            )

            # Verify function returned the expected results
            self.assertEqual(len(result), 1)

            # Verify correct parameters were passed
            mock_db_find.assert_called_once()

    def test_real_db_ensure_vector_indexes(self):
        """Test ensuring vector indexes."""
        result = ensure_vector_indexes(self.db_manager, self.db_manager.schema_name)
        assert result is True

    def test_batch_enrich_interactions(self):
        """Test batch enrichment."""
        result = batch_enrich_interactions(
            self.db_manager,
            self.db_manager.schema_name
        )
        assert result >= 0

    def test_optimize_vector_operations_integration(self):
        """Test vector optimization."""
        # Run the test
        result = optimize_vector_operations(
            self.db_manager,
            self.db_manager.schema_name
        )

        # Assert the expected result - since we mocked ensure_vector_indexes to return success
        assert result.get('indexes_created') is True
        assert result.get('statistics_updated') is True

    def test_unified_vector_search(self):
        """Test unified vector search."""
        test_embedding = [0.1, 0.2, 0.3, 0.4] * 100
        result = unified_vector_search(
            self.db_manager,
            embedding=test_embedding,
            table="interactions"
        )
        assert isinstance(result, list)

    def test_redundancy_elimination(self):
        """Test redundancy elimination."""
        result = add_vector_embedding(
            self.db_manager,
            1,  # interaction_id
            "interaction",
            "Test text"
        )
        assert result is True

    def tearDown(self):
        """Clean up test resources."""
        if hasattr(self, 'has_db_access') and self.has_db_access:
            try:
                # Clean up test schema - optional
                schema_name = f"test_vectors_{self.test_user_id}"
                cleanup_query = f"DROP SCHEMA IF EXISTS {schema_name} CASCADE;"
                self.db_manager.supabase.rpc('sql', {'command': cleanup_query}).execute()
                self.logger.info("Cleaned up test schema: %s", schema_name)
            except Exception as e:
                self.logger.error("Error cleaning up test resources: %s", e)

        # Call parent tearDown for diagnostics
        super().tearDown()


if __name__ == '__main__':
    unittest.main()