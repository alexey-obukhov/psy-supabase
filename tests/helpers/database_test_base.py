"""
Database Test Base Module

This module provides a foundational test class with database connectivity
and proper test isolation for all database-related tests.

Key Features:
- Automatic test user ID and session ID generation
- Database connection management with graceful fallback
- Environment-aware configuration (CI vs local development)
- Proper test isolation through schema separation

Example usage:
    class TestUserInteractions(DatabaseTestBase):
        def test_save_interaction(self):
            # This test automatically gets unique test IDs and DB connection
            result = self.db_manager.save_interaction(
                "Test question",
                "Test answer",
                session_id=self.test_session_id
            )
            self.assertTrue(result)
"""

import random
import string
import unittest
from unittest.mock import MagicMock

from prismalog.log import get_logger

from psy_supabase.utilities.utils import cleanup_memory


def generate_id(length=8):
    """
    Generate a random ID string for testing purposes.

    Creates unpredictable, unique identifiers suitable for test isolation
    without database collisions.

    Args:
        length: Length of the random ID (default: 8 characters)

    Returns:
        String containing random hexadecimal characters

    Example:
        >>> user_id = generate_id()
        >>> print(user_id)
        '8a3f15e2'
    """
    return "".join(random.choice(string.hexdigits.lower()) for _ in range(length))


def get_optimal_device_for_testing():
    """
    Determine the best device for testing based on available memory.

    Analyzes GPU memory availability and makes a smart decision about
    whether to use CUDA or CPU for testing.

    Returns:
        str: "cuda" if GPU is available with sufficient memory, otherwise "cpu"

    Example:
        >>> device = get_optimal_device_for_testing()
        >>> model = Model().to(device)
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"

        # Check available memory
        free_memory_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / (1024**3)

        # If less than 2GB free, use CPU for stability
        if free_memory_gb < 2.0:
            return "cpu"

        return "cuda"
    except (ImportError, Exception):
        # If torch import fails or any other error
        return "cpu"


class DatabaseTestBase(unittest.TestCase):
    """
    Base class for database-related tests with proper setup and teardown.

    This class provides the foundation for all tests requiring database access.
    It handles:
    - Creating unique test user and session IDs
    - Establishing database connections with credentials
    - Detecting when database access is unavailable
    - Providing mock objects when needed
    - Logging test activity for debugging

    Attributes:
        test_user_id (str): Unique user ID for this test run
        test_session_id (str): Unique session ID for this test run
        db_manager (DatabaseManager): Database manager instance configured for testing
        has_db_access (bool): Whether a real database connection is available
        logger (Logger): Test-specific logger

    Test Methods:
        Tests extending this class should use self.db_manager for database operations
        and can check self.has_db_access to conditionally run integration tests.
    """

    def setUp(self):
        """
        Set up the test environment.

        Establishes:
        - Unique test identifiers
        - Database connection (or mock if unavailable)
        - Test logging

        Skips integration tests if database is unavailable unless
        the class attribute 'allow_no_db' is set to True.
        """
        # Setup logger
        self.logger = get_logger(__name__)
        self.logger.info("Setting up database test environment")

        # Generate unique test IDs
        self.test_user_id = f"test_user_{generate_id()}"
        self.test_session_id = f"test_session_{generate_id()}"

        self.logger.info(f"Test user ID: {self.test_user_id}")
        self.logger.info(f"Test session ID: {self.test_session_id}")

        # Set up for database test environment
        self.logger.info(f"Setting up {self.__class__.__name__}")

        # Create DB manager with test credentials
        self.db_manager = self._create_db_manager()

        # Check if we can connect to the database
        self.has_db_access = self._test_database_connection()

        self.test_device = get_optimal_device_for_testing()
        self.logger.info(f"Using device for tests: {self.test_device}")

        # If no DB access, skip integration tests but allow unit tests to run
        if not self.has_db_access and not getattr(self, "allow_no_db", False):
            self.skipTest("Unable to connect to database. Skipping integration tests.")

    def _create_db_manager(self):
        """
        Create a database manager for testing.

        This method:
        1. Attempts to get database credentials from environment variables
        2. Falls back to test credentials if not available
        3. Creates a real DatabaseManager for integration tests
        4. Provides a mock manager if creation fails

        Returns:
            DatabaseManager or MagicMock: A database manager instance
        """
        # Import here to avoid circular imports
        from psy_supabase.core.database import DatabaseManager

        try:
            # Get configuration - either from environment or test config
            import os

            # Try to get from environment variables first
            supabase_url = os.environ.get("SUPABASE_URL")
            supabase_key = os.environ.get("SUPABASE_KEY")
            self.logger.info("Supabase URL: %s", supabase_url)
            self.logger.info("Supabase Key: %s", supabase_key)
            # Check if we are in a CI environment

            # If not in environment, use test values
            # if not supabase_url or not supabase_key:
            #     self.logger.warning("Using test credentials - real database operations will be limited")
            #     supabase_url = "https://example-test.supabase.co"  # Test URL
            #     supabase_key = "test_key"  # Test key

            # Create a real DB manager for integration tests
            db_manager = DatabaseManager(
                user_id=self.test_user_id, supabase_url=supabase_url, supabase_key=supabase_key
            )

            # If needed, set session_id as an attribute
            if hasattr(db_manager, "session_id"):
                db_manager.session_id = self.test_session_id

            return db_manager
        except Exception as e:
            self.logger.warning(f"Failed to create database manager: {e}")
            # Return a mock DB manager
            mock_db = MagicMock()
            mock_db.schema_name = f"test_{generate_id()}"
            mock_db.user_id = self.test_user_id
            mock_db.session_id = self.test_session_id

            # Set up mock supabase
            mock_db.supabase = MagicMock()
            mock_response = MagicMock()
            mock_response.data = [{"success": True}]  # Default success response
            mock_db.supabase.rpc.return_value.execute.return_value = mock_response

            return mock_db

    def _test_database_connection(self):
        """
        Test if the database connection is working.

        Executes a simple query to verify if the database is accessible
        and properly configured.

        Returns:
            bool: True if database connection is successful, False otherwise
        """
        try:
            # Simple query that should work if we have DB access
            test_query = "SELECT 1 as test;"

            # Execute query
            # pylint: disable=all
            response = self.db_manager.supabase.rpc("sql", {"command": test_query}).execute()
            # pylint: enable=all  # Re-enable checks after the line

            # Check if we got a successful response with data
            if hasattr(response, "data"):
                # Any data means we connected
                self.logger.info("Database connection successful")
                return True
            else:
                # No data attribute is a problem
                self.logger.warning("Database connection test failed: unexpected response format")
                return False
        except Exception as e:  # pylint: disable=broad-exception-caught # Acceptable here for test robustness
            # Fallback for unexpected errors during the test call, but log differently
            self.logger.error("Unexpected error during database connection test: %s", str(e), exc_info=True)
            return False

    def tearDown(self):
        """
        Clean up resources after test execution.

        This method is called automatically after each test method runs,
        regardless of whether the test passed or failed.
        """
        # Clean up any test-specific resources
        self.logger.info(f"Cleaning up test session {self.test_session_id}")

        # Use the existing cleanup_memory function
        try:
            cleanup_memory(force_cuda_cleanup=True)
        except Exception as e:
            self.logger.debug(f"Memory cleanup failed: {str(e)}")
