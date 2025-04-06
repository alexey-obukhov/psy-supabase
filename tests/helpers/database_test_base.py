import unittest
import random
import string
from unittest.mock import patch, MagicMock, Mock
from typing import Optional, Dict, Any

from school_logging.log import ColoredLogger

def generate_id(length=8):
    """
    Generate a random ID string for testing purposes.

    Args:
        length: Length of the random ID

    Returns:
        String containing random hexadecimal characters
    """
    return ''.join(random.choice(string.hexdigits.lower()) for _ in range(length))

class DatabaseTestBase(unittest.TestCase):
    """Base class for database-related tests with proper setup and teardown."""

    def setUp(self):
        """Set up the test environment."""
        # Setup logger
        self.logger = ColoredLogger(__name__)
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

        # If no DB access, skip integration tests but allow unit tests to run
        if not self.has_db_access and not getattr(self, 'allow_no_db', False):
            self.skipTest("Unable to connect to database. Skipping integration tests.")

    def _create_db_manager(self):
        """Create a database manager for testing."""
        # Import here to avoid circular imports
        from psy_supabase.core.database import DatabaseManager

        try:
            # Get configuration - either from environment or test config
            import os

            # Try to get from environment variables first
            supabase_url = os.environ.get('SUPABASE_URL')
            supabase_key = os.environ.get('SUPABASE_KEY')

            # If not in environment, use test values
            if not supabase_url or not supabase_key:
                self.logger.warning("Using test credentials - real database operations will be limited")
                supabase_url = "https://example-test.supabase.co"  # Test URL
                supabase_key = "test_key"  # Test key

            # Create a real DB manager for integration tests
            db_manager = DatabaseManager(
                user_id=self.test_user_id,
                supabase_url=supabase_url,
                supabase_key=supabase_key
            )

            # If needed, set session_id as an attribute
            if hasattr(db_manager, 'session_id'):
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
            mock_response.data = [{'success': True}]  # Default success response
            mock_db.supabase.rpc.return_value.execute.return_value = mock_response

            return mock_db

    def _test_database_connection(self):
        """Test database connection."""
        try:
            # Simple query that should work if we have DB access
            test_query = "SELECT 1 as test;"

            # Execute query
            response = self.db_manager.supabase.rpc('sql', {'command': test_query}).execute()

            # Check if we got a successful response with data
            if hasattr(response, 'data'):
                # Any data means we connected
                self.logger.info("Database connection successful")
                return True
            else:
                # No data attribute is a problem
                self.logger.warning("Database connection test failed: unexpected response format")
                return False
        except Exception as e:
            self.logger.warning(f"Unable to connect to database: {str(e)}")
            return False

    def tearDown(self):
        """Teardown resources after test."""
        pass