"""
Test Configuration and Fixtures for the Psy Supabase Package.

This module contains shared pytest fixtures and test configuration that are automatically
discovered and used by pytest. Configuration includes:

1. Test constants (e.g., user IDs, schema names, URLs)
2. Mock objects for database testing without actual database connections
3. Sample data fixtures for consistent test scenarios
4. Logging configuration to suppress noisy logs during test runs

All test modules will automatically have access to these fixtures without explicit imports.
"""
import pytest
import logging
import json
from unittest.mock import Mock, patch
from psy_supabase.core.database import DatabaseManager

# Test data constants - shared across test modules
TEST_USER_ID = "test_user_123"
TEST_SCHEMA = "test_user_123"
TEST_SESSION_ID = "test_session_123"
TEST_URL = "https://fake-supabase-url.com"
TEST_KEY = "fake-api-key"

@pytest.fixture(autouse=True)
def suppress_logging():
    """
    Completely suppress all logs during test runs.

    This fixture:
    1. Captures the original logging configuration
    2. Replaces all handlers with a NullHandler
    3. Sets log levels to CRITICAL to minimize output
    4. Restores original configuration after test completes

    The fixture applies automatically to all tests without explicit usage.
    """
    # Save the original logging configuration
    root_logger = logging.getLogger()
    original_level = root_logger.level
    original_handlers = root_logger.handlers.copy()

    # Remove all handlers and set critical level
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)

    # Create a null handler that won't output anything
    null_handler = logging.NullHandler()
    root_logger.addHandler(null_handler)
    root_logger.setLevel(logging.CRITICAL)

    # Also suppress specific module loggers
    module_loggers = [
        logging.getLogger("psy_supabase"),
        logging.getLogger("psy_supabase.core.database")
    ]
    original_module_levels = {logger: logger.level for logger in module_loggers}

    for logger in module_loggers:
        logger.setLevel(logging.CRITICAL)

    # Yield control to the test
    yield

    # Restore original logging configuration after test
    root_logger.removeHandler(null_handler)
    root_logger.setLevel(original_level)

    for handler in original_handlers:
        root_logger.addHandler(handler)

    for logger, level in original_module_levels.items():
        logger.setLevel(level)

@pytest.fixture
def mock_supabase():
    """
    Create a mock Supabase client for testing.

    This fixture creates a comprehensive mock that simulates:
    1. RPC calls with .rpc().execute()
    2. Table operations with .table().insert().execute()
    3. Default successful responses for common operations

    The mock allows tests to verify if methods were called with correct
    parameters without making actual database connections.

    Returns:
        Mock: A configured mock object simulating the Supabase client
    """
    mock_client = Mock()

    # Mock responses for various methods
    mock_rpc_response = Mock()
    mock_rpc_response.data = True
    mock_rpc_response.error = None

    # Configure execute() to return the mock response
    mock_execute = Mock(return_value=mock_rpc_response)

    # Configure rpc() to return an object with execute method
    mock_rpc = Mock()
    mock_rpc.execute = mock_execute

    # Configure table() to return object with insert method
    mock_insert = Mock()
    mock_insert.execute = mock_execute
    mock_table = Mock(return_value=mock_insert)

    # Attach all these mocks to the main client
    mock_client.rpc = Mock(return_value=mock_rpc)
    mock_client.table = Mock(return_value=mock_table)

    return mock_client

@pytest.fixture
def db_manager(mock_supabase):
    """
    Create a DatabaseManager instance with a mock Supabase client.

    This fixture:
    1. Patches the create_client function to return a mock
    2. Creates a DatabaseManager with test credentials
    3. Returns the manager for use in tests

    This allows testing DatabaseManager methods without actual
    database connections while verifying correct behavior.

    Args:
        mock_supabase: The mock Supabase client fixture

    Returns:
        DatabaseManager: Configured with mock client
    """
    with patch('psy_supabase.core.database.create_client', return_value=mock_supabase):
        manager = DatabaseManager(TEST_URL, TEST_KEY, TEST_USER_ID)
        return manager

@pytest.fixture
def sample_interaction():
    """
    Return a sample interaction dictionary for testing.

    Provides a standardized interaction object with realistic fields
    including context, question, answer and metadata with psychological
    metrics.

    Returns:
        dict: A sample interaction dictionary
    """
    return {
        'context': 'Test context',
        'question': 'How are you feeling today?',
        'answer': 'I am feeling better, thanks for asking.',
        'metadata': {'topic': 'Wellness', 'effectiveness': {'term_overlap': 0.8}}
    }

@pytest.fixture
def sample_history():
    """
    Return sample conversation history data.

    Provides a standardized list of conversation interactions with
    realistic timestamps, questions, answers, and metadata in the format
    returned by the database.

    Returns:
        list: List of conversation history dictionaries
    """
    return [
        {
            'interactionid': 1,
            'question': 'How are you feeling today?',
            'answer': 'I am feeling better, thanks for asking.',
            'context': 'Test context',
            'metadata': json.dumps({'topic': 'Wellness', 'effectiveness': {'term_overlap': 0.8}}),
            'created_at': '2023-01-01T12:00:00'
        },
        {
            'interactionid': 2,
            'question': 'What has been bothering you lately?',
            'answer': 'I have been stressed about work.',
            'context': 'CBT session',
            'metadata': json.dumps({'topic': 'Anxiety', 'effectiveness': {'term_overlap': 0.7}}),
            'created_at': '2023-01-01T12:05:00'
        }
    ]

@pytest.fixture
def sample_similar_documents():
    """
    Return sample vector similarity search results.

    Provides standardized document similarity results as would be returned
    by vector embedding similarity searches, including content and
    similarity scores.

    Returns:
        list: List of document dictionaries with similarity scores
    """
    return [
        {
            'id': 1,
            'content': 'This is a sample document about anxiety management techniques.',
            'similarity': 0.95
        },
        {
            'id': 2,
            'content': 'Another document about stress reduction strategies.',
            'similarity': 0.85
        }
    ]
