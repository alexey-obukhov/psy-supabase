"""
Fixtures for integration tests.
These fixtures create real components with minimal mocking to test actual integration.
"""
import os
import pytest
import uuid
from unittest.mock import MagicMock, patch

# Import from main test fixtures
from tests.conftest import (
    mock_supabase,
    mock_db_manager,
    mock_db_manager_with_spy,
    TEST_URL,
    TEST_KEY
)

from psy_supabase.core.database import DatabaseManager
from psy_supabase.core.rag_processor import RAGProcessor
from psy_supabase.core.response_generator import ResponseGenerator
from psy_supabase.core.text_generator import TextGenerator
from psy_supabase.utilities.prompt_selector import PromptSelector

@pytest.fixture
def mock_text_generator():
    """Create a mock text generator that returns predetermined responses."""
    generator = MagicMock(spec=TextGenerator)
    generator.generate_text.return_value = "This is a test response"
    generator.generate_therapeutic_response.return_value = "This is a therapeutic response"
    generator.is_toxic.return_value = False
    return generator

@pytest.fixture
def mock_embedding_provider():
    """Create a mock embedding provider for tests."""
    mock = MagicMock()
    mock.generate_embedding.return_value = [0.1] * 768
    mock.get_embedding_dimension.return_value = 768
    return mock

@pytest.fixture
def real_prompt_selector(mock_text_generator):
    """Create a real prompt selector with the required generator argument."""
    return PromptSelector(generator=mock_text_generator)

@pytest.fixture
def integration_db_manager(mock_db_manager):
    """Create a mocked database manager that properly returns conversation history."""
    # Enhance the mock_db_manager with proper history tracking
    db_manager = mock_db_manager

    # Dictionary to store fake conversation history for tests
    conversation_store = {}

    # Override save_interaction to store in our test dictionary
    original_save = db_manager.save_interaction

    def save_interaction_with_history(session_id, question, answer, metadata=None, context=None):
        # Call original method first
        result = original_save(session_id, question, answer, metadata, context)

        # Then store in our test dictionary
        if session_id not in conversation_store:
            conversation_store[session_id] = []

        conversation_store[session_id].append({
            'question': question,
            'answer': answer,
            'metadata': metadata,
            'context': context,
            'timestamp': str(uuid.uuid4())  # Fake timestamp for ordering
        })

        return result

    # Override get_conversation_history to return from our dictionary
    original_get = db_manager.get_conversation_history

    def get_conversation_history_with_data(session_id):
        # If we have data in our store, return it
        if session_id in conversation_store:
            return conversation_store[session_id]
        # Otherwise call original method
        return original_get(session_id)

    # Apply our overrides
    db_manager.save_interaction = save_interaction_with_history
    db_manager.get_conversation_history = get_conversation_history_with_data

    return db_manager

@pytest.fixture
def integration_response_generator(mock_text_generator, integration_db_manager, real_prompt_selector):
    """Create a response generator for integration tests with real prompt selector."""
    return ResponseGenerator(
        text_generator=mock_text_generator,
        db_manager=integration_db_manager,
        prompt_selector=real_prompt_selector
    )

@pytest.fixture
def integration_rag_processor(mock_text_generator, integration_db_manager, mock_embedding_provider):
    """Create a RAG processor for integration tests.

    This uses real components where possible, with mocks only for external dependencies.
    """
    with patch('psy_supabase.core.model_manager.EmbeddingProviderAdapter', return_value=mock_embedding_provider):
        processor = RAGProcessor(
            db_manager=integration_db_manager,
            generator=mock_text_generator,
            intelligent_processing_enabled=True
        )

        # Make sure embedding provider is mocked
        processor.embedding_provider = mock_embedding_provider

        return processor

@pytest.fixture
def mock_dynamic_retriever():
    """Create a mock dynamic retriever."""
    mock = MagicMock()
    mock.retrieve.return_value = []
    mock.get_conversation_context.return_value = {
        "relevant_past_messages": [],
        "relevant_topics": ["test_topic"]
    }
    return mock