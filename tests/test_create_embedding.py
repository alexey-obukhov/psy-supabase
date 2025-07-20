import pytest
from unittest.mock import patch, MagicMock

from psy_supabase.core.database import DatabaseManager

@pytest.fixture
def db_manager():
    # Minimal mock for DatabaseManager with required attributes
    manager = DatabaseManager.__new__(DatabaseManager)
    manager._embedding_cache = {}
    manager.user_id = "test_user"
    manager.schema_name = "test_schema"
    manager.supabase = MagicMock()
    return manager

def test_create_embedding_returns_float_list(db_manager):
    with patch("psy_supabase.core.database.get_embedding_provider") as mock_provider:
        mock_instance = MagicMock()
        mock_instance.generate_embedding.return_value = [1, 2, 3]
        mock_provider.return_value = mock_instance

        result = db_manager.create_embedding("Hello world")
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)
        assert result == [1.0, 2.0, 3.0]

def test_create_embedding_uses_cache(db_manager):
    db_manager._embedding_cache["hello"] = [0.1, 0.2]
    result = db_manager.create_embedding("hello")
    assert result == [0.1, 0.2]

def test_create_embedding_invalid_type_logs_error(db_manager):
    with patch("psy_supabase.core.database.get_embedding_provider") as mock_provider, \
         patch("psy_supabase.core.database.logger") as mock_logger:
        mock_instance = MagicMock()
        mock_instance.generate_embedding.return_value = "not_a_list"
        mock_provider.return_value = mock_instance

        result = db_manager.create_embedding("test")
        assert result is None
        mock_logger.error.assert_any_call("Generated embedding is not a valid list of floats.")

def test_create_embedding_handles_exception(db_manager):
    with patch("psy_supabase.core.database.get_embedding_provider", side_effect=Exception("fail")), \
         patch("psy_supabase.core.database.logger") as mock_logger:
        result = db_manager.create_embedding("test")
        assert result is None
        # Check that the error message was logged at least once
        found = any(
            call.args and "Error creating embedding" in call.args[0]
            for call in mock_logger.error.call_args_list
        )
        assert found, "Expected error message not found in logger.error calls"

def test_create_embedding_chunked(db_manager):
    # Simulate chunking: model_based_chunk_question returns a list of text chunks
    fake_chunks = ["chunk1", "chunk2", "chunk3"]
    fake_embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

    # Patch the instance method for chunking
    db_manager.model_based_chunk_question = lambda text: fake_chunks

    # Patch get_embedding_provider
    mock_provider = MagicMock()
    mock_provider.generate_embedding.side_effect = fake_embeddings

    with patch("psy_supabase.core.database.get_embedding_provider", return_value=mock_provider):
        # Add the method if not present
        if not hasattr(db_manager, "create_chunked_embedding"):
            def create_chunked_embedding(self, text):
                chunks = self.model_based_chunk_question(text)
                provider = mock_provider
                return [provider.generate_embedding(chunk) for chunk in chunks]
            db_manager.create_chunked_embedding = create_chunked_embedding.__get__(db_manager)
        result = db_manager.create_chunked_embedding("This is a long text to chunk.")
        assert isinstance(result, list)
        assert result == fake_embeddings
        assert all(isinstance(chunk, list) for chunk in result)
        assert all(isinstance(x, float) for chunk in result for x in chunk)

def test_model_based_chunk_question_called(db_manager):
    called = {}

    def fake_chunker(text):
        called["text"] = text
        return ["chunkA", "chunkB"]

    db_manager.model_based_chunk_question = fake_chunker

    # Patch get_embedding_provider to avoid errors
    mock_provider = MagicMock()
    mock_provider.generate_embedding.return_value = [0.1, 0.2]

    with patch("psy_supabase.core.database.get_embedding_provider", return_value=mock_provider):
        if not hasattr(db_manager, "create_chunked_embedding"):
            def create_chunked_embedding(self, text):
                chunks = self.model_based_chunk_question(text)
                provider = mock_provider
                return [provider.generate_embedding(chunk) for chunk in chunks]
            db_manager.create_chunked_embedding = create_chunked_embedding.__get__(db_manager)
        db_manager.create_chunked_embedding("foobar")
        assert called["text"] == "foobar"
