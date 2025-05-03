""" Tests for DynamicRAGRetriever functionality and template selection."""

from unittest.mock import create_autospec


def test_with_autospec_mock():
    """Test using autospec to create a more conformant mock."""
    # Import the real DatabaseManager
    from psy_supabase.core.database import DatabaseManager
    from psy_supabase.core.rag_processor import RAGProcessor
    from psy_supabase.core.text_generator import TextGenerator

    # Create autospec mocks
    mock_db = create_autospec(DatabaseManager, instance=True)
    mock_generator = create_autospec(TextGenerator, instance=True)

    # Set return values
    mock_db.save_interaction.return_value = True
    mock_generator.generate_text.return_value = "Test response"

    # Create RAGProcessor with these mocks
    processor = RAGProcessor(db_manager=mock_db, generator=mock_generator)

    # Override toxicity check
    processor.check_toxicity = lambda text: {"is_toxic": False, "score": 0.1}

    # Add a minimal prompt selector
    from unittest.mock import MagicMock

    processor.prompt_selector = MagicMock()
    processor.prompt_selector.analyze_question.return_value = {"topic": "test"}
    processor.prompt_selector.determine_topic.return_value = "test_topic"
    processor.prompt_selector.generate_category_info.return_value = {"Test": 1.0}

    # Call generate_response
    processor.generate_response("Test question", session_id="test")

    # Check if save_interaction was called
    assert mock_db.save_interaction.called, "save_interaction should be called"

    # Print debug info
    print(f"save_interaction called: {mock_db.save_interaction.called}")
    if mock_db.save_interaction.call_args:
        print(f"call args: {mock_db.save_interaction.call_args}")
