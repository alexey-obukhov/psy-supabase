"""Debug RAGProcessor's save_interaction call."""

import pytest
import traceback
from unittest.mock import MagicMock, patch

def test_rag_processor_save_interaction_call():
    """Debug why save_interaction isn't being called."""
    from psy_supabase.core.rag_processor import RAGProcessor
    import logging

    # Set up root logger to see all logs
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger()

    # Create simple mocks
    mock_db = MagicMock()
    mock_generator = MagicMock()
    mock_generator.generate_text.return_value = "Test response"

    print("\n--- TEST SETUP ---")
    print(f"Mock DB: {mock_db}")

    # Create a processor with verbose mode
    processor = RAGProcessor(
        db_manager=mock_db,
        generator=mock_generator
    )

    # Simplify the execution path as much as possible
    processor.check_toxicity = lambda text: {"is_toxic": False, "score": 0.1}

    # Create a mock selector that behaves predictably
    mock_selector = MagicMock()
    mock_selector.generate_prompt.return_value = "Test prompt"
    mock_selector._determine_topic.return_value = "test_topic"
    mock_selector.generate_category_info.return_value = {"Test": 1.0}
    mock_selector.analyze_question.return_value = {"topic": "test", "confidence": 1.0}
    processor.prompt_selector = mock_selector

    # Set up DB mock with monitoring
    save_mock = MagicMock()
    mock_db.save_interaction = save_mock

    # Add a debug wrapper around generate_response to catch exceptions
    original_generate = processor.generate_response
    def debug_generate(*args, **kwargs):
        print("\n--- GENERATE RESPONSE CALLED ---")
        print(f"Args: {args}")
        print(f"Kwargs: {kwargs}")
        try:
            result = original_generate(*args, **kwargs)
            print(f"Response generated successfully: {result[:50]}..." if result else "None")
            print(f"save_interaction called: {save_mock.called}")
            return result
        except Exception as e:
            print(f"ERROR in generate_response: {e}")
            traceback.print_exc()
            raise

    processor.generate_response = debug_generate

    # Inject a wrapper around save_interaction to see if it's called
    original_save = mock_db.save_interaction
    def debug_save(*args, **kwargs):
        print("\n--- SAVE_INTERACTION CALLED ---")
        print(f"Args: {args}")
        print(f"Kwargs: {kwargs}")
        result = original_save(*args, **kwargs)
        print("save_interaction completed")
        return result

    mock_db.save_interaction = debug_save

    # Actually generate a response
    print("\n--- STARTING TEST ---")
    response = processor.generate_response("Test question", session_id="test_session")

    print("\n--- TEST COMPLETE ---")
    print(f"Response: {response[:50]}...")
    print(f"save_interaction called: {save_mock.called}")

    # Since this is likely to fail, let's check key conditions
    # to see what might be happening
    print("\n--- DEBUGGING INFO ---")
    print(f"RAGProcessor DB manager: {processor.db_manager}")
    print(f"Mock DB: {mock_db}")
    print(f"Are they the same? {processor.db_manager is mock_db}")