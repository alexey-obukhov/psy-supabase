"""Debug RAGProcessor's save_interaction call."""

import traceback
from unittest.mock import MagicMock

from tests.helpers.database_test_base import DatabaseTestBase, get_optimal_device_for_testing


class TestRAGProcessorSaveInteraction(DatabaseTestBase):
    """Test class for debugging the RAGProcessor's save_interaction call."""

    def setUp(self):
        """Set up test environment with specialized mocks for RAG debugging."""
        # First call parent setup to establish database connection/mocking
        super().setUp()

        # Import here to avoid circular imports
        from psy_supabase.core.rag_processor import RAGProcessor

        # Use optimal device for testing
        self.test_device = get_optimal_device_for_testing()
        self.logger.info(f"Using device for tests: {self.test_device}")

        # Create simple mocks
        self.mock_generator = MagicMock()
        self.mock_generator.generate_text.return_value = "Test response"
        self.mock_generator.device = self.test_device

        # Create a processor with our test setup
        self.processor = RAGProcessor(
            db_manager=self.db_manager, generator=self.mock_generator  # Use the one from DatabaseTestBase
        )

        # Simplify the execution path as much as possible
        self.processor.check_toxicity = lambda text: {"is_toxic": False, "score": 0.1}

        # Create a mock selector that behaves predictably
        mock_selector = MagicMock()
        mock_selector.generate_prompt.return_value = "Test prompt"
        mock_selector.determine_topic.return_value = "test_topic"
        mock_selector.generate_category_info.return_value = {"Test": 1.0}
        mock_selector.analyze_question.return_value = {"topic": "test", "confidence": 1.0}
        self.processor.prompt_selector = mock_selector

        # Set up DB mock with monitoring if we're using a mock DB
        if not self.has_db_access:
            self.save_mock = MagicMock()
            self.db_manager.save_interaction = self.save_mock

            # Inject a wrapper around save_interaction to see if it's called
            original_save = self.db_manager.save_interaction

            def debug_save(*args, **kwargs):
                self.logger.info("--- SAVE_INTERACTION CALLED ---")
                self.logger.info(f"Args: {args}")
                self.logger.info(f"Kwargs: {kwargs}")
                result = original_save(*args, **kwargs)
                self.logger.info("save_interaction completed")
                return result

            self.db_manager.save_interaction = debug_save
        else:
            # For real DB, just monitor the call
            self.save_mock = None

        # Add a debug wrapper around generate_response to catch exceptions
        original_generate = self.processor.generate_response

        def debug_generate(*args, **kwargs):
            self.logger.info("--- GENERATE RESPONSE CALLED ---")
            self.logger.info(f"Args: {args}")
            self.logger.info(f"Kwargs: {kwargs}")
            try:
                result = original_generate(*args, **kwargs)
                self.logger.info(f"Response generated successfully: {result[:50]}..." if result else "None")
                if self.save_mock:
                    self.logger.info(f"save_interaction called: {self.save_mock.called}")
                return result
            except Exception as e:
                self.logger.error(f"ERROR in generate_response: {e}")
                self.logger.error(traceback.format_exc())
                raise

        self.processor.generate_response = debug_generate

    def test_rag_processor_save_interaction_call(self):
        """Test whether save_interaction is called when generating a response."""
        self.logger.info("--- STARTING TEST ---")

        # Actually generate a response using our test session ID from parent class
        response = self.processor.generate_response("Test question", session_id=self.test_session_id)

        self.logger.info("--- TEST COMPLETE ---")
        self.logger.info(f"Response: {response[:50]}...")

        if self.save_mock:
            self.logger.info(f"save_interaction called: {self.save_mock.called}")
            # Assert that save_interaction was called
            self.assertTrue(self.save_mock.called, "save_interaction was not called")
        else:
            self.logger.info("Using real DB, checking if response was generated")
            # With real DB we just verify we got a response
            self.assertIsNotNone(response)
            self.assertTrue(len(response) > 0)

        # Debug information
        self.logger.info("--- DEBUGGING INFO ---")
        self.logger.info(f"RAGProcessor DB manager: {self.processor.db_manager}")
        self.logger.info(f"Test DB manager: {self.db_manager}")
        self.logger.info(f"Are they the same? {self.processor.db_manager is self.db_manager}")

    def tearDown(self):
        """Clean up resources after test."""
        # Add any special cleanup for this test
        if hasattr(self, "processor") and self.processor:
            # Clean up processor resources if needed
            pass

        # Call parent tearDown (which will clean up DB and GPU)
        super().tearDown()


if __name__ == "__main__":
    # This allows running just this test file directly
    import unittest

    unittest.main()
