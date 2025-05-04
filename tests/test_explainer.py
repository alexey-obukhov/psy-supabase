import json
import unittest
import uuid
from unittest.mock import MagicMock, patch

import pytest

from psy_supabase.core.database import DatabaseManager
from psy_supabase.core.rag_processor import RAGProcessor
from psy_supabase.core.response_generator import ResponseGenerator
from psy_supabase.core.text_generator import TextGenerator
from psy_supabase.utilities.prompt_selector import PromptSelector


class TestExplainer(unittest.TestCase):
    """Demonstrates the core logic flow of your program using mocks."""

    def setUp(self):
        """Set up a mock environment before each test."""
        self.db_manager = MagicMock(spec=DatabaseManager)
        self.db_manager.schema_name = "public"  # or any schema name you need

        self.text_generator = MagicMock(spec=TextGenerator)
        self.response_generator = MagicMock(spec=ResponseGenerator)
        self.prompt_selector = MagicMock(spec=PromptSelector)

        # Create a RAGProcessor with these mocks
        self.rag_processor = RAGProcessor(
            db_manager=self.db_manager, generator=self.text_generator, intelligent_processing_enabled=True
        )
        # RAGProcessor uses a prompt_selector inside, so substitute our mock
        self.rag_processor.prompt_selector = self.prompt_selector

    @patch("psy_supabase.core.model_manager.EmbeddingProviderAdapter")
    def test_full_pipeline_example(self, mock_embedding_adapter):
        # 1) Mock prompt_selector to identify a topic/emotion
        self.prompt_selector.analyze_question.return_value = {
            "topic": "anxiety",
            "emotion": "anxiety",
            "confidence": 0.95,
            "emotion_confidence": 0.9,
        }

        # 2) Mock the embedding provider
        mock_instance = MagicMock()
        mock_instance.generate_embedding.return_value = [0.1] * 768
        mock_embedding_adapter.return_value = mock_instance

        # 3) Mock text generation
        self.text_generator.is_toxic.return_value = False
        # Mock the method your code really calls
        self.text_generator.generate_therapeutic_response_with_dynamic_retrieval.return_value = "Therapeutic response"

        session_id = f"test_{uuid.uuid4().hex[:8]}"

        # 4) Patch process_query so it returns a non-empty embedding
        with patch.object(self.rag_processor, "process_query", return_value=[0.1] * 768):
            response = self.rag_processor.generate_response(
                user_question="How do I deal with my anxiety?", session_id=session_id
            )

        # 5) Check that the final response is as we mocked
        self.assertEqual(response, "Therapeutic response")
        self.prompt_selector.analyze_question.assert_called_once()
        mock_instance.generate_embedding.assert_called()
        self.text_generator.generate_therapeutic_response_with_dynamic_retrieval.assert_called_once()
        print(f"Final response: {response}")
        print("✓ Pipeline demonstration test passed!")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
