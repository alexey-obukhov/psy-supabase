import unittest
import torch
import uuid
from unittest.mock import patch, MagicMock
import json

from psy_supabase.core.text_generator import TextGenerator
from psy_supabase.core.model_manager import ModelManager, get_embedding_provider
from tests.helpers.database_test_base import DatabaseTestBase
from tests.helpers.model_mocks import MockTextGenerator


class TestEmbeddingGeneration(DatabaseTestBase):
    """
    Test cases for embedding generation functionality.
    """

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        # Create minimal mocks to test embedding functions
        self.text_generator = TextGenerator(model_name="microsoft/phi-1_5", device=None)
        self.text_generator.model = MagicMock()
        self.text_generator.tokenizer = MagicMock()

        # Ensure session_id is set for all tests
        self.session_id = f"test_session_{uuid.uuid4().hex[:8]}"

    def tearDown(self):
        """Clean up after each test."""
        super().tearDown()
        # Additional cleanup if needed
        del self.text_generator.model
        del self.text_generator.tokenizer
        del self.text_generator

    def test_tokenizer_output_handling(self):
        """Test that the get_embedding method properly handles BatchEncoding output."""
        # Mock the tokenizer to return a dict-like object (similar to BatchEncoding)
        mock_encoding = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        self.text_generator.tokenizer.return_value = mock_encoding

        # Setup the mock model to return an object with hidden_states
        mock_output = MagicMock()
        mock_output.hidden_states = [torch.rand(1, 3, 768) for _ in range(3)]  # 3 layers
        self.text_generator.model.return_value = mock_output

        # Test the function
        result = self.text_generator.get_embedding("test text")

        # Assertions
        self.assertIsNotNone(result)
        self.assertIsInstance(result, torch.Tensor)
        # Should be shape [1, 768] - a single embedding vector
        self.assertEqual(result.dim(), 2)
        self.assertEqual(result.shape[0], 1)

    def test_device_handling(self):
        """Test that the get_embedding method handles device properly."""
        # Set up mocks
        mock_encoding = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        self.text_generator.tokenizer.return_value = mock_encoding
        mock_output = MagicMock()
        mock_output.hidden_states = [torch.rand(1, 3, 768) for _ in range(3)]
        self.text_generator.model.return_value = mock_output

        # Test with device=None
        self.text_generator.device = None
        result1 = self.text_generator.get_embedding("test text")
        self.assertIsNotNone(result1)

        # Test with device='cpu'
        self.text_generator.device = 'cpu'
        result2 = self.text_generator.get_embedding("test text")
        self.assertIsNotNone(result2)

    def test_direct_batchencoding_to_device(self):
        """Test that we properly handle BatchEncoding objects (can't go directly to device)."""
        # Create a mock BatchEncoding that raises TypeError when .to() is called directly
        class MockBatchEncoding(dict):
            def to(self, device):
                raise TypeError("Attempting to cast a BatchEncoding to type None. This is not supported.")

        mock_encoding = MockBatchEncoding({
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        })
        self.text_generator.tokenizer.return_value = mock_encoding

        mock_output = MagicMock()
        mock_output.hidden_states = [torch.rand(1, 3, 768) for _ in range(3)]
        self.text_generator.model.return_value = mock_output

        # Test the function - should not raise an error
        result = self.text_generator.get_embedding("test text")
        self.assertIsNotNone(result)

    @patch('psy_supabase.core.text_generator.TextGenerator._load_model')
    def test_embedding_when_model_not_loaded(self, mock_load_model):
        """Test that embedding works when model needs to be loaded first."""
        # Setup initial state without model
        self.text_generator.model = None
        self.text_generator.tokenizer = None

        # Configure the _load_model mock to set up the model and tokenizer
        def side_effect():
            self.text_generator.model = MagicMock()
            self.text_generator.tokenizer = MagicMock()

            # Configure model and tokenizer behavior
            mock_encoding = {
                'input_ids': torch.tensor([[1, 2, 3]]),
                'attention_mask': torch.tensor([[1, 1, 1]])
            }
            self.text_generator.tokenizer.return_value = mock_encoding

            mock_output = MagicMock()
            mock_output.hidden_states = [torch.rand(1, 3, 768) for _ in range(3)]
            self.text_generator.model.return_value = mock_output

        mock_load_model.side_effect = side_effect

        # Test the function - should trigger model loading
        result = self.text_generator.get_embedding("test text")

        # Assertions
        mock_load_model.assert_called_once()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, torch.Tensor)

    @patch('psy_supabase.core.model_manager.TextGenerator')
    def test_real_batchencoding_handling(self, mock_text_generator_class):
        """Test handling of real BatchEncoding objects using our mocked TextGenerator."""
        # Use our mock text generator
        mock_generator = MockTextGenerator()

        # Add get_embedding method to our mock
        mock_embedding = torch.tensor([[0.1] * 768], dtype=torch.float32)
        mock_generator.get_embedding = MagicMock(return_value=mock_embedding)

        # Make the ModelManager return our mock
        mock_text_generator_class.return_value = mock_generator

        # Get the text generator through ModelManager
        model_manager = ModelManager(model_name="microsoft/phi-1_5")
        text_generator = model_manager.get_generator()

        # Test get_embedding directly
        test_text = "Test this handling of BatchEncoding objects"
        embedding = text_generator.get_embedding(test_text)

        # Verify results
        self.assertIsNotNone(embedding)
        self.assertTrue(torch.equal(embedding, mock_embedding),
                   "Embedding should match mock embedding")
        mock_generator.get_embedding.assert_called_once_with(test_text)

    def test_dict_context_storage(self):
        """Test that dictionary context is properly handled in the database."""
        # Create a dictionary context
        complex_context = {
            "topics": ["anxiety", "depression"],
            "metadata": {"source": "user_query", "timestamp": "2025-04-07"},
            "embedding": [0.1, 0.2, 0.3]
        }

        # Save an interaction with this context
        interaction_id = self.db_manager.save_interaction(
            question="How do I handle anxiety?",
            answer="Practice deep breathing techniques.",
            session_id=self.session_id,
            context=complex_context  # Pass dict context
        )

        self.assertIsNotNone(interaction_id)
        self.assertIsInstance(interaction_id, int)

    def test_dict_context_storage_edge_cases(self):
        """Test that dictionary context is properly handled in the database with edge cases."""
        # Test case 1: Regular dictionary context
        complex_context = {
            "topics": ["anxiety", "depression"],
            "metadata": {"source": "user_query", "timestamp": "2025-04-07"},
            "embedding": [0.1, 0.2, 0.3]
        }

        interaction_id1 = self.db_manager.save_interaction(
            question="How do I handle anxiety?",
            answer="Practice deep breathing techniques.",
            session_id=self.session_id,
            context=complex_context
        )
        self.assertIsNotNone(interaction_id1, "Should save interaction with dictionary context")

        # Test case 2: Nested complex dictionary with unusual values
        nested_context = {
            "topics": ["anxiety", "depression", None, ""],
            "metadata": {
                "source": "user_query",
                "timestamp": "2025-04-07",
                "deep": {
                    "nested": {
                        "value": [1, 2, None, "text"]
                    }
                },
                "empty_dict": {},
                "none_value": None
            },
            "embedding": [0.1, 0.2, 0.3],
            "empty_list": [],
            "mixed_list": [1, "two", None, {"key": "value"}]
        }

        interaction_id2 = self.db_manager.save_interaction(
            question="How do I handle depression?",
            answer="Consider speaking with a therapist.",
            session_id=self.session_id,
            context=nested_context
        )
        self.assertIsNotNone(interaction_id2, "Should save interaction with nested dictionary context")

        # Test case 3: Very large context object
        large_context = {
            "topics": ["anxiety"] * 100,
            "metadata": {"item_" + str(i): "value_" + str(i) for i in range(1000)},
            "embedding": [0.1] * 768
        }

        interaction_id3 = self.db_manager.save_interaction(
            question="Can anxiety be treated?",
            answer="Yes, anxiety is treatable with proper support.",
            session_id=self.session_id,
            context=large_context
        )
        self.assertIsNotNone(interaction_id3, "Should save interaction with large dictionary context")

        # Test case 4: Context with special characters and SQL injection attempts
        injection_context = {
            "topics": ["anxiety; DROP TABLE users;"],
            "sql_injection": "'; DELETE FROM interactions; --",
            "quotes": {"single": "'", "double": '"', "both": "'\""},
            "special": "\n\t\r\0\b"
        }

        interaction_id4 = self.db_manager.save_interaction(
            question="How do I handle anxiety?",
            answer="Practice deep breathing techniques.",
            session_id=self.session_id,
            context=injection_context
        )
        self.assertIsNotNone(interaction_id4, "Should safely save interaction with potentially harmful context")

        # Test case 5: None context
        interaction_id5 = self.db_manager.save_interaction(
            question="What is mindfulness?",
            answer="Mindfulness is being present in the moment.",
            session_id=self.session_id,
            context=None
        )
        self.assertIsNotNone(interaction_id5, "Should save interaction with None context")

    @patch('psy_supabase.core.model_manager.get_embedding_provider')
    def test_embedding_end_to_end(self, mock_get_provider):
        """Test the full embedding generation pipeline from query to storage."""
        # Create a mock embedding provider
        mock_provider = MagicMock()
        mock_provider.generate_embedding.return_value = [0.1] * 768
        mock_get_provider.return_value = mock_provider

        # Generate an embedding
        test_text = "How can I manage my anxiety and stress levels?"
        embedding = mock_provider.generate_embedding(test_text)

        # Verify it's a valid list of floats
        self.assertIsInstance(embedding, list)
        self.assertTrue(len(embedding) > 0)

        # Use this embedding to create a context
        context = {
            "query_embedding": embedding,
            "extracted_topics": ["anxiety", "stress"],
            "psychological_context": {
                "topic": "anxiety",
                "emotion": "fear"
            }
        }

        # Save an interaction with this context
        interaction_id = self.db_manager.save_interaction(
            question=test_text,
            answer="Here are some techniques to help manage anxiety...",
            session_id=self.session_id,
            context=context
        )

        self.assertIsNotNone(interaction_id)
        self.assertIsInstance(interaction_id, int)


if __name__ == "__main__":
    unittest.main()