import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from psy_supabase.core.dynamic_rag import DynamicRAGRetriever


class TestDynamicRAGEdgeCases(unittest.TestCase):
    """Test suite for edge cases in DynamicRAGRetriever."""

    def setUp(self):
        """Set up test environment before each test."""
        self.mock_db = MagicMock()
        self.session_id = "test_session_123"
        self.retriever = DynamicRAGRetriever(self.mock_db, self.session_id)

    def test_empty_input(self):
        """Test handling of empty input strings."""
        # Test with empty query
        result = self.retriever.get_knowledge_by_query("")
        self.assertEqual(result, "", "Empty query should return empty result")

        # Test with None query (should handle gracefully)
        result = self.retriever.get_knowledge_by_query(None)
        self.assertEqual(result, "", "None query should return empty result")

    def test_special_characters_input(self):
        """Test handling of input with special characters."""
        special_query = "!@#$%^&*()"
        self.mock_db.create_embedding.return_value = [0.1] * 384
        self.mock_db.find_similar_documents_via_rpc.return_value = []

        # Should not raise exceptions
        result = self.retriever.get_knowledge_by_query(special_query)
        self.assertEqual(result, "", "Special characters should be handled gracefully")

    def test_database_error_handling(self):
        """Test handling of database errors."""
        # Setup DB to raise exception
        self.mock_db.create_embedding.side_effect = Exception("Database connection failure")

        # Should catch exception and return empty string
        result = self.retriever.get_knowledge_by_query("test query")
        self.assertEqual(result, "", "Database errors should be handled gracefully")

    def test_reset_cache(self):
        """Test that cache reset works properly."""
        # Populate cache
        self.retriever.query_cache = {
            "test_key": "test_value",
            "another_key": "another_value"
        }

        # Reset cache
        self.retriever.reset_cache()

        # Verify cache is empty
        self.assertEqual(len(self.retriever.query_cache), 0, "Cache should be empty after reset")

    def test_analyze_emotion_with_empty_input(self):
        """Test emotion analysis with empty input."""
        self.mock_db.create_embedding.return_value = None

        result = self.retriever.analyze_emotion("")
        self.assertIn("error", result, "Empty input should produce error message")

    def test_analyze_emotion_with_valid_input(self):
        """Test emotion analysis with valid input."""
        # Mock DB responses
        self.mock_db.create_embedding.return_value = [0.1] * 384
        self.mock_db.analyze_text_emotional_spectrum.return_value = {
            "primary_emotion": "joy",
            "intensity": 0.8,
            "spectrum": [{"emotion": "joy", "score": 0.8}, {"emotion": "surprise", "score": 0.3}]
        }

        result = self.retriever.analyze_emotion("I am feeling really happy today!")

        # Verify result
        self.assertEqual(result["primary_emotion"], "joy")
        self.assertEqual(result["intensity"], 0.8)
        self.assertEqual(len(result["spectrum"]), 2)

    def test_analyze_emotion_caching(self):
        """Test that emotion analysis results are properly cached."""
        # Setup mock
        self.mock_db.create_embedding.return_value = [0.1] * 384
        self.mock_db.analyze_text_emotional_spectrum.return_value = {"primary_emotion": "anger"}

        # First call - should hit database
        self.retriever.analyze_emotion("I am angry")
        self.assertEqual(self.mock_db.analyze_text_emotional_spectrum.call_count, 1)

        # Second call with same input - should use cache
        self.retriever.analyze_emotion("I am angry")
        self.assertEqual(self.mock_db.analyze_text_emotional_spectrum.call_count, 1,
                         "Should not call database again for same input")

    def test_get_related_concepts_disabled_queries(self):
        """Test behavior when dynamic queries are disabled."""
        # Create retriever with dynamic queries disabled
        retriever = DynamicRAGRetriever(self.mock_db, self.session_id, allow_dynamic_queries=False)

        result = retriever.get_related_concepts("anxiety")
        self.assertEqual(result, "Dynamic querying is disabled.")

    def test_standardize_cache_key(self):
        """Test cache key standardization."""
        # Test normal input
        key = self.retriever._standardize_cache_key("Test String")
        self.assertEqual(key, "test_string")

        # Test empty input
        key = self.retriever._standardize_cache_key("")
        self.assertEqual(key, "none")

        # Test None input
        key = self.retriever._standardize_cache_key(None)
        self.assertEqual(key, "none")

    def test_get_pain_point_no_session(self):
        """Test pain point detection with no session ID."""
        # Create retriever with no session ID
        retriever = DynamicRAGRetriever(self.mock_db, "", allow_dynamic_queries=True)

        result = retriever.get_pain_point()
        self.assertIsNone(result, "Should return None when no session ID is provided")

    def test_get_past_interactions_unicode(self):
        """Test past interactions with unicode characters."""
        # Mock conversation history with unicode
        self.mock_db.get_conversation_history.return_value = [
            {"question": "¿Cómo estás?", "answer": "Estoy bien, gracias!"},
            {"question": "你好吗?", "answer": "我很好，谢谢!"}
        ]

        result = self.retriever.get_past_interactions()

        # Verify unicode is preserved
        self.assertIn("¿Cómo estás?", result)
        self.assertIn("你好吗?", result)

    def test_analyze_topics_response_handling(self):
        """Test topic analysis with different response formats."""
        # Mock empty response
        self.mock_db.supabase.rpc().execute.return_value.data = []

        result = self.retriever.analyze_topics()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["topic"], "No significant topics identified")

        # Mock valid response
        self.mock_db.supabase.rpc().execute.return_value.data = [
            {"topic": "anxiety", "frequency": 5},
            {"topic": "depression", "frequency": 3}
        ]

        # Reset cache to force re-fetch
        self.retriever.reset_cache()

        result = self.retriever.analyze_topics()
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["topic"], "anxiety")
        self.assertEqual(result[0]["frequency"], 5)


class TestDynamicRAGIntegration(unittest.TestCase):
    """Integration test suite for DynamicRAGRetriever with more realistic scenarios."""

    def setUp(self):
        self.mock_db = MagicMock()
        self.session_id = "test_integration_session_123"
        self.retriever = DynamicRAGRetriever(self.mock_db, self.session_id)

    def test_combined_retrieval_workflow(self):
        """Test a realistic workflow combining multiple retrieval methods."""
        # Setup mocks for each component
        self.mock_db.create_embedding.return_value = [0.1] * 384

        # Knowledge retrieval setup
        self.mock_db.find_similar_documents_via_rpc.return_value = [
            {"content": "Anxiety is a normal response to stress.", "similarity": 0.9}
        ]

        # Past interactions setup - UPDATED IMPLEMENTATION
        self.mock_db.get_conversation_history.return_value = [
            {"question": "I feel anxious about my exam", "answer": "That's understandable."}
        ]

        # Add mock for find_similar_interactions_by_embedding
        self.mock_db.find_similar_interactions_by_embedding.return_value = [
            {"question": "I feel anxious about my exam", "answer": "That's understandable."}
        ]

        # Emotion analysis setup
        self.mock_db.analyze_text_emotional_spectrum.return_value = {
            "primary_emotion": "anxiety",
            "intensity": 0.7,
            "spectrum": [{"emotion": "anxiety", "score": 0.7}]
        }

        # Pain point setup
        self.mock_db.detect_pain_points.return_value = {
            "pain_points": [{"recurring_terms": ["anxiety", "worry"], "count": 3}],
            "severity": "moderate",
            "first_detected_at": "2025-03-30T12:00:00Z"
        }
        self.mock_db.get_recommended_therapeutic_approach.return_value = "CBT"

        # Execute a multi-step workflow
        query = "How can I manage anxiety before an important presentation?"

        # 1. Get knowledge
        knowledge = self.retriever.get_knowledge_by_query(query)
        self.assertIn("Anxiety is a normal response to stress", knowledge)

        # 2. Get past interactions
        interactions = self.retriever.get_past_interactions(topic="anxiety")
        self.assertIn("I feel anxious about my exam", interactions)

        # 3. Analyze emotion
        emotion = self.retriever.analyze_emotion(query)
        self.assertEqual(emotion["primary_emotion"], "anxiety")

        # 4. Get pain point
        pain_point = self.retriever.get_pain_point()
        self.assertEqual(pain_point["pain_point"], "anxiety")
        self.assertEqual(pain_point["approach"], "CBT")

        # Verify cache has been populated with all these items
        self.assertGreaterEqual(len(self.retriever.query_cache), 4)

    @patch('time.time')
    def test_large_scale_caching(self, mock_time):
        """Test caching behavior with many sequential queries."""
        # Mock time to ensure consistent cache keys
        mock_time.return_value = 1648717800.0

        # Setup DB mock to return data
        self.mock_db.create_embedding.return_value = [0.1] * 384
        self.mock_db.find_similar_documents_via_rpc.return_value = [
            {"content": "Test content", "similarity": 0.9}
        ]

        # Make multiple unique queries
        for i in range(100):
            query = f"test query {i}"
            self.retriever.get_knowledge_by_query(query)

        # Verify DB was called 100 times (once per unique query)
        self.assertEqual(self.mock_db.find_similar_documents_via_rpc.call_count, 100)

        # Reset cache and verify
        self.retriever.reset_cache()
        self.assertEqual(len(self.retriever.query_cache), 0)


if __name__ == '__main__':
    unittest.main()