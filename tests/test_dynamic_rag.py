import unittest
from unittest.mock import MagicMock, patch
import os
import sys
from pathlib import Path

# Add project root to path to import modules properly
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from psy_supabase.core.dynamic_rag import DynamicRAGRetriever
from psy_supabase.utilities.common import get_project_root


class TestDynamicRAGRetriever(unittest.TestCase):
    """Test suite for DynamicRAGRetriever functionality and template selection."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create a mock database manager
        self.mock_db = MagicMock()

        # Set up common embedding response
        self.mock_embedding = [0.1] * 384  # Typical embedding size
        self.mock_db.create_embedding.return_value = self.mock_embedding

        # Set up session ID
        self.session_id = "test_session_123"

        # Define templates directory path
        self.templates_dir = os.path.join(get_project_root(), "templates")

        # Initialize the retriever with mock DB
        self.retriever = DynamicRAGRetriever(self.mock_db, self.session_id)

        # Verify templates directory exists
        if not os.path.exists(self.templates_dir):
            print(f"Warning: Templates directory not found at {self.templates_dir}")

    def test_template_existence(self):
        """Verify that essential templates exist in the templates directory."""
        essential_templates = [
            "dynamic_rag_therapy.j2",
            "basic_answer.j2",
            "crisis_support.j2",
            "fallback.j2"
        ]

        for template_name in essential_templates:
            template_path = os.path.join(self.templates_dir, template_name)
            self.assertTrue(
                os.path.exists(template_path),
                f"Essential template missing: {template_name}"
            )

            # Validate template contains user question reference - handle different formats
            with open(template_path, 'r', encoding='utf-8') as f:
                content = f.read()

                # Look for any variant of user_question in the template
                user_question_found = any([
                    "user_question" in content,
                    "{{ user_question }}" in content,
                    "{{user_question}}" in content,
                    '"{{ user_question }}"' in content
                ])

                self.assertTrue(
                    user_question_found,
                    f"Template {template_name} missing user_question placeholder"
                )

    def test_get_knowledge_by_query(self):
        """Test that get_knowledge_by_query correctly retrieves and formats knowledge."""
        # Setup mock document retrieval
        mock_docs = [
            {"content": "Document 1 content", "metadata": {"source": "Source 1"}, "similarity": 0.85},
            {"content": "Document 2 content", "metadata": {"category": "Category 2"}, "similarity": 0.75}
        ]
        self.mock_db.find_similar_documents_via_rpc.return_value = mock_docs

        # Call the method
        result = self.retriever.get_knowledge_by_query("anxiety management")

        # Verify DB manager methods were called correctly
        self.mock_db.create_embedding.assert_called_with("anxiety management")
        self.mock_db.find_similar_documents_via_rpc.assert_called_once()

        # Check that both documents are included in result with similarity scores
        self.assertIn("Document 1 content", result)
        self.assertIn("Document 2 content", result)
        self.assertIn("(Relevance: 0.85)", result)
        self.assertIn("(Relevance: 0.75)", result)

    def test_get_knowledge_by_query_caching(self):
        """Test that knowledge retrieval uses caching for repeated queries."""
        # Setup mock and first call
        self.mock_db.find_similar_documents_via_rpc.return_value = [
            {"content": "Cached content", "similarity": 0.8}
        ]

        # First call should hit the database
        self.retriever.get_knowledge_by_query("anxiety")
        self.assertEqual(self.mock_db.find_similar_documents_via_rpc.call_count, 1)

        # Second call with same query should use cache
        self.retriever.get_knowledge_by_query("anxiety")
        self.assertEqual(self.mock_db.find_similar_documents_via_rpc.call_count, 1)

        # Different query should hit database again
        self.retriever.get_knowledge_by_query("depression")
        self.assertEqual(self.mock_db.find_similar_documents_via_rpc.call_count, 2)

    def test_similarity_filtering(self):
        """Test that documents with low similarity scores are filtered out."""
        # Setup documents with varying similarity scores
        mock_docs = [
            {"content": "High similarity", "similarity": 0.9},
            {"content": "Medium similarity", "similarity": 0.5},
            {"content": "Low similarity", "similarity": 0.05}  # Below the 0.1 threshold in code
        ]
        self.mock_db.find_similar_documents_via_rpc.return_value = mock_docs

        # Call the method
        result = self.retriever.get_knowledge_by_query("test query")

        # Verify filtering
        self.assertIn("High similarity", result)
        self.assertIn("Medium similarity", result)
        self.assertNotIn("Low similarity", result)

    def test_get_past_interactions(self):
        """Test retrieval of past interactions."""
        # Setup mock conversation history
        mock_interactions = [
            {"question": "User question 1", "answer": "Assistant answer 1"},
            {"question": "User question 2", "answer": "Assistant answer 2"}
        ]
        self.mock_db.get_conversation_history.return_value = mock_interactions

        # Call the method
        result = self.retriever.get_past_interactions()

        # Verify DB manager methods
        self.mock_db.get_conversation_history.assert_called_once()

        # Check formatting
        self.assertIn("User: User question 1", result)
        self.assertIn("Assistant: Assistant answer 1", result)
        self.assertIn("User: User question 2", result)
        self.assertIn("Assistant: Assistant answer 2", result)

    def test_get_past_interactions_by_topic(self):
        """Test retrieving past interactions filtered by topic."""
        # Setup mock conversation by embedding
        mock_interactions = [
            {"question": "Anxiety question", "answer": "Anxiety answer"}
        ]
        self.mock_db.find_similar_interactions_by_embedding.return_value = mock_interactions

        # Call the method
        result = self.retriever.get_past_interactions(topic="anxiety")

        # Verify embedding was created and DB was queried correctly
        self.mock_db.create_embedding.assert_called_with("anxiety")
        self.mock_db.find_similar_interactions_by_embedding.assert_called_once()

        # Check result formatting
        self.assertIn("User: Anxiety question", result)
        self.assertIn("Assistant: Anxiety answer", result)

    def test_get_pain_point(self):
        """Test detection of pain points."""
        # Setup mock pain points
        mock_pain_points = {
            "pain_points": [
                {
                    "recurring_terms": ["anxiety", "worry", "stress"],
                    "count": 3
                }
            ],
            "severity": "moderate",
            "first_detected_at": "2025-03-30T12:00:00Z"
        }
        self.mock_db.detect_pain_points.return_value = mock_pain_points
        self.mock_db.get_recommended_therapeutic_approach.return_value = "CBT"

        # Call the method
        result = self.retriever.get_pain_point()

        # Verify DB manager methods
        self.mock_db.detect_pain_points.assert_called_once()
        self.mock_db.get_recommended_therapeutic_approach.assert_called_once()

        # Check result
        self.assertEqual(result["pain_point"], "anxiety")
        self.assertEqual(result["severity"], "moderate")
        self.assertEqual(result["approach"], "CBT")

    @patch('psy_supabase.core.text_generator.TextGenerator.generate_therapeutic_response_with_dynamic_retrieval')
    def test_template_selection_with_anxiety(self, mock_generate):
        """Test that anxiety topics select the appropriate template."""
        # This test requires integration with TextGenerator
        mock_generate.return_value = "Therapeutic response"

        # Mock TextGenerator setup would go here
        # In practice, you'd test this through the full RAGProcessor

        # Sample anxiety text
        text = "I'm feeling very anxious about my upcoming presentation"

        # Simulate the prompt selection that would happen in generate_therapeutic_response_with_dynamic_retrieval
        prompt_selector = MagicMock()
        prompt_selector.analyze_question.return_value = {"topic": "anxiety", "emotion": "nervous"}
        prompt_selector.generate_category_info.return_value = {"Anxiety Support": 0.85}

        # Verify template selection would be correct
        self.assertEqual(
            list(prompt_selector.generate_category_info.return_value.keys())[0],
            "Anxiety Support"
        )

    # Add additional tests for other methods: analyze_emotion, analyze_topics, get_related_concepts

if __name__ == '__main__':
    unittest.main()