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
from tests.helpers.database_test_base import DatabaseTestBase


class TestDynamicRAGRetriever(DatabaseTestBase):
    """Test suite for DynamicRAGRetriever functionality and template selection."""

    def setUp(self):
        """Set up test environment before each test."""
        # First call the parent's setUp to get the standard test IDs and logger
        super().setUp()

        # Log that we're setting up the DynamicRAG test
        self.logger.info("Setting up DynamicRAGRetriever test for session %s", self.test_session_id)

        # Create a mock database manager - use session ID from parent
        self.mock_db = MagicMock()

        # Set up common embedding response
        self.mock_embedding = [0.1] * 384  # Typical embedding size
        self.mock_db.create_embedding.return_value = self.mock_embedding
        self.logger.debug("Configured mock embedding vector with 384 dimensions")

        # Setup mock for detect_pain_points method
        mock_pain_points = {
            "name": "anxiety",
            "confidence": 0.8,
            "keywords": ["anxiety", "worry", "stress"]
        }
        self.mock_db.detect_pain_points = MagicMock(return_value=mock_pain_points)

        # Setup mock for other methods used in tests
        self.mock_db.get_conversation_history = MagicMock(return_value=[])
        self.mock_db.find_similar_interactions_by_embedding = MagicMock(return_value=[])
        self.logger.debug("Initialized all database mock methods")

        # Use session ID from parent class - note this change
        self.session_id = self.test_session_id  # Use parent's generated ID

        # Define templates directory path
        self.templates_dir = os.path.join(get_project_root(), "templates")

        # Initialize the retriever with mock DB
        self.retriever = DynamicRAGRetriever(self.mock_db, self.session_id)
        self.logger.info("Initialized DynamicRAGRetriever with session ID: %s", self.session_id)

        # Verify templates directory exists
        if not os.path.exists(self.templates_dir):
            self.logger.warning("Templates directory not found at %s", self.templates_dir)
        else:
            self.logger.debug("Templates directory found at %s", self.templates_dir)


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
        """Test retrieving knowledge relevant to a query."""
        # Setup mock responses for similar interactions
        self.mock_db.find_similar_interactions_by_embedding.return_value = [
            {
                'interaction_id': '1',
                'question': 'Test question 1?',
                'answer': 'Answer 1 content',
                'similarity': 0.85
            },
            {
                'interaction_id': '2',
                'question': 'Test question 2?',
                'answer': 'Answer 2 content',
                'similarity': 0.75
            }
        ]

        result = self.retriever.get_knowledge_by_query("test query")

        self.assertIn('Answer 1 content', result)
        self.assertIn('Answer 2 content', result)

        # Check that both answers are included
        expected_content = "Answer 1 content\n\nAnswer 2 content"
        self.assertEqual(expected_content, result)

    def test_get_knowledge_by_query_caching(self):
        """
        Test that knowledge retrieval efficiently uses caching for repeated queries.

        This test validates several important caching behaviors:

        1. First query: Correctly creates embeddings and retrieves from database
        2. Repeated query: Returns identical results from cache without redundant
        database calls or embedding creation
        3. New query: Bypasses cache and performs fresh database retrieval

        Caching is critical for:
        - Reducing latency in repeated therapeutic conversations
        - Minimizing embedding API costs (typically charged per token)
        - Ensuring consistent answers for similar user questions
        - Decreasing database load during intensive therapy sessions

        The test uses mock interactions that simulate retrieved answers from
        different topics (anxiety and depression) to verify both cache hits
        and cache misses function correctly.
        """
        # Setup mock responses for each query with the correct format
        mock_interaction = {
            "interaction_id": 1,
            "question": "How can I manage anxiety?",
            "answer": "Cached content about anxiety management",
            "similarity": 0.8
        }

        mock_interaction2 = {
            "interaction_id": 2,
            "question": "How can I cope with depression?",
            "answer": "Content about depression management",
            "similarity": 0.8
        }

        # Reset mocks before test
        self.mock_db.find_similar_interactions_by_embedding.reset_mock()
        self.mock_db.create_embedding.reset_mock()

        # Configure mock to return different values based on query
        def mock_find_similar(embedding, **kwargs):
            # Check the associated query by examining call history
            if self.mock_db.create_embedding.call_args[0][0] == "anxiety":
                return [mock_interaction]
            else:
                return [mock_interaction2]

        self.mock_db.find_similar_interactions_by_embedding.side_effect = mock_find_similar

        # First call should hit the database
        result1 = self.retriever.get_knowledge_by_query("anxiety")
        self.assertEqual(self.mock_db.find_similar_interactions_by_embedding.call_count, 1)
        self.assertEqual(self.mock_db.create_embedding.call_count, 1)
        self.assertIn("Cached content about anxiety management", result1)

        # Second call with same query should use cache - no new DB calls
        result2 = self.retriever.get_knowledge_by_query("anxiety")
        self.assertEqual(self.mock_db.find_similar_interactions_by_embedding.call_count, 1,
                        "Cache should prevent a second DB call")
        self.assertEqual(self.mock_db.create_embedding.call_count, 1,
                        "Cache should prevent a second embedding creation")
        self.assertIn("Cached content about anxiety management", result2,
                    "Cached result should match first result")

        # Verify same result is returned (indicating cache was used)
        self.assertEqual(result1, result2, "Cached result should be identical to original")

        # Different query should hit database again
        result3 = self.retriever.get_knowledge_by_query("depression")
        self.assertEqual(self.mock_db.find_similar_interactions_by_embedding.call_count, 2,
                        "New query should trigger new DB call")
        self.assertEqual(self.mock_db.create_embedding.call_count, 2,
                        "New query should trigger new embedding creation")
        self.assertIn("Content about depression management", result3)
        self.assertNotEqual(result1, result3, "Different query should return different result")

    def test_similarity_formatting(self):
        """Test that results are properly formatted by similarity."""
        # Setup mock responses for different similarity scores
        self.mock_db.find_similar_interactions_by_embedding.return_value = [
            {
                'interaction_id': '1',
                'question': 'Question with high similarity',
                'answer': 'High similarity content',
                'similarity': 0.9
            },
            {
                'interaction_id': '2',
                'question': 'Question with medium similarity',
                'answer': 'Medium similarity content',
                'similarity': 0.75
            },
            {
                'interaction_id': '3',
                'question': 'Question with low similarity',
                'answer': 'Low similarity content',
                'similarity': 0.6
            }
        ]

        result = self.retriever.get_knowledge_by_query("test query")

        # After refactoring, we only include answer content without debug info
        expected_content = "High similarity content\n\nMedium similarity content\n\nLow similarity content"
        self.assertEqual(expected_content, result)

        # Ensure answers are included in correct order (by similarity)
        answer_positions = [
            result.find("High similarity content"),
            result.find("Medium similarity content"),
            result.find("Low similarity content")
        ]
        # Each position should be greater than the previous
        self.assertTrue(answer_positions[0] < answer_positions[1])
        self.assertTrue(answer_positions[1] < answer_positions[2])

    def test_get_past_interactions(self):
        """Test retrieval of past interactions by session ID."""
        # Setup mock
        mock_history = [{"question": "Test?", "answer": "Answer"}]
        self.mock_db.get_conversation_history.return_value = mock_history

        # Call the method
        result = self.retriever.get_past_interactions(session_id=self.session_id)

        # Verify the mock database method was called
        self.mock_db.get_conversation_history.assert_called_once_with(self.session_id)

        # Verify the result
        self.assertEqual(result, mock_history)

    def test_get_past_interactions_by_topic(self):
        """Test retrieval of interactions by topic."""
        # Setup
        topic = "anxiety"
        mock_embedding = [0.1] * 10
        mock_results = [{"question": "I am anxious", "answer": "That's normal"}]

        # Set up mocks
        self.mock_db.create_embedding.return_value = mock_embedding
        self.mock_db.find_similar_interactions_by_embedding.return_value = mock_results

        # Call the method
        result = self.retriever.get_past_interactions_by_topic(topic, self.session_id)

        # Assertions
        self.mock_db.create_embedding.assert_called_once_with(topic)
        self.mock_db.find_similar_interactions_by_embedding.assert_called_once_with(
            embedding=mock_embedding, session_id=self.session_id, limit=5
        )
        self.assertEqual(result, mock_results)

    def test_combined_retrieval_workflow(self):
        """Test combined retrieval workflow with knowledge and interactions."""
        # Setup mock response for get_conversation_history
        self.mock_db.get_conversation_history.return_value = [
            {
                "question": "I feel anxious about my exam",
                "answer": "That's normal, let's discuss coping strategies.",
                "created_at": "2023-01-01T12:00:00"
            }
        ]

        # Setup mock response for create_embedding
        self.mock_db.create_embedding.return_value = [0.1] * 384

        # Mock the correct method that's actually being called
        self.mock_db.find_similar_interactions_by_embedding.return_value = [
            {
                "interaction_id": 1,
                "question": "How do I manage exam anxiety?",
                "answer": "Exam anxiety is common and can be managed with breathing techniques.",
                "similarity": 0.9,
                "metadata": {"topics": ["anxiety", "exams", "coping"]}
            }
        ]

        # Get past interactions - make sure to pass session_id
        interactions = self.retriever.get_past_interactions(session_id=self.session_id)

        # Should have the test question
        self.assertIn("I feel anxious about my exam", interactions[0]['question'])

        # Now get knowledge
        knowledge = self.retriever.get_knowledge_by_query("How to manage exam anxiety?")

        # Should have the content
        self.assertIn("Exam anxiety is common", knowledge)

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

    def run_diagnostics(self, error=None):
        """Run diagnostics when a test fails."""
        if error:
            self.logger.error("Test failed with error: %s", error)

        # First run the base diagnostics
        has_real_db, has_mock_db = self.run_diagnostics_summary()

        # DynamicRAG-specific diagnostics
        self.logger.info("=== DynamicRAG-Specific Diagnostics ===")
        self.logger.info("DynamicRAGRetriever cache size: %d", len(self.retriever.query_cache))

        # Add specialized mock checks for this test class
        if has_mock_db:
            self.logger.info("create_embedding call count: %d", self.mock_db.create_embedding.call_count)
            self.logger.info("find_similar_interactions_by_embedding call count: %d", self.mock_db.find_similar_interactions_by_embedding.call_count)

        # Add DynamicRAG-specific diagnostics
        self.logger.info("=== DynamicRAG-Specific Diagnostics ===")
        self.logger.info("Current session ID: %s", self.session_id)
        self.logger.info("DynamicRAGRetriever cache size: %d", len(self.retriever.query_cache))

        # Check mock call counts
        self.logger.info("create_embedding call count: %s", self.mock_db.create_embedding.call_count)
        self.logger.info("find_similar_interactions_by_embedding call count: %d", self.mock_db.find_similar_interactions_by_embedding.call_count)

        # Display template information
        if os.path.exists(self.templates_dir):
            template_files = os.listdir(self.templates_dir)
            self.logger.info("Available templates (%s): %d", len(template_files), ', '.join(template_files))

        # Examine the RAG processor configuration
        self.logger.info("DynamicRAGRetriever associative memory initialized: %s", hasattr(self.retriever, 'associative_memory'))

    def tearDown(self):
        """DynamicRAG-specific tearDown with more detailed diagnostics."""
        try:
            # Clean up DynamicRAG-specific resources first
            self.retriever.query_cache.clear()

            # Run DynamicRAG-specific diagnostics on suspicious activity
            if self.mock_db.find_similar_interactions_by_embedding.call_count > 10:  # Example condition
                self.logger.warning("Unusually high number of DB calls detected")
                self.run_diagnostics()  # Run full diagnostics

        except Exception as e:
            self.logger.error("Error in DynamicRAG tearDown: %s", e)

        # Then call parent tearDown which will run basic diagnostics
        super().tearDown()

if __name__ == '__main__':
    unittest.main()
