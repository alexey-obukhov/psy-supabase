import unittest
from unittest.mock import MagicMock
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from psy_supabase.core.dynamic_rag import DynamicRAGRetriever


class TestAssociativeMemory(unittest.TestCase):
    """Test suite for brain-like associative memory in DynamicRAGRetriever."""

    def setUp(self):
        self.mock_db = MagicMock()
        self.session_id = "test_memory_session"
        self.retriever = DynamicRAGRetriever(self.mock_db, self.session_id)

        # Set up common embedding response
        self.mock_embedding = [0.1] * 384
        self.mock_db.create_embedding.return_value = self.mock_embedding

    def test_associative_query_chain(self):
        """Test that retrieving one memory triggers associated memories."""
        # Initial query about anxiety
        primary_docs = [
            {"content": "Anxiety is a response to stress", "similarity": 0.95,
             "metadata": {"related_topics": ["stress, breathing"]}}
        ]

        # Associated documents that should be retrieved
        associated_docs = [
            {"content": "Deep breathing helps manage stress", "similarity": 0.8,
             "metadata": {"category": "coping_strategies"}}
        ]

        # Setup mock behaviors
        self.mock_db.find_similar_documents_via_rpc.side_effect = [
            primary_docs,  # First call returns primary docs
            associated_docs  # Second call returns associated docs
        ]

        # Call method with associative_memory=True
        result = self.retriever.get_knowledge_by_query("anxiety", associative_memory=True)

        # Verify both primary and associated content is included
        self.assertIn("Anxiety is a response to stress", result)
        self.assertIn("Deep breathing helps manage stress", result)

        # Verify metadata was used to find associations
        self.assertEqual(self.mock_db.find_similar_documents_via_rpc.call_count, 2)

    def test_empty_related_topics(self):
        """Test behavior when documents have no related topics."""
        # Primary docs without any related topics
        primary_docs = [
            {"content": "Anxiety symptoms include worry", "similarity": 0.95,
             "metadata": {"category": "symptoms"}}  # No related_topics field
        ]

        self.mock_db.find_similar_documents_via_rpc.return_value = primary_docs

        # Associative memory should work without error
        result = self.retriever.get_knowledge_by_query("anxiety symptoms", associative_memory=True)

        # Should only include primary content
        self.assertIn("Anxiety symptoms include worry", result)
        self.assertEqual(self.mock_db.find_similar_documents_via_rpc.call_count, 1)

    def test_no_associative_results(self):
        """Test behavior when related topics exist but yield no documents."""
        # Primary docs with related topics
        primary_docs = [
            {"content": "Depression affects mood", "similarity": 0.95,
             "metadata": {"related_topics": ["serotonin", "mood"]}}
        ]

        embedding_calls = []

        def side_effect_create_embedding(text):
            embedding_calls.append(text)
            return self.mock_embedding

        self.mock_db.create_embedding = MagicMock(side_effect=side_effect_create_embedding)

        # Setup dynamic side effects for find_similar_documents_via_rpc
        def side_effect_find_docs(session_id, embedding, **kwargs):
            if len(embedding_calls) == 1:  # First call for main query
                return primary_docs
            else:
                return []  # No results for topic queries

        self.mock_db.find_similar_documents_via_rpc = MagicMock(side_effect=side_effect_find_docs)

        # Call the method
        result = self.retriever.get_knowledge_by_query("depression", associative_memory=True)

        # Should only include primary content
        self.assertIn("Depression affects mood", result)

        # Verify that we attempted to find docs for topics
        self.assertGreater(len(embedding_calls), 1)

    def test_multi_topic_associations(self):
        """Test retrieving associations from multiple related topics."""
        # Primary docs with multiple related topics
        primary_docs = [
            {"content": "PTSD can cause flashbacks", "similarity": 0.9,
             "metadata": {"related_topics": ["trauma", "therapy", "coping"]}}
        ]

        # Associated documents for different topics
        trauma_docs = [
            {"content": "Trauma can have long-lasting effects", "similarity": 0.85}
        ]

        therapy_docs = [
            {"content": "EMDR therapy is effective for PTSD", "similarity": 0.8}
        ]

        coping_docs = [
            {"content": "Coping strategies help manage PTSD symptoms", "similarity": 0.75}
        ]

        # Setup DB manager mock behavior
        self.mock_db.find_similar_documents_via_rpc = MagicMock()

        # First call returns primary docs
        self.mock_db.find_similar_documents_via_rpc.return_value = primary_docs

        embedding_calls = []

        def side_effect_create_embedding(text):
            embedding_calls.append(text)
            return self.mock_embedding

        self.mock_db.create_embedding = MagicMock(side_effect=side_effect_create_embedding)

        # Setup dynamic side effects for find_similar_documents_via_rpc
        # This is the key fix - we need to respond differently based on the embedding input
        def side_effect_find_docs(session_id, embedding, **kwargs):
            if len(embedding_calls) == 1:  # First call with main query
                return primary_docs
            elif "trauma" in embedding_calls[-1]:
                return trauma_docs
            elif "therapy" in embedding_calls[-1]:
                return therapy_docs
            elif "coping" in embedding_calls[-1]:
                return coping_docs
            return []

        self.mock_db.find_similar_documents_via_rpc = MagicMock(side_effect=side_effect_find_docs)

        # Call with associative memory
        result = self.retriever.get_knowledge_by_query("PTSD", associative_memory=True)

        # Should include content from all topics
        self.assertIn("PTSD can cause flashbacks", result)
        self.assertIn("Trauma can have long-lasting effects", result)
        self.assertIn("EMDR therapy is effective for PTSD", result)

    def test_caching_with_associations(self):
        """Test that caching works properly with associative memory."""
        # Setup docs
        primary_docs = [
            {"content": "Mindfulness reduces anxiety", "similarity": 0.95,
             "metadata": {"related_topics": ["meditation"]}}
        ]

        associated_docs = [
            {"content": "Meditation improves focus", "similarity": 0.8}
        ]

        # Setup mock behavior
        self.mock_db.find_similar_documents_via_rpc.side_effect = [
            primary_docs,
            associated_docs
        ]

        # First call should query the database
        result1 = self.retriever.get_knowledge_by_query("mindfulness", associative_memory=True)
        self.assertEqual(self.mock_db.find_similar_documents_via_rpc.call_count, 2)

        # Reset mock to verify it's not called again
        self.mock_db.find_similar_documents_via_rpc.reset_mock()

        # Second call should use cache
        result2 = self.retriever.get_knowledge_by_query("mindfulness", associative_memory=True)
        self.assertEqual(self.mock_db.find_similar_documents_via_rpc.call_count, 0)

        # Results should be identical
        self.assertEqual(result1, result2)

    def test_format_of_associated_results(self):
        """Test that associated memories are properly formatted."""
        # Setup docs
        primary_docs = [
            {"content": "Insomnia is difficulty sleeping", "similarity": 0.9,
             "metadata": {"related_topics": ["sleep"]}}
        ]

        associated_docs = [
            {"content": "Sleep hygiene improves rest quality", "similarity": 0.85}
        ]

        # Setup mock
        self.mock_db.find_similar_documents_via_rpc.side_effect = [
            primary_docs,
            associated_docs
        ]

        # Get results
        result = self.retriever.get_knowledge_by_query("insomnia", associative_memory=True)

        # Check that associated memories are marked differently
        self.assertIn("(Relevance: 0.90)", result)  # Primary result
        self.assertIn("(Associated Memory, Relevance: 0.85)", result)


if __name__ == '__main__':
    unittest.main()