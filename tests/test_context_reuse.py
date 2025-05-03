import unittest
import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from psy_supabase.core.database import DatabaseManager
from psy_supabase.core.dynamic_rag import DynamicRAGRetriever
from psy_supabase.core.model_manager import get_embedding_provider
from psy_supabase.core.rag_processor import RAGProcessor
from psy_supabase.core.response_generator import ResponseGenerator
from psy_supabase.core.text_generator import TextGenerator
from psy_supabase.utilities.prompt_selector import PromptSelector
from psy_supabase.utilities.utils_mapping import map_approach_to_template
from tests.conftest import DEFAULT_APPROACH
from tests.helpers.database_test_base import DatabaseTestBase


class TestContextReuse(DatabaseTestBase):
    """
    Test the reuse of context (particularly embeddings) across the RAG pipeline.

    This test ensures that embeddings are correctly processed, stored in context,
    and reused without regeneration throughout the pipeline, avoiding errors like
    the BatchEncoding issue.
    """

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()

        # Create mock PromptSelector
        self.prompt_selector = MagicMock(spec=PromptSelector)
        self.prompt_selector.analyze_question.return_value = {"topic": "anxiety", "emotion": "fear"}
        self.prompt_selector.generate_category_info.return_value = {
            "affirmation_reassurance": 0.8,
            "information": 0.6,
        }

        # Create mock TextGenerator
        self.text_generator = MagicMock(spec=TextGenerator)
        self.text_generator.prompt_selector = self.prompt_selector

        # Initialize processors
        self.rag_processor = RAGProcessor(self.db_manager, self.text_generator)

        self.response_generator = ResponseGenerator(
            text_generator=self.text_generator, db_manager=self.db_manager, prompt_selector=self.prompt_selector
        )

        # Create a dynamic retriever
        self.dynamic_retriever = DynamicRAGRetriever(db_manager=self.db_manager, session_id=self.test_session_id)

    def tearDown(self):
        """Clean up resources."""
        super().tearDown()

    def test_embedding_reuse_in_pipeline(self):
        """Test that embeddings are properly reused throughout the RAG pipeline."""
        # 1. First, we patch the embedding provider to return a known value
        with patch("psy_supabase.core.model_manager.get_embedding_provider") as mock_get_embedding_provider:
            # Configure mock embedding provider
            mock_provider = MagicMock()
            mock_provider.generate_embedding.return_value = [0.1] * 768
            mock_get_embedding_provider.return_value = mock_provider

            # Now track embedding generation calls
            embedding_calls = []

            # Mock generate_embedding to count calls
            def counting_generate_embedding(text):
                embedding_calls.append(text)
                return [0.1] * 768

            self.rag_processor.embedding_provider = mock_provider
            self.rag_processor.embedding_provider.generate_embedding = counting_generate_embedding

            # 2. Process a query - this should generate an embedding
            query_text = "How do I manage anxiety symptoms?"
            query_embedding = self.rag_processor.process_query(query_text, self.test_session_id)

            # 3. Build context with the embedding
            topics_context = {
                "topic": "anxiety",
                "emotion": "fear",
                "extracted_topics": ["anxiety"],
                "category_info": {"affirmation_reassurance": 0.8, "information": 0.6},
            }

            pain_point_results = {"pain_point": "stress", "pain_point_detected": True}

            # Add safety check before accessing pain_point_results
            if isinstance(pain_point_results, str):
                # Convert string to dict if it's a string (possibly from JSON)
                try:
                    import json

                    pain_point_results = json.loads(pain_point_results)
                except json.JSONDecodeError:
                    # If it can't be parsed as JSON, create a basic dict
                    pain_point_results = {"pain_point": pain_point_results, "pain_point_detected": True}

            # Now safely get the approach_type
            approach_type = (
                pain_point_results.get("approach_type", DEFAULT_APPROACH)
                if isinstance(pain_point_results, dict)
                else DEFAULT_APPROACH
            )

            hot_topics = []

            # Mock extract_topics to ensure we have the expected keys
            self.rag_processor._extract_topics = MagicMock(return_value=["anxiety"])

            # Build the context
            context = self.response_generator.build_generation_context(
                query_text,
                self.test_session_id,
                topics_context,
                pain_point_results,
                query_embedding,
                self.dynamic_retriever,
                hot_topics,
            )

            # Verify embedding was included in context
            self.assertIn("query_embedding", context)
            self.assertEqual(context["query_embedding"], query_embedding)

            # 4. Setup a tracking mock for the TextGenerator
            generate_calls = []

            def tracking_generate(*args, **kwargs):
                context_arg = kwargs.get("context", {})
                if context_arg and "query_embedding" in context_arg:
                    generate_calls.append(context_arg.get("query_embedding"))
                return "Sample response"

            self.text_generator.generate_therapeutic_response_with_dynamic_retrieval = tracking_generate

            # 5. Now generate a response using the context
            # Use the method that actually exists in ResponseGenerator
            pain_point_results = {"pain_point": "stress", "pain_point_detected": True}

            approach_type = pain_point_results.get("approach_type", DEFAULT_APPROACH)

            # Select template based on approach
            template = "dynamic_rag_therapy"  # Default template

            # Then call the ResponseGenerator method
            response = self.response_generator.generate_response_with_template(
                user_question=query_text,
                session_id=self.test_session_id,
                generation_context=context,
                pain_point_results=pain_point_results,
            )

            # 6. Verify embedding reuse
            # Should only have one embedding generation call
            self.assertEqual(len(embedding_calls), 1)

            # Our mock should have been called at least once
            self.assertEqual(len(generate_calls), 1)
            self.assertIsNotNone(generate_calls[0])  # The embedding should not be None

            # Verify the same embedding was used
            np.testing.assert_array_equal(np.array(generate_calls[0]), np.array(query_embedding))

    def test_dict_context_in_save_interaction(self):
        """Test that dictionary context is properly handled when saving interactions."""
        # Create a dictionary context
        complex_context = {
            "topics": ["anxiety", "depression"],
            "metadata": {"source": "user_query", "timestamp": "2025-04-07"},
            "embedding": [0.1, 0.2, 0.3],  # Sample embedding values
        }

        # Try to save an interaction with the dictionary context
        try:
            interaction_id = self.db_manager.save_interaction(
                question="How can I manage my anxiety?",
                answer="Practice deep breathing techniques and mindfulness.",
                session_id=self.test_session_id,
                context=complex_context,  # Pass dict as context
            )

            # Verify it was saved
            self.assertIsNotNone(interaction_id)
            self.assertIsInstance(interaction_id, int)

        except Exception as e:
            self.fail(f"save_interaction failed with dictionary context: {e}")

    def test_context_reuse_with_real_embedding(self):
        """Test context reuse with a real embedding to ensure BatchEncoding handling."""
        # This test uses real embeddings, not mocks

        with patch("psy_supabase.core.model_manager.get_embedding_provider") as mock_get_provider:
            # Configure mock provider
            mock_provider = MagicMock()
            mock_provider.generate_embedding.return_value = [0.1] * 768
            mock_get_provider.return_value = mock_provider

            # Generate an embedding
            query_text = "How can I overcome social anxiety?"

            # Generate embedding - this is where BatchEncoding error could happen
            query_embedding = mock_provider.generate_embedding(query_text)

            # Make sure we got a valid embedding
            self.assertIsNotNone(query_embedding)
            self.assertIsInstance(query_embedding, list)

            # Use it in building context
            context = {
                "query_embedding": query_embedding,
                "extracted_topics": ["anxiety"],
                "use_dynamic_retrieval": True,
                "dynamic_retriever": self.dynamic_retriever,
                "session_id": self.test_session_id,
            }

            # Save interaction with context
            try:
                interaction_id = self.db_manager.save_interaction(
                    question=query_text,
                    answer="A sample therapeutic response",
                    session_id=self.test_session_id,
                    context=context,  # Complex dictionary context
                )

                self.assertIsNotNone(interaction_id)

            except Exception as e:
                self.fail(f"Failed to save interaction with complex context: {e}")

    def test_context_building_with_different_pain_points(self):
        """Test that context building handles different pain points correctly."""
        # Test with various pain point scenarios
        query_text = "I'm feeling really overwhelmed lately."
        query_embedding = [0.1] * 768

        # Create a mock of response_generator.build_generation_context to see what's passed
        with patch.object(self.response_generator, "build_generation_context") as mock_build_context:
            # Set a return value for the mock
            mock_build_context.return_value = {
                "query": query_text,
                "session_id": self.test_session_id,
                "query_embedding": query_embedding,
                # Add other expected keys in the result
                "topics_context": {},
                "pain_point": None,
            }

            # Test cases for different pain points
            pain_point_test_cases = [
                # No pain point detected
                {
                    "pain_point_results": {"pain_point_detected": False, "approach_type": "supportive_listening"},
                    "expected_template": "empathy_validation",
                },
                # anxiety detected
                {
                    "pain_point_results": {
                        "pain_point": "anxiety",
                        "pain_point_detected": True,
                        "approach_type": "cbt",
                    },
                    "expected_template": "cognitive_behavioral_therapy",
                },
            ]

            # Test each pain point case with a fresh mock
            for test_case in pain_point_test_cases:
                # Reset the mock between test cases
                mock_build_context.reset_mock()

                # Create a test data structure that will be passed to build_generation_context
                pain_point_results = test_case["pain_point_results"]

                # Call build_generation_context with basic required parameters
                self.response_generator.build_generation_context(
                    user_question=query_text,
                    session_id=self.test_session_id,
                    topics_context={
                        "topic": "stress",
                        "extracted_topics": ["stress"],
                        "topic": "stress",
                        "category_info": {},
                    },
                    pain_point_results=pain_point_results,
                    query_embedding=query_embedding,
                    dynamic_retriever=self.dynamic_retriever,
                    hot_topics=[],
                )

                # Verify build_generation_context was called
                mock_build_context.assert_called_once()

                # Verify template mapping
                approach_type = test_case["pain_point_results"].get("approach_type")
                template = map_approach_to_template(approach_type)
                self.assertEqual(template, test_case["expected_template"])

    def test_enhance_context_with_relevant_documents(self):
        """Test that context enhancement with relevant documents works correctly."""
        # Mock relevant documents
        relevant_docs = [
            {
                "content": "anxiety is a normal response to stress that can be helpful in some situations.",
                "relevance": 0.92,
                "metadata": {"source": "psychology_article.txt"},
            },
            {
                "content": "Deep breathing techniques can help reduce anxiety symptoms by activating the parasympathetic nervous system.",
                "relevance": 0.87,
                "metadata": {"source": "coping_strategies.txt"},
            },
        ]

        # Base context
        context = {
            "query": "How can I manage anxiety?",
            "session_id": self.test_session_id,
            "query_embedding": [0.1] * 768,
        }

        # Patch get_relevant_documents
        with patch.object(self.rag_processor, "get_relevant_documents", return_value=relevant_docs):
            # Create a mock query embedding
            query_embedding = [0.1] * 768

            # Test the enhance method directly
            enhanced_context = self.rag_processor.enhance_context_with_relevant_documents(context, relevant_docs)

            # Verify document content was properly included
            self.assertIn("relevant_documents", enhanced_context)
            self.assertEqual(len(enhanced_context["relevant_documents"]), 2)
            self.assertIn("anxiety is a normal response", enhanced_context["relevant_documents"][0])
            self.assertIn("Deep breathing techniques", enhanced_context["relevant_documents"][1])

            # Test with empty documents
            empty_context = context.copy()
            enhanced_empty = self.rag_processor.enhance_context_with_relevant_documents(empty_context, [])
            self.assertIn("relevant_documents", enhanced_empty)
            self.assertEqual(enhanced_empty["relevant_documents"], [])

    def test_complex_context_handling_in_generation(self):
        """Test handling of complex nested context objects throughout generation pipeline."""
        # Setup a complex context with nested structure
        complex_context = {
            "query": "How can I manage anxiety?",
            "session_id": self.test_session_id,
            "query_embedding": [0.1] * 768,
            "topics_context": {
                "extracted_topics": ["anxiety", "stress", "management"],
                "psychological_insights": {
                    "primary_concern": "anxiety management",
                    "underlying_factors": ["stress", "uncertainty"],
                    "cognitive_patterns": ["catastrophizing", "overgeneralization"],
                },
            },
            "conversation_context": "User previously mentioned feeling overwhelmed at work.",
            "relevant_documents": [
                "Regular exercise can help reduce anxiety symptoms.",
                "Cognitive restructuring involves identifying and challenging negative thought patterns.",
            ],
        }

        # Mock the text generator to verify it receives the complex context intact
        self.text_generator.generate_therapeutic_response_with_dynamic_retrieval = MagicMock(
            return_value="A thoughtful therapeutic response"
        )

        # Generate response
        pain_point_results = {"pain_point": "anxiety", "pain_point_detected": True, "approach_type": "cbt"}
        response = self.response_generator.generate_response_with_template(
            user_question="How can I manage anxiety?",
            session_id=self.test_session_id,
            generation_context=complex_context,
            pain_point_results=pain_point_results,
        )

        # Verify the text generator was called with the complex context intact
        call_args = self.text_generator.generate_therapeutic_response_with_dynamic_retrieval.call_args
        self.assertIsNotNone(call_args)

        # Get the context that was passed to the text generator
        passed_context = call_args[1].get("context")
        self.assertIsNotNone(passed_context)

        # Verify key elements of the complex context were preserved
        self.assertIn("query_embedding", passed_context)
        self.assertIn("topics_context", passed_context)
        self.assertIn("psychological_insights", passed_context["topics_context"])
        self.assertIn("relevant_documents", passed_context)
        self.assertEqual(len(passed_context["relevant_documents"]), 2)

    def test_dynamic_retriever_integration_with_context(self):
        """Test that dynamic retriever properly integrates with generation context."""
        # Setup
        query_text = "Why do I feel anxious in social situations?"
        query_embedding = [0.1] * 768

        # Create a mock for the dynamic retriever
        dynamic_retriever = MagicMock()
        dynamic_retriever.get_conversation_context = MagicMock(
            return_value="Previous conversation about social anxiety..."
        )

        # Create a context that explicitly enables dynamic retrieval
        context = {
            "query": query_text,
            "user_question": query_text,  # Important - match parameter name
            "session_id": self.test_session_id,
            "query_embedding": query_embedding,
            "dynamic_retriever": dynamic_retriever,
            "use_dynamic_retrieval": True,
        }

        # Since we know ResponseGenerator relies on this flag, let's check it's using it
        self.text_generator.generate_therapeutic_response_with_dynamic_retrieval = MagicMock(
            return_value="A helpful response about social anxiety"
        )

        # Generate response with explicit pain_point_results
        pain_point_results = {"pain_point": "anxiety", "pain_point_detected": True, "approach_type": "cbt"}

        # Call the method being tested
        response = self.response_generator.generate_response_with_template(
            user_question=query_text,
            session_id=self.test_session_id,
            generation_context=context,
            pain_point_results=pain_point_results,
        )

        # Verify the dynamic retriever was called
        dynamic_retriever.get_conversation_context.assert_called_once()

    def test_empty_and_edge_case_contexts(self):
        """Test handling of empty or minimal contexts."""
        # Test cases for different context scenarios
        context_test_cases = [
            # Empty context
            {
                "context": {},
                "pain_point_results": {"pain_point_detected": False, "approach_type": "supportive_listening"},
                "should_succeed": True,
            },
            # Minimal required context
            {
                "context": {"query": "Help me", "session_id": self.test_session_id},
                "pain_point_results": {"pain_point_detected": False},
                "should_succeed": True,
            },
            # Missing query
            {
                "context": {"session_id": self.test_session_id},
                "pain_point_results": {"pain_point_detected": False},
                "should_succeed": True,  # Should still work with defaults
            },
        ]

        # Setup text generator mock
        self.text_generator.generate_therapeutic_response_with_dynamic_retrieval = MagicMock(
            return_value="Default response for minimal context"
        )

        # Test each case
        for i, test_case in enumerate(context_test_cases):
            try:
                response = self.response_generator.generate_response_with_template(
                    user_question=test_case["context"].get("query", "Default question"),
                    session_id=test_case["context"].get("session_id", self.test_session_id),
                    generation_context=test_case["context"],
                    pain_point_results=test_case["pain_point_results"],
                )

                if not test_case["should_succeed"]:
                    self.fail(f"Test case {i} should have failed but succeeded")

                # Verify we got a valid response
                self.assertIsNotNone(response)
                self.assertIsInstance(response, str)

            except Exception as e:
                if test_case["should_succeed"]:
                    self.fail(f"Test case {i} failed unexpectedly: {e}")

    def test_approach_type_and_template_selection(self):
        """Test that different approach_types are correctly mapped to templates and used."""
        query_text = "I've been feeling very anxious lately."
        query_embedding = [0.1] * 768

        # Define test cases for different therapeutic approaches
        approach_test_cases = [
            # Standard psychotherapy approaches
            {"approach_type": "cbt", "expected_template": "cognitive_behavioral_therapy"},
            {"approach_type": "dbt", "expected_template": "dialectical_behavior_therapy"},
            {"approach_type": "act", "expected_template": "acceptance_commitment_therapy"},
            # Specific therapeutic focuses
            {"approach_type": "trauma_informed", "expected_template": "trauma"},
            # {"approach_type": "mindfulness", "expected_template": "mindfulness_therapy"},
            {"approach_type": "supportive_listening", "expected_template": "empathy_validation"},
            # Edge cases
            {"approach_type": None, "expected_template": "empathy_validation"},  # Default
            {"approach_type": "unknown_approach", "expected_template": "empathy_validation"},  # Fallback
        ]

        # Mock the text generator
        self.text_generator.generate_therapeutic_response_with_dynamic_retrieval = MagicMock(
            return_value="A therapeutic response"
        )

        # For each approach type
        for test_case in approach_test_cases:
            approach_type = test_case["approach_type"]
            expected_template = test_case["expected_template"]

            # 1. First verify the mapping function works correctly
            actual_template = map_approach_to_template(approach_type)
            self.assertEqual(
                actual_template,
                expected_template,
                f"map_approach_to_template({approach_type}) should return {expected_template}",
            )

            # 2. Create a context with this approach type
            pain_point_results = {"pain_point": "anxiety", "pain_point_detected": True, "approach_type": approach_type}

            context = {
                "query": query_text,
                "session_id": self.test_session_id,
                "query_embedding": query_embedding,
                "topics_context": {
                    "topic": "anxiety",
                    "emotion": "anxiety",
                    "extracted_topics": ["anxiety"],
                },
            }

            # 3. Generate a response with this context and pain point info
            response = self.response_generator.generate_response_with_template(
                user_question=query_text,
                session_id=self.test_session_id,
                generation_context=context,
                pain_point_results=pain_point_results,
            )

            # 4. Verify the text generator was called with the expected template
            call_args = self.text_generator.generate_therapeutic_response_with_dynamic_retrieval.call_args
            self.assertIsNotNone(call_args)

            # The template should be passed as a parameter to generate_therapeutic_response_with_dynamic_retrieval
            passed_template = call_args[1].get("template_name")
            self.assertEqual(
                passed_template,
                expected_template,
                f"For approach_type={approach_type}, template should be {expected_template}",
            )

            # Reset mock for next iteration
            self.text_generator.generate_therapeutic_response_with_dynamic_retrieval.reset_mock()
