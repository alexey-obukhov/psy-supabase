import gc
import os
from unittest.mock import patch

import pytest
import torch
from prismalog.log import get_logger

from psy_supabase.core.rag_processor import RAGProcessor
from psy_supabase.core.text_generator import TextGenerator
from psy_supabase.utilities.utils import cleanup_memory
from tests.conftest import SUPPORTIVE_TERMS

# Check if running in GitHub Actions
RUNNING_IN_GITHUB_ACTIONS = os.environ.get("GITHUB_ACTIONS") == "true"

# Check if CUDA is available
CUDA_AVAILABLE = torch.cuda.is_available()
# Determine device to use
DEVICE = "cpu" if RUNNING_IN_GITHUB_ACTIONS else ("cuda" if CUDA_AVAILABLE else "cpu")

logger = get_logger(__name__)


@pytest.fixture(autouse=True)
def cleanup_gpu_memory():
    """
    Automatically clean up GPU memory after each test.
    This fixture runs for all tests without needing to be explicitly requested.
    """
    # Setup: yield to test
    yield

    # Teardown: clean up memory after test completes (or fails)
    logger.info("Cleaning up GPU memory after test...")

    # Call the enhanced cleanup function with force_cuda_cleanup=True
    cleanup_memory(force_cuda_cleanup=True)


@pytest.fixture
def setup_response_generator(mock_db_manager, mock_dynamic_retriever):
    """
    Create a RAGProcessor with enhanced mock objects and ensure cleanup.
    This fixture includes memory cleanup even if the test fails.
    """
    try:
        logger.info("Setting up TextGenerator on %s...", DEVICE)

        # For GitHub Actions, use a fully mocked model
        if RUNNING_IN_GITHUB_ACTIONS:
            # Create a mock model that's good enough for testing
            from unittest.mock import MagicMock

            # Create a fully mocked TextGenerator
            text_generator = MagicMock()
            text_generator.device = "cpu"
            text_generator.generate_text.return_value = "This is a mock response for GitHub Actions testing."
            text_generator.generate_therapeutic_response.return_value = "This is a mock therapeutic response."
            text_generator.is_toxic.return_value = False

            logger.info("Created mock TextGenerator for GitHub Actions")
        else:
            # Use real TextGenerator with appropriate device
            text_generator = TextGenerator(model_name="rasyosef/Phi-1_5-Instruct-v0.1", device=DEVICE, quantize=False)

        # Create the RAG processor with mock dependencies
        rag_processor = RAGProcessor(db_manager=mock_db_manager, generator=text_generator)

        # Add the dynamic retriever mock to ensure query_knowledge works
        rag_processor.dynamic_retriever = mock_dynamic_retriever

        yield rag_processor

    finally:
        # Cleanup code that runs even if the test fails
        logger.info("Cleaning up TextGenerator resources...")

        # Release model resources if possible
        if "text_generator" in locals() and not RUNNING_IN_GITHUB_ACTIONS:
            if hasattr(text_generator, "model"):
                del text_generator.model
            del text_generator

        # Force garbage collection
        gc.collect()


@pytest.mark.slow
class TestResponseQuality:
    """Test suite focused on ensuring high-quality responses."""

    cleanup_memory()

    def test_response_does_not_contain_illustration_paragraph(self, setup_response_generator):
        """Ensure responses don't contain 'Illustration paragraph' pattern."""
        rag_processor = setup_response_generator

        # Mock the generate_text method to return a problematic response
        problematic_response = """Here's a response with helpful content.

Illustration paragraph: This should be removed but leave other content intact.

This part should remain in the final output and provide enough length to pass the test.
This is quality therapeutic content about depression that should be preserved."""

        # Patch the method that generates text
        with patch.object(rag_processor.text_generator, "generate_text", return_value=problematic_response):
            # Generate a response
            response = rag_processor.generate_response(
                "I am feeling sad and don't know what to do please help me", session_id="test_response_quality"
            )

            # Debug output for troubleshooting
            print(f"Original length: {len(problematic_response)}")
            print(f"Cleaned length: {len(response)}")
            print(f"Response: {response}")

            # Check that we got a non-empty response
            assert response, "Response should not be empty"
            assert len(response) > 20, "Response should have meaningful length"

            # Verify problematic patterns are removed
            assert "Illustration paragraph:" not in response
            assert "Here's a response with helpful content." in response
            assert "This part should remain" in response

    def test_response_addresses_depression_content(self, setup_response_generator):
        """Ensure responses address the emotional content of the query."""
        rag_processor = setup_response_generator

        # Mock the generate_text method with a response that will survive cleaning
        mock_response = """I understand you're feeling down. depression can be challenging to navigate, and your feelings are valid.

There are several approaches that might help, including talking to a therapist or exploring self-care activities that bring you joy.

Would you like to discuss some strategies that could help with your feelings of sadness?"""

        with patch.object(rag_processor.text_generator, "generate_text", return_value=mock_response):
            # Use a depression-related question
            response = rag_processor.generate_response(
                "I've been feeling really down lately and can't find motivation", session_id="test_response_quality"
            )

            # Debug output
            print(f"Response length: {len(response)}")
            print(f"Response: {response}")

            # Should contain empathetic/therapeutic language
            therapeutic_phrases = [
                "understand",
                "feel",
                "depression",
                "support",
                "help",
                "therapy",
                "emotion",
                "difficult",
            ]

            # At least some therapeutic content should be present
            matches = [phrase for phrase in therapeutic_phrases if phrase.lower() in response.lower()]
            assert len(matches) >= 1, f"Response should contain therapeutic language. Found: {matches}"

    def test_dynamic_retrieval_works_with_mock(self, mock_dynamic_retriever, setup_response_generator):
        """Test that dynamic retrieval works properly with mock objects."""
        rag_processor = setup_response_generator

        # Make sure the mock is properly set
        rag_processor.dynamic_retriever = mock_dynamic_retriever

        # Verify the retriever can be called without error
        try:
            results = rag_processor.dynamic_retriever.query_knowledge("depression")
            print(f"Retrieved: {results}")

            # Verify we can access content via subscripting
            first_content = results[0]["content"]
            assert "depression" in first_content.lower()
            assert len(results) > 0, "Should return at least one result"

            # Test that template rendering works with this mock
            context = {
                "dynamic_retriever": mock_dynamic_retriever,
                "extracted_topics": ["depression"],
                "use_dynamic_retrieval": True,
            }

            # This should now work without subscripting errors
            result = rag_processor.generate_response("I'm feeling depressed", session_id="test_dynamic")
            assert result, "Should generate a response with dynamic retrieval"

        except Exception as e:
            pytest.fail(f"Dynamic retrieval failed: {e}")

    def test_response_handles_depression_topic(self, setup_response_generator):
        """Test that depression-related queries receive appropriate responses."""
        rag_processor = setup_response_generator

        mock_response = """I understand you're feeling down. depression can be challenging to navigate, and your feelings are valid.

        There are several approaches that might help, including talking to a therapist or exploring self-care activities that bring you joy.

        Would you like to discuss some strategies that could help with your feelings of sadness?"""

        with patch.object(rag_processor.text_generator, "generate_text", return_value=mock_response):
            response = rag_processor.generate_response(
                "I've been feeling really down lately and can't find motivation", session_id="test_depression"
            )

            print(f"depression response: {response}")

            # Verify response quality
            assert "depression" in response.lower(), "Response should mention depression"
            assert len(response) > 100, "Response should have substantial content"
            assert (
                "therapist" in response.lower() or "self-care" in response.lower()
            ), "Response should include coping strategies"

    def test_response_handles_stress_topic(self, setup_response_generator):
        """Test that stress-related queries receive appropriate responses."""
        rag_processor = setup_response_generator

        mock_response = """I understand that you're feeling stressed. It's a common experience, and it's important to acknowledge it.

        Stress can be managed through various techniques such as mindfulness, exercise, and time management strategies.

        Would you like to explore some specific techniques that might help you cope with stress?"""

        with patch.object(rag_processor.text_generator, "generate_text", return_value=mock_response):
            response = rag_processor.generate_response(
                "I'm overwhelmed with work and personal life", session_id="test_stress"
            )

            print(f"Stress response: {response}")

            # Verify response quality
            assert "stress" in response.lower(), "Response should mention stress"
            assert len(response) > 100, "Response should have substantial content"
            assert (
                "mindfulness" in response.lower() or "exercise" in response.lower()
            ), "Response should include coping strategies"

    def test_response_handles_burnout_topic(self, setup_response_generator):
        """Test that burnout-related queries receive appropriate responses."""
        rag_processor = setup_response_generator

        mock_response = """I understand you're feeling burnt out. It's a serious condition that can affect both your mental and physical health.

        Taking breaks, setting boundaries, and seeking support from friends or professionals can be effective ways to combat burnout.

        Would you like to discuss some strategies that might help you recover from burnout?"""

        with patch.object(rag_processor.text_generator, "generate_text", return_value=mock_response):
            response = rag_processor.generate_response(
                "I'm exhausted and can't keep up with everything", session_id="test_burnout"
            )

            print(f"Burnout response: {response}")

            # Verify response quality
            assert "burnout" in response.lower(), "Response should mention burnout"
            assert len(response) > 100, "Response should have substantial content"
            assert (
                "breaks" in response.lower() or "boundaries" in response.lower()
            ), "Response should include coping strategies"

    def test_response_handles_stigma_topic(self, setup_response_generator):
        """Test that stigma-related queries receive appropriate responses."""
        rag_processor = setup_response_generator

        mock_response = """Stigma around mental health is a significant barrier for many. It's important to remember that seeking help is a sign of strength, not weakness.

        Education and open conversations can help reduce stigma. Would you like to discuss how to approach this topic with others?"""

        with patch.object(rag_processor.text_generator, "generate_text", return_value=mock_response):
            response = rag_processor.generate_response(
                "I'm worried about what others will think if I seek help for my mental health", session_id="test_stigma"
            )

            print(f"Stigma response: {response}")

            # Verify response quality
            assert "stigma" in response.lower(), "Response should mention stigma"
            assert len(response) > 100, "Response should have substantial content"
            assert (
                "strength" in response.lower() or "education" in response.lower()
            ), "Response should include coping strategies"

    def test_response_handles_coping_topic(self, setup_response_generator):
        """Test that coping-related queries receive appropriate responses."""
        rag_processor = setup_response_generator

        mock_response = """Coping strategies are essential for managing stress and emotional challenges. Some effective techniques include mindfulness, journaling, and physical activity.

        Would you like to explore some specific coping strategies that might work for you?"""

        with patch.object(rag_processor.text_generator, "generate_text", return_value=mock_response):
            response = rag_processor.generate_response(
                "What are some good ways to cope with stress?", session_id="test_coping"
            )

            print(f"Coping response: {response}")

            # Verify response quality
            assert "coping" in response.lower(), "Response should mention coping"
            assert len(response) > 100, "Response should have substantial content"
            assert (
                "mindfulness" in response.lower() or "journaling" in response.lower()
            ), "Response should include coping strategies"

    def test_response_handles_support_topic(self, setup_response_generator):
        """Test that support-related queries receive appropriate responses."""
        rag_processor = setup_response_generator

        mock_response = """It's great that you're looking for support. Having a strong support system is crucial for mental well-being.

        Would you like to discuss how to build or strengthen your support network?"""

        with patch.object(rag_processor.text_generator, "generate_text", return_value=mock_response):
            response = rag_processor.generate_response(
                "I feel alone and need someone to talk to", session_id="test_support"
            )

            print(f"Support response: {response}")

            # Verify response quality
            assert "support" in response.lower(), "Response should mention support"
            assert len(response) > 100, "Response should have substantial content"
            assert (
                "network" in response.lower() or "system" in response.lower()
            ), "Response should include coping strategies"

    def test_response_handles_therapy_topic(self, setup_response_generator):
        """Test that therapy-related queries receive appropriate responses."""
        rag_processor = setup_response_generator

        mock_response = """Therapy can be a valuable resource for many people. It provides a safe space to explore feelings and develop coping strategies.

        Would you like to discuss different types of therapy or how to find a therapist?"""

        with patch.object(rag_processor.text_generator, "generate_text", return_value=mock_response):
            response = rag_processor.generate_response(
                "I'm considering therapy but don't know where to start", session_id="test_therapy"
            )

            print(f"Therapy response: {response}")

            # Verify response quality
            assert "therapy" in response.lower(), "Response should mention therapy"
            assert len(response) > 100, "Response should have substantial content"
            assert (
                "therapist" in response.lower() or "types of therapy" in response.lower()
            ), "Response should include coping strategies"

    def test_response_handles_mindfulness_topic(self, setup_response_generator):
        """Test that mindfulness-related queries receive appropriate responses."""
        rag_processor = setup_response_generator

        mock_response = """mindfulness is a powerful tool for managing stress and anxiety. It involves being present in the moment and accepting your thoughts and feelings without judgment.

        Would you like to explore some mindfulness exercises or techniques?"""

        with patch.object(rag_processor.text_generator, "generate_text", return_value=mock_response):
            response = rag_processor.generate_response(
                "I'm interested in mindfulness but don't know how to start", session_id="test_mindfulness"
            )

            print(f"mindfulness response: {response}")

            # Verify response quality
            assert "mindfulness" in response.lower(), "Response should mention mindfulness"
            assert len(response) > 100, "Response should have substantial content"
            assert (
                "exercises" in response.lower() or "techniques" in response.lower()
            ), "Response should include coping strategies"

    def test_response_handles_communication_topic(self, setup_response_generator):
        """Test that communication-related queries receive appropriate responses."""
        rag_processor = setup_response_generator

        mock_response = """Effective communication is key to healthy relationships. It involves active listening and expressing your thoughts and feelings clearly.

        Would you like to discuss some strategies for improving communication with others?"""

        with patch.object(rag_processor.text_generator, "generate_text", return_value=mock_response):
            response = rag_processor.generate_response(
                "I struggle to communicate my feelings to others", session_id="test_communication"
            )

            print(f"Communication response: {response}")

            # Verify response quality
            assert "communication" in response.lower(), "Response should mention communication"
            assert len(response) > 100, "Response should have substantial content"
            assert (
                "listening" in response.lower() or "strategies" in response.lower()
            ), "Response should include coping strategies"

    def test_response_handles_addiction_topic(self, setup_response_generator):
        """Test that addiction-related queries receive appropriate responses."""
        rag_processor = setup_response_generator

        mock_response = """Addiction is a complex issue that affects many people. It's important to seek help and support when dealing with addiction.

        Would you like to discuss some resources or strategies for managing addiction?"""

        with patch.object(rag_processor.text_generator, "generate_text", return_value=mock_response):
            response = rag_processor.generate_response(
                "I'm struggling with addiction and need help", session_id="test_addiction"
            )

            print(f"Addiction response: {response}")

            # Verify response quality
            assert "addiction" in response.lower(), "Response should mention addiction"
            assert len(response) > 100, "Response should have substantial content"
            assert (
                "resources" in response.lower() or "strategies" in response.lower()
            ), "Response should include coping strategies"

    def test_response_handles_suicidal_thoughts_topic(self, setup_response_generator):
        """Test that suicidal thoughts-related queries receive appropriate responses."""
        rag_processor = setup_response_generator

        mock_response = """I'm really sorry to hear that you're feeling this way. It's important to talk to someone who can help you, like a mental health professional or a trusted person in your life.

        You are not alone, and there are people who care about you and want to help. Please consider reaching out for support."""

        with patch.object(rag_processor.text_generator, "generate_text", return_value=mock_response):
            response = rag_processor.generate_response(
                "I'm having thoughts of hurting myself", session_id="test_suicidal_thoughts"
            )

            print(f"Suicidal thoughts response: {response}")

            # Verify response quality
            assert "help" in response.lower(), "Response should mention help"
            assert len(response) > 100, "Response should have substantial content"
            assert "not alone" in response.lower(), "Response should include supportive language"
            assert "mental health professional" in response.lower(), "Response should mention professional support"

    def test_response_handles_anger_topic(self, setup_response_generator):
        """Test that anger-related queries receive appropriate responses."""
        rag_processor = setup_response_generator

        mock_response = """I understand that you're feeling angry. Anger is a normal emotion, but it's important to find healthy ways to express and manage it.

        Would you like to discuss some techniques for coping with anger, such as deep breathing or physical activity?"""

        with patch.object(rag_processor.text_generator, "generate_text", return_value=mock_response):
            response = rag_processor.generate_response(
                "I'm really angry and don't know how to deal with it", session_id="test_anger"
            )

            print(f"Anger response: {response}")

            # Verify response quality
            assert "anger" in response.lower(), "Response should mention anger"
            assert len(response) > 100, "Response should have substantial content"
            assert (
                "deep breathing" in response.lower() or "physical activity" in response.lower()
            ), "Response should include coping strategies"

    def test_response_handles_grief_topic(self, setup_response_generator):
        """Test that grief-related queries receive appropriate responses."""
        rag_processor = setup_response_generator

    def test_response_handles_anxiety_topic(self, setup_response_generator):
        """Test that anxiety-related queries receive appropriate responses."""
        rag_processor = setup_response_generator

        mock_response = """I understand you're experiencing anxiety. Many people struggle with anxiety, and it can be overwhelming.

        anxiety often manifests as both physical sensations (like rapid heartbeat or shallow breathing) and racing thoughts about potential threats or dangers.

        Some strategies that may help include deep breathing exercises, grounding techniques, and gradually facing situations that trigger anxiety with proper support."""

        with patch.object(rag_processor.text_generator, "generate_text", return_value=mock_response):
            response = rag_processor.generate_response(
                "I'm constantly anxious and worried about everything", session_id="test_anxiety"
            )

            print(f"anxiety response: {response}")

            # Verify response quality
            assert "anxiety" in response.lower(), "Response should mention anxiety"
            assert len(response) > 100, "Response should have substantial content"
            assert (
                "deep breathing" in response.lower() or "grounding" in response.lower()
            ), "Response should include coping strategies"

    def test_response_handles_trauma_topic(self, setup_response_generator):
        """Test that trauma-related queries receive sensitive responses."""
        rag_processor = setup_response_generator

        mock_response = """I'm truly sorry to hear about the trauma you've experienced. Your feelings are valid, and it took courage to share this.

        Healing from trauma is a personal journey that takes time. Many people find that working with a trauma-informed therapist can be helpful.

        In the meantime, focusing on safety and self-care is important. Would you like to discuss some grounding techniques that might help when difficult memories arise?"""

        with patch.object(rag_processor.text_generator, "generate_text", return_value=mock_response):
            response = rag_processor.generate_response(
                "I experienced a traumatic event and can't stop thinking about it", session_id="test_trauma"
            )

            print(f"Trauma response: {response}")

            # Verify trauma-appropriate response
            assert "sorry" in response.lower() or "understand" in response.lower(), "Response should show empathy"
            assert "trauma" in response.lower(), "Response should acknowledge trauma"
            assert (
                "therapist" in response.lower() or "support" in response.lower()
            ), "Response should mention professional support"

    def test_response_handles_empty_input(self, setup_response_generator):
        """Test that empty inputs are handled gracefully."""
        rag_processor = setup_response_generator

        response = rag_processor.generate_response("", session_id="test_empty")

        print(f"Empty input response: {response}")

        has_supportive_language = any(term in response.lower() for term in SUPPORTIVE_TERMS)
        # Should provide a helpful, non-error response
        assert response, "Response should not be empty"
        assert len(response) > 20, "Response should have meaningful content"
        assert has_supportive_language, "Response should offer support"

    def test_response_handles_invalid_input(self, setup_response_generator):
        """Test that invalid inputs are handled gracefully."""
        rag_processor = setup_response_generator

        response = rag_processor.generate_response("!@#$%^&*()", session_id="test_invalid")

        print(f"Invalid input response: {response}")

        # Should provide a helpful, non-error response
        assert response, "Response should not be empty"
        assert len(response) > 20, "Response should have meaningful content"

        # Check for a broader range of supportive language
        supportive_terms = ["help", "support", "assist", "here for you", "share", "talk", "listen"]
        has_supportive_term = any(term in response.lower() for term in supportive_terms)
        assert has_supportive_term, f"Response should offer support. Found: {response}"

        # Check for any problematic patterns
        assert "error" not in response.lower(), "Response should not contain error messages"
        assert "invalid" not in response.lower(), "Response should not mention invalid input"

    def test_response_handles_long_input(self, setup_response_generator):
        """Test that long inputs are handled gracefully."""
        rag_processor = setup_response_generator

        long_input = "I'm feeling really down lately and can't find motivation. " * 50
        response = rag_processor.generate_response(long_input, session_id="test_long")
        print(f"Long input response: {response}")

        has_supportive_language = any(term in response.lower() for term in SUPPORTIVE_TERMS)
        # Should provide a helpful, non-error response
        assert response, "Response should not be empty"
        assert len(response) > 20, "Response should have meaningful content"
        assert has_supportive_language, "Response should offer support"
        # Check for any problematic patterns
        assert "error" not in response.lower(), "Response should not contain error messages"
        assert "invalid" not in response.lower(), "Response should not mention invalid input"

    def test_response_handles_special_characters(self, setup_response_generator):
        """Test that special characters in input are handled gracefully."""
        rag_processor = setup_response_generator

        special_input = "I'm feeling really down lately! Can you help me with my anxiety? @#$%^&*()"
        response = rag_processor.generate_response(special_input, session_id="test_special")

        print(f"Special characters input response: {response}")

        has_supportive_language = any(term in response.lower() for term in SUPPORTIVE_TERMS)
        # Should provide a helpful, non-error response
        assert response, "Response should not be empty"
        assert len(response) > 20, "Response should have meaningful content"
        assert has_supportive_language, "Response should offer support"
        # Check for any problematic patterns
        assert "error" not in response.lower(), "Response should not contain error messages"
        assert "invalid" not in response.lower(), "Response should not mention invalid input"

    def test_response_handles_multiple_topics(self, setup_response_generator):
        """Test that responses can handle multiple topics in one input."""
        rag_processor = setup_response_generator

        mock_response = """I understand you're feeling overwhelmed with both stress and anxiety. It's common to feel this way, especially when facing multiple challenges.

        Some techniques that may help include deep breathing exercises, mindfulness, and talking to someone about your feelings.

        Would you like to explore some specific strategies for managing both stress and anxiety?"""

        with patch.object(rag_processor.text_generator, "generate_text", return_value=mock_response):
            response = rag_processor.generate_response(
                "I'm feeling really stressed and anxious about everything", session_id="test_multiple_topics"
            )

            print(f"Multiple topics response: {response}")

            # Verify response quality
            assert "stress" in response.lower(), "Response should mention stress"
            assert "anxiety" in response.lower(), "Response should mention anxiety"
            assert len(response) > 100, "Response should have substantial content"
            assert (
                "deep breathing" in response.lower() or "mindfulness" in response.lower()
            ), "Response should include coping strategies"

    def test_response_handles_cultural_sensitivity(self, setup_response_generator):
        """Test that responses are culturally sensitive and appropriate."""
        rag_processor = setup_response_generator

        mock_response = """I understand that cultural differences can impact how we express and experience emotions. It's important to approach these topics with sensitivity and respect.

        Would you like to discuss how cultural factors may influence your feelings or coping strategies?"""

        with patch.object(rag_processor.text_generator, "generate_text", return_value=mock_response):
            response = rag_processor.generate_response(
                "I'm feeling overwhelmed and want to talk about my cultural background",
                session_id="test_cultural_sensitivity",
            )

            print(f"Cultural sensitivity response: {response}")

            # Verify response quality
            assert "cultural" in response.lower(), "Response should mention cultural sensitivity"
            assert len(response) > 100, "Response should have substantial content"
            assert (
                "sensitivity" in response.lower() or "respect" in response.lower()
            ), "Response should include culturally sensitive language"
            # Check for any problematic patterns
            assert "error" not in response.lower(), "Response should not contain error messages"
            assert "invalid" not in response.lower(), "Response should not mention invalid input"

    def test_response_handles_mental_health_resources(self, setup_response_generator):
        """Test that responses provide information about mental health resources."""
        rag_processor = setup_response_generator

        mock_response = """There are many resources available for mental health support. Some options include therapy, support groups, and hotlines.

        Would you like to discuss some specific resources that might be helpful for you?"""

        with patch.object(rag_processor.text_generator, "generate_text", return_value=mock_response):
            response = rag_processor.generate_response(
                "I'm looking for mental health resources", session_id="test_resources"
            )

            print(f"Mental health resources response: {response}")

            # Verify response quality
            assert "resources" in response.lower(), "Response should mention mental health resources"
            assert len(response) > 100, "Response should have substantial content"
            assert (
                "therapy" in response.lower() or "support groups" in response.lower()
            ), "Response should include coping strategies"
            # Check for any problematic patterns
            assert "error" not in response.lower(), "Response should not contain error messages"
            assert "invalid" not in response.lower(), "Response should not mention invalid input"

    def test_response_handles_urgent_support(self, setup_response_generator):
        """Test that responses provide information about urgent support options."""
        rag_processor = setup_response_generator

        mock_response = """If you're in crisis or need immediate support, please reach out to a mental health professional or a crisis hotline.

        Your safety is the most important thing. Would you like help finding resources for urgent support?"""

        with patch.object(rag_processor.text_generator, "generate_text", return_value=mock_response):
            response = rag_processor.generate_response(
                "I'm in crisis and need urgent help", session_id="test_urgent_support"
            )

            print(f"Urgent support response: {response}")

            # Verify response quality
            assert "urgent" in response.lower(), "Response should mention urgent support"
            assert len(response) > 100, "Response should have substantial content"
            assert (
                "crisis" in response.lower() or "hotline" in response.lower()
            ), "Response should include coping strategies"
            # Check for any problematic patterns
            assert "error" not in response.lower(), "Response should not contain error messages"
            assert "invalid" not in response.lower(), "Response should not mention invalid input"
