import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from psy_supabase.core.database import DatabaseManager
from psy_supabase.core.pain_point_detector import PainPointDetector


class TestPainPointDetection(unittest.TestCase):
    """Test suite for pain point detection functionality."""

    def setUp(self):
        """Set up test fixtures using existing patterns from conftest.py."""
        from tests.conftest import TEST_KEY, TEST_SCHEMA, TEST_URL, TEST_USER_ID

        # Create mock Supabase client (following test_database.py pattern)
        self.mock_supabase = Mock()

        # Create DatabaseManager with mock
        self.db_manager = DatabaseManager.__new__(DatabaseManager)
        self.db_manager.supabase_url = TEST_URL
        self.db_manager.supabase_key = TEST_KEY
        self.db_manager.user_id = TEST_USER_ID
        self.db_manager.schema_name = TEST_SCHEMA
        self.db_manager.supabase = self.mock_supabase

        # Create PainPointDetector instance
        self.pain_point_detector = PainPointDetector(self.db_manager)

        self.session_id = "test_session_123"

        self.mock_embedding = [0.1] * 384
        self.db_manager.create_embedding = MagicMock(return_value=self.mock_embedding)

    def test_detect_pain_points_no_history(self):
        """Test pain point detection when no conversation history exists."""
        # Mock empty conversation history
        self.db_manager.get_conversation_history = MagicMock(return_value=[])

        result = self.pain_point_detector.detect_pain_points(self.session_id)

        expected = {"pain_points": [], "severity": "none", "first_detected_at": None}
        self.assertEqual(result, expected)

    def test_detect_pain_points_insufficient_data(self):
        """Test pain point detection with insufficient data."""
        mock_history = [{"question": "I feel anxious", "created_at": "2024-01-01"}]
        self.db_manager.get_conversation_history = MagicMock(return_value=mock_history)

        result = self.pain_point_detector.detect_pain_points(self.session_id, min_occurrences=2)

        expected = {"pain_points": [], "severity": "none", "first_detected_at": None}
        self.assertEqual(result, expected)

    def test_semantic_chunk_question_basic(self):
        """Test basic semantic chunking functionality."""
        question = "I feel anxious about work. My sleep is affected. I can't concentrate."

        chunks = self.pain_point_detector._semantic_chunk_question(question)

        # Should break into meaningful chunks or at least preserve content
        self.assertGreaterEqual(len(chunks), 1)

        # Verify content is preserved
        combined = " ".join(chunks)
        self.assertIn("anxious", combined)
        self.assertIn("sleep", combined)

    def test_semantic_chunk_question_single_sentence(self):
        """Test semantic chunking with single sentence."""
        question = "I feel very anxious"

        chunks = self.pain_point_detector._semantic_chunk_question(question)

        # Should return chunks that preserve the content (may be more than 1 due to phrase extraction)
        self.assertGreaterEqual(len(chunks), 1)

        # Verify the original content is preserved
        combined = " ".join(chunks)
        self.assertIn("anxious", combined)
        self.assertIn("feel", combined)

    def test_semantic_chunk_question_empty(self):
        """Test semantic chunking with empty input."""

        chunks = self.pain_point_detector._semantic_chunk_question("")

        # Should handle empty input gracefully
        self.assertTrue(len(chunks) >= 1)

    def test_detect_pain_points_with_real_data(self):
        """Test pain point detection with realistic conversation data."""
        mock_history = [
            {"question": "I feel anxious about work deadlines", "created_at": "2024-01-01"},
            {"question": "Work stress is affecting my sleep", "created_at": "2024-01-02"},
            {"question": "I'm worried about my job performance", "created_at": "2024-01-03"},
            {"question": "My anxiety is getting worse at work", "created_at": "2024-01-04"},
        ]

        self.db_manager.get_conversation_history = MagicMock(return_value=mock_history)

        result = self.pain_point_detector.detect_pain_points(self.session_id, threshold=0.3, min_occurrences=2)

        # Should detect some structure
        self.assertIsInstance(result, dict)
        self.assertIn("pain_points", result)
        self.assertIn("severity", result)
        self.assertIn("first_detected_at", result)

    @patch("psy_supabase.core.pain_point_detector.logger")
    def test_detect_pain_points_error_handling(self, mock_logger):
        """Test error handling in pain point detection."""
        # Mock get_conversation_history to raise an exception
        self.db_manager.get_conversation_history = MagicMock(side_effect=Exception("Database error"))

        result = self.pain_point_detector.detect_pain_points(self.session_id)

        # Should return safe default values on error
        expected = {"pain_points": [], "severity": "none", "first_detected_at": None}
        self.assertEqual(result, expected)

    def test_semantic_chunk_question_max_length(self):
        """Test semantic chunking with long questions."""
        long_question = "I feel very anxious about work and deadlines and my performance and everything else that is happening in my life right now."

        chunks = self.pain_point_detector._semantic_chunk_question(long_question)

        # Should return valid chunks
        self.assertGreaterEqual(len(chunks), 1)

        # Content should be preserved
        combined = " ".join(chunks)
        self.assertIn("anxious", combined)
        self.assertIn("work", combined)
        self.assertIn("deadlines", combined)

        # Test that chunking handles long text reasonably
        # (Each chunk should be meaningful, not just the original text)
        if len(chunks) > 1:
            # If multiple chunks, each should contain meaningful content
            for chunk in chunks:
                self.assertGreater(len(chunk.strip()), 0)
        else:
            # If single chunk, should be the original or a processed version
            self.assertIn("anxious", combined)

    def test_pain_point_detector_initialization(self):
        """Test that PainPointDetector initializes correctly."""

        self.assertIsInstance(self.pain_point_detector, PainPointDetector)
        self.assertEqual(self.pain_point_detector.db_manager, self.db_manager)

        # Test that config is loaded
        self.assertIsNotNone(self.pain_point_detector.config)
        self.assertIsNotNone(self.pain_point_detector.chunking_config)

    def test_pain_point_detector_config_access(self):
        """Test that PainPointDetector can access configuration."""

        config = self.pain_point_detector.config
        chunking_config = self.pain_point_detector.chunking_config

        self.assertIsInstance(config, dict)
        self.assertIsInstance(chunking_config, dict)

        # Test expected config keys
        self.assertIn("similarity_threshold", config)
        self.assertIn("min_occurrences", config)
        self.assertIn("include_full_question", chunking_config)


# === PYTEST-STYLE TESTS USING PROPER MOCKING ===


@pytest.fixture
def configured_pain_point_detector():
    """Create a properly configured PainPointDetector for testing."""
    from tests.conftest import TEST_KEY, TEST_SCHEMA, TEST_URL, TEST_USER_ID

    # Create mock Supabase client
    mock_supabase = Mock()

    # Create DatabaseManager with mock
    db_manager = DatabaseManager.__new__(DatabaseManager)
    db_manager.supabase_url = TEST_URL
    db_manager.supabase_key = TEST_KEY
    db_manager.user_id = TEST_USER_ID
    db_manager.schema_name = TEST_SCHEMA
    db_manager.supabase = mock_supabase

    # Mock existing methods
    db_manager.create_embedding = MagicMock(return_value=[0.1] * 384)
    db_manager.get_conversation_history = MagicMock(return_value=[])

    pain_point_detector = PainPointDetector(db_manager)

    return pain_point_detector


def test_pain_point_detection_basic_functionality(configured_pain_point_detector):
    """Test basic pain point detection functionality."""

    detector = configured_pain_point_detector

    # Verify configuration is set correctly
    from tests.conftest import TEST_KEY, TEST_URL

    assert detector.db_manager.supabase_url == TEST_URL
    assert detector.db_manager.supabase_key == TEST_KEY

    # Mock conversation history
    mock_history = [
        {"question": "I feel anxious", "created_at": "2024-01-01"},
        {"question": "Still anxious today", "created_at": "2024-01-02"},
    ]

    detector.db_manager.get_conversation_history = MagicMock(return_value=mock_history)

    result = detector.detect_pain_points("test_session")

    assert isinstance(result, dict)
    assert "pain_points" in result
    assert "severity" in result
    assert "first_detected_at" in result


def test_semantic_chunking_preserves_content(configured_pain_point_detector):
    """Test that semantic chunking preserves all content."""

    detector = configured_pain_point_detector

    from tests.conftest import TEST_KEY, TEST_URL

    assert detector.db_manager.supabase_url == TEST_URL
    assert detector.db_manager.supabase_key == TEST_KEY

    original_text = "I feel anxious about work. My sleep is affected by stress. I can't concentrate during meetings."

    chunks = detector._semantic_chunk_question(original_text)

    # All original content should be preserved when chunks are combined
    combined = " ".join(chunks)
    key_words = ["anxious", "work", "sleep", "stress", "concentrate", "meetings"]

    for word in key_words:
        assert word in combined


def test_pain_point_detection_empty_input(configured_pain_point_detector):
    """Test pain point detection with various empty inputs."""

    detector = configured_pain_point_detector

    from tests.conftest import TEST_KEY, TEST_URL

    assert detector.db_manager.supabase_url == TEST_URL
    assert detector.db_manager.supabase_key == TEST_KEY

    # Test with empty history
    detector.db_manager.get_conversation_history = MagicMock(return_value=[])
    result = detector.detect_pain_points("test_session")

    expected = {"pain_points": [], "severity": "none", "first_detected_at": None}
    assert result == expected

    # Test with history containing empty questions
    mock_history = [{"question": "", "created_at": "2024-01-01"}, {"question": "   ", "created_at": "2024-01-02"}]
    detector.db_manager.get_conversation_history = MagicMock(return_value=mock_history)
    result = detector.detect_pain_points("test_session")

    assert result == expected


def test_pain_point_detection_threshold_parameter(configured_pain_point_detector):
    """Test pain point detection with different threshold values."""

    detector = configured_pain_point_detector

    from tests.conftest import TEST_KEY, TEST_URL

    assert detector.db_manager.supabase_url == TEST_URL
    assert detector.db_manager.supabase_key == TEST_KEY

    mock_history = [
        {"question": "I feel anxious about work", "created_at": "2024-01-01"},
        {"question": "Work makes me nervous", "created_at": "2024-01-02"},
        {"question": "Job stress is overwhelming", "created_at": "2024-01-03"},
    ]

    detector.db_manager.get_conversation_history = MagicMock(return_value=mock_history)

    # Test with different thresholds
    result_high = detector.detect_pain_points("test_session", threshold=0.9)
    result_low = detector.detect_pain_points("test_session", threshold=0.1)

    # Both should return valid results
    assert isinstance(result_high, dict)
    assert isinstance(result_low, dict)


def test_pain_point_detection_min_occurrences_parameter(configured_pain_point_detector):
    """Test pain point detection with different min_occurrences values."""

    detector = configured_pain_point_detector

    from tests.conftest import TEST_KEY, TEST_URL

    assert detector.db_manager.supabase_url == TEST_URL
    assert detector.db_manager.supabase_key == TEST_KEY

    mock_history = [
        {"question": "I feel anxious", "created_at": "2024-01-01"},
        {"question": "Still anxious", "created_at": "2024-01-02"},
    ]

    detector.db_manager.get_conversation_history = MagicMock(return_value=mock_history)

    # Should find pain points with min_occurrences=2
    result_min2 = detector.detect_pain_points("test_session", min_occurrences=2)

    # Should not find pain points with min_occurrences=5
    result_min5 = detector.detect_pain_points("test_session", min_occurrences=5)

    assert isinstance(result_min2, dict)
    assert isinstance(result_min5, dict)
    assert result_min5 == {"pain_points": [], "severity": "none", "first_detected_at": None}


def test_pain_point_detector_configuration_loading(configured_pain_point_detector):
    """Test that PainPointDetector loads configuration correctly."""

    detector = configured_pain_point_detector

    # Test that config is loaded
    assert hasattr(detector, "config")
    assert hasattr(detector, "chunking_config")

    # Test config structure
    assert isinstance(detector.config, dict)
    assert isinstance(detector.chunking_config, dict)

    # Test expected keys
    assert "similarity_threshold" in detector.config
    assert "min_occurrences" in detector.config
    assert "include_full_question" in detector.chunking_config


if __name__ == "__main__":
    # Run both unittest and pytest
    unittest.main(exit=False)
    pytest.main([__file__, "-v"])
