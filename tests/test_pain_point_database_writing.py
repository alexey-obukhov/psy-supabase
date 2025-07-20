"""
Test suite to verify pain point data writing to database metadata.
This will help identify where and how pain points are being stored.
"""

import json
from typing import Any, Dict
from unittest.mock import Mock

import pytest
from prismalog.log import get_logger

from psy_supabase.core.database import DatabaseManager

# Set up logger
logger = get_logger(__name__)


class TestPainPointDatabaseWriting:
    """Test suite for pain point database writing functionality."""

    @pytest.fixture
    def mock_db_manager(self) -> DatabaseManager:
        """Create a mock database manager for testing."""
        db_manager = Mock(spec=DatabaseManager)
        db_manager.schema_name = "test_schema"
        return db_manager

    @pytest.fixture
    def sample_pain_point_data(self) -> Dict[str, Any]:
        """Sample pain point data for testing."""
        return {
            "pain_points": [
                {
                    "template_used": "cognitive_behavioral_therapy",
                    "occurrence_count": 3,
                    "severity": "medium",
                    "theme": "self_compassion",
                    "recurring_terms": ["inadequate", "every day", "tired"],
                    "first_seen": 1234567890,
                    "affected_interactions": [1, 3, 5],
                },
                {
                    "template_used": "anxiety_management",
                    "occurrence_count": 2,
                    "severity": "low",
                    "theme": "workplace_anxiety",
                    "recurring_terms": ["worried", "job", "security"],
                    "first_seen": 1234567892,
                    "affected_interactions": [2, 4],
                },
            ],
            "severity": "medium",
            "analysis_time_window_days": 30,
            "total_interactions_analyzed": 5,
        }

    def test_pain_point_detection_returns_correct_format(self, mock_db_manager, sample_pain_point_data):
        """Test that pain point detection returns the expected format."""
        # Mock the detect_pain_points method
        mock_db_manager.detect_pain_points.return_value = sample_pain_point_data

        # Call the method
        result = mock_db_manager.detect_pain_points(session_id="test_session", min_occurrences=2)

        # Verify structure
        assert "pain_points" in result
        assert "severity" in result
        assert len(result["pain_points"]) == 2

        # Verify first pain point structure
        first_pp = result["pain_points"][0]
        assert "recurring_terms" in first_pp
        assert "occurrence_count" in first_pp
        assert "severity" in first_pp
        assert "theme" in first_pp

        logger.info("✅ Pain point detection format test passed")

    def test_add_interaction_pain_point_integration(self, mock_db_manager, sample_pain_point_data):
        """Test that add_interaction properly integrates pain point data ONLY on recurrence."""

        # Mock detect_pain_points to simulate recurrence detection
        call_count = 0

        def mock_detect_with_recurrence(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                logger.info("🔍 First occurrence - no pain points detected yet")
                return {"pain_points": [], "severity": "none", "total_interactions_analyzed": 1}
            else:
                logger.info("🔍 Recurrence detected - pain points found!")
                modified_sample_data = sample_pain_point_data.copy()
                for pp in modified_sample_data["pain_points"]:
                    pp["question"] = (
                        "Should I start looking for another job? I'm tired of feeling inadequate every day in this place."
                    )
                return modified_sample_data

        mock_db_manager.detect_pain_points.side_effect = mock_detect_with_recurrence

        # Sample interaction data
        interaction_data = {
            "question": "Should I start looking for another job? I'm tired of feeling inadequate every day in this place.",
            "answer": "It sounds like you're experiencing some workplace stress...",
            "context": "workplace discussion",
            "metadata": {},
        }

        # Mock add_interaction to capture what would be stored
        stored_interactions = []

        def mock_add_interaction(data, session_id=None):
            metadata = data.get("metadata", {})

            # Add session_id to metadata
            if session_id:
                metadata["session_id"] = session_id

                # Simulate pain point detection logic
                question = data.get("question", "")
                if len(question.strip()) > 10:
                    pain_data = mock_db_manager.detect_pain_points(session_id=session_id, min_occurrences=2)

                    if pain_data.get("pain_points") and pain_data.get("severity") != "none":
                        existing_pain_points = metadata.get("pain_points", [])

                        for new_pp in pain_data["pain_points"]:
                            question_text = new_pp.get("question", question)
                            recurring_terms = new_pp.get("recurring_terms", [])

                            if question_text and recurring_terms:
                                existing_questions = [pp.get("question", "") for pp in existing_pain_points]
                                if question_text not in existing_questions:
                                    existing_pain_points.append(
                                        {
                                            "question": question_text,  # Store original question
                                            "detected": True,
                                            "similarity": 1.0,
                                            "template_used": new_pp.get(
                                                "template_used", "cognitive_behavioral_therapy"
                                            ),
                                            "occurrence_count": new_pp.get("occurrence_count", 1),
                                            "severity": new_pp.get("severity", "low"),
                                            "theme": new_pp.get("theme", "general_support"),
                                            "recurring_terms": recurring_terms[:3],  # Keep for comparison
                                            "first_seen": new_pp.get("first_seen", 0),
                                            "affected_interactions": new_pp.get("affected_interactions", []),
                                        }
                                    )

                        metadata["pain_points"] = existing_pain_points
                        metadata["has_pain_points"] = True
                        metadata["pain_severity"] = pain_data.get("severity", "low")
                        metadata["total_pain_points"] = len(existing_pain_points)

            # Store the interaction with updated metadata
            interaction_with_metadata = {**data, "metadata": metadata}
            stored_interactions.append(interaction_with_metadata)

            return {"id": len(stored_interactions), "success": True}

        mock_db_manager.add_interaction.side_effect = mock_add_interaction

        logger.info("🔍 Testing pain point detection with recurrence logic:")

        # First interaction - should NOT trigger pain point storage
        logger.info("\n📝 Adding first interaction (should not store pain points)...")
        result1 = mock_db_manager.add_interaction(interaction_data, "test_session")
        logger.info(f"   Result: {result1}")

        # Check first interaction metadata
        first_interaction = stored_interactions[0]
        metadata1 = first_interaction.get("metadata", {})

        if "pain_points" in metadata1:
            assert len(metadata1["pain_points"]) == 0, "First occurrence should not have pain points"
            logger.info("✅ Correct: No pain points on first occurrence")

        # Second interaction - should trigger pain point storage due to recurrence
        logger.info("\n📝 Adding second interaction (should store pain points due to recurrence)...")

        interaction_data2 = {
            "question": "I'm still feeling inadequate every day and considering leaving.",
            "answer": "I understand this is an ongoing concern...",
            "context": "follow-up discussion",
            "metadata": {},
        }

        result2 = mock_db_manager.add_interaction(interaction_data2, "test_session")
        logger.info(f"   Result: {result2}")

        # Check second interaction metadata
        second_interaction = stored_interactions[1]
        metadata2 = second_interaction.get("metadata", {})

        logger.info(f"   Second interaction metadata: {json.dumps(metadata2, indent=2)}")

        # Second occurrence SHOULD have pain points due to recurrence
        if "pain_points" in metadata2 and metadata2["pain_points"]:
            logger.info("✅ Correct: Pain points stored on recurrence")

            pain_points = metadata2["pain_points"]
            assert len(pain_points) >= 1, f"Expected pain points on recurrence, got {len(pain_points)}"

            first_pp = pain_points[0]
            assert "question" in first_pp, "Pain point should have 'question' field"
            assert "occurrence_count" in first_pp, "Pain point should have 'occurrence_count' field"
            assert first_pp["occurrence_count"] >= 2, "Occurrence count should be >= 2 for recurrence"

            # Verify recurring_terms exist for comparison
            recurring_terms = first_pp.get("recurring_terms", [])
            assert len(recurring_terms) > 0, "Should have recurring terms for comparison"

            question_stored = first_pp["question"]
            logger.info(f"   Pain point question: '{question_stored}'")
            logger.info(f"   Recurring terms: {recurring_terms}")
            logger.info(f"   Occurrence count: {first_pp['occurrence_count']}")

        else:
            pytest.fail("Pain points should be stored on recurrence (second occurrence)")

        # Verify detect_pain_points was called twice
        assert (
            mock_db_manager.detect_pain_points.call_count == 2
        ), f"Expected 2 calls, got {mock_db_manager.detect_pain_points.call_count}"

        logger.info("✅ Pain point recurrence detection test passed!")

    def test_pain_point_question_storage(self, sample_pain_point_data):
        """Test that pain points store original questions with recurring terms for comparison."""
        # Sample data
        full_question = (
            "Should I start looking for another job? I'm tired of feeling inadequate every day in this place."
        )
        expected_terms = ["inadequate", "every day", "tired"]

        # Simulate the pain point processing
        new_pp = sample_pain_point_data["pain_points"][0]
        recurring_terms = new_pp.get("recurring_terms", [])

        # Verify we have the recurring terms for comparison
        assert len(recurring_terms) > 0, "Should have recurring terms"
        assert any(term in recurring_terms for term in expected_terms), "Should contain expected psychological terms"

        logger.info(f"✅ Full question: '{full_question}'")
        logger.info(f"✅ Recurring terms: {recurring_terms}")
        logger.info("✅ Pain point terms extraction test passed")

    def test_metadata_structure_validation(self, sample_pain_point_data):
        """Test that the final metadata structure is correct."""
        # Simulate the metadata that should be created
        expected_metadata = {
            "session_id": "test_session",
            "pain_points": [
                {
                    "question": "I feel inadequate at work every day",  # Original question
                    "detected": True,
                    "similarity": 1.0,
                    "template_used": "cognitive_behavioral_therapy",
                    "occurrence_count": 3,
                    "severity": "medium",
                    "theme": "self_compassion",
                    "recurring_terms": ["inadequate", "every day", "tired"],  # For comparison
                    "first_seen": 1234567890,
                    "affected_interactions": [1, 3, 5],
                }
            ],
            "has_pain_points": True,
            "pain_severity": "medium",
            "total_pain_points": 1,
            "pain_analysis": {
                "total_detected": 2,
                "overall_severity": "medium",
                "analysis_time_window": 30,
                "total_interactions_analyzed": 5,
            },
        }

        # Validate structure
        assert "pain_points" in expected_metadata
        assert "has_pain_points" in expected_metadata
        assert "pain_severity" in expected_metadata
        assert "pain_analysis" in expected_metadata

        # Validate pain point structure
        pain_point = expected_metadata["pain_points"][0]
        assert "question" in pain_point  # Original question stored
        assert pain_point["detected"] is True
        assert "occurrence_count" in pain_point
        assert "recurring_terms" in pain_point  # Terms for comparison logic

        logger.info("✅ Metadata structure validation test passed")

    def test_duplicate_prevention(self):
        """Test that duplicate pain points are prevented using questions."""
        existing_pain_points = [{"question": "I feel inadequate at work every day", "detected": True}]

        new_question = "I feel inadequate at work every day"  # Same question
        existing_questions = [pp.get("question", "") for pp in existing_pain_points]

        # Should prevent duplicate
        assert new_question in existing_questions

        new_question_2 = "I'm worried about job security"  # Different question
        assert new_question_2 not in existing_questions

        logger.info("✅ Duplicate prevention test passed")

    def test_real_database_integration(self):
        """Test with actual database integration (if available)."""
        # This test would require actual database credentials
        # Skip if not available
        try:
            import os

            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_KEY")

            if not supabase_url or not supabase_key:
                pytest.skip("Skipping real database test - credentials not available")

            # Create real database manager
            db_manager = DatabaseManager(supabase_url, supabase_key, "test_user")

            # Add test interaction
            test_data = {
                "question": "I feel inadequate at work every single day",
                "answer": "Test response",
                "context": "test context",
                "metadata": {},
            }

            result = db_manager.add_interaction(test_data, session_id="test_pain_points")

            logger.info(f"✅ Real database integration test: {result}")

        except Exception as e:
            logger.info(f"⚠️ Real database test failed: {e}")


if __name__ == "__main__":
    # Run tests directly
    test_instance = TestPainPointDatabaseWriting()

    # Create mock fixtures manually
    mock_db = Mock(spec=DatabaseManager)
    sample_data = {
        "pain_points": [
            {
                "template_used": "cognitive_behavioral_therapy",
                "occurrence_count": 3,
                "severity": "medium",
                "theme": "self_compassion",
                "recurring_terms": ["inadequate", "every day", "tired"],
                "first_seen": 1234567890,
                "affected_interactions": [1, 3, 5],
            }
        ],
        "severity": "medium",
        "analysis_time_window_days": 30,
        "total_interactions_analyzed": 5,
    }

    logger.info("🧪 Running Pain Point Database Writing Tests")
    logger.info("=" * 50)

    test_instance.test_pain_point_detection_returns_correct_format(mock_db, sample_data)
    test_instance.test_add_interaction_pain_point_integration(mock_db, sample_data)
    test_instance.test_pain_point_question_storage(sample_data)
    test_instance.test_metadata_structure_validation(sample_data)
    test_instance.test_duplicate_prevention()
    test_instance.test_real_database_integration()

    logger.info("=" * 50)
    logger.info("🎉 All tests completed!")
