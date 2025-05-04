import unittest
from unittest.mock import MagicMock, patch

from psy_supabase.utilities.utils_mapping import map_approach_to_template
from tests.helpers.database_test_base import DatabaseTestBase


class TestApproachMapping(DatabaseTestBase):
    """
    Test suite for the approach to template mapping functionality.

    This ensures that different therapeutic approaches are correctly
    mapped to their corresponding template names.
    """

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        # No additional setup needed for these tests

    def tearDown(self):
        """Clean up resources."""
        super().tearDown()

    def test_direct_matches(self):
        """Test direct matches from approach type to template."""
        self.assertEqual(map_approach_to_template("cbt"), "cognitive_behavioral_therapy")
        self.assertEqual(map_approach_to_template("behavioral_activation"), "depression")
        self.assertEqual(map_approach_to_template("interpersonal_therapy"), "relationship_issues")
        self.assertEqual(map_approach_to_template("self-compassion"), "empathy_validation")
        self.assertEqual(map_approach_to_template("trauma"), "trauma")
        self.assertEqual(map_approach_to_template("connection_building"), "loneliness")
        self.assertEqual(map_approach_to_template("grief_processing"), "grief_loss")
        self.assertEqual(map_approach_to_template("supportive_listening"), "empathy_validation")

    def test_common_variations(self):
        """Test common variations that might appear in approach types."""
        self.assertEqual(map_approach_to_template("anxiety"), "anxiety")
        self.assertEqual(map_approach_to_template("depression"), "depression")
        self.assertEqual(map_approach_to_template("mindfulness"), "mindfulness_relaxation")
        self.assertEqual(map_approach_to_template("crisis"), "crisis_support")
        self.assertEqual(map_approach_to_template("workplace"), "workplace_trauma")
        self.assertEqual(map_approach_to_template("heartbreak"), "heartbreak")
        self.assertEqual(map_approach_to_template("information"), "information")
        self.assertEqual(map_approach_to_template("acceptance"), "acceptance_commitment_therapy")
        self.assertEqual(map_approach_to_template("dbt"), "dialectical_behavior_therapy")
        self.assertEqual(map_approach_to_template("sfbt"), "solution_focused_brief_therapy")
        self.assertEqual(map_approach_to_template("motivational"), "motivational_interviewing")

    def test_null_or_empty_values(self):
        """Test behavior with null or empty approach types."""
        self.assertEqual(map_approach_to_template(None), "empathy_validation")
        self.assertEqual(map_approach_to_template(""), "empathy_validation")

    def test_partial_matches(self):
        """Test partial matching behavior."""
        # These should match based on the partial matching logic
        self.assertEqual(map_approach_to_template("cbt"), "cognitive_behavioral_therapy")
        self.assertEqual(map_approach_to_template("mindfulness"), "mindfulness_relaxation")
        self.assertEqual(map_approach_to_template("acceptance"), "acceptance_commitment_therapy")
        self.assertEqual(map_approach_to_template("dbt"), "dialectical_behavior_therapy")

    def test_fallback_for_unknown_approaches(self):
        """Test that unknown approaches fall back to the default template."""
        self.assertEqual(map_approach_to_template("nonexistent_approach"), "empathy_validation")
        self.assertEqual(map_approach_to_template("random_text"), "empathy_validation")

    def test_case_insensitivity(self):
        """Test that matching is case-insensitive."""
        self.assertEqual(map_approach_to_template("cbt"), "cognitive_behavioral_therapy")
        self.assertEqual(map_approach_to_template("CBT"), "cognitive_behavioral_therapy")
        self.assertEqual(map_approach_to_template("Cbt"), "cognitive_behavioral_therapy")

    def test_integration_with_pain_point_results(self):
        """Test mapping approaches from typical pain_point_results dictionary."""
        pain_point_results = {"pain_point": "anxiety", "pain_point_detected": True, "approach_type": "cbt"}
        template = map_approach_to_template(pain_point_results.get("approach_type"))
        self.assertEqual(template, "cognitive_behavioral_therapy")

        # Test with common approach_type format from the codebase
        pain_point_results = {
            "pain_point": "depression",
            "pain_point_detected": True,
            "approach_type": "behavioral_activation",
        }
        template = map_approach_to_template(pain_point_results.get("approach_type"))
        self.assertEqual(template, "depression")


if __name__ == "__main__":
    unittest.main()
