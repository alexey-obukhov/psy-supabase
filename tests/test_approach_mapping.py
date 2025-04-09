import unittest
from unittest.mock import patch, MagicMock
from tests.helpers.database_test_base import DatabaseTestBase
from psy_supabase.utilities.utils_mapping import map_approach_to_template


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
        self.assertEqual(map_approach_to_template("CBT"), "Cognitive Behavioral Therapy (CBT)")
        self.assertEqual(map_approach_to_template("Behavioral_Activation"), "Depression")
        self.assertEqual(map_approach_to_template("Interpersonal_Therapy"), "Relationship Issues")
        self.assertEqual(map_approach_to_template("Self_Compassion"), "Empathy and Validation")
        self.assertEqual(map_approach_to_template("Trauma_Informed"), "Trauma")
        self.assertEqual(map_approach_to_template("Connection_Building"), "Others")
        self.assertEqual(map_approach_to_template("Grief_Processing"), "Grief and Loss")
        self.assertEqual(map_approach_to_template("Supportive_Listening"), "Empathy and Validation")

    def test_common_variations(self):
        """Test common variations that might appear in approach types."""
        self.assertEqual(map_approach_to_template("Anxiety"), "Anxiety")
        self.assertEqual(map_approach_to_template("Depression"), "Depression")
        self.assertEqual(map_approach_to_template("Mindfulness"), "Mindfulness and Relaxation")
        self.assertEqual(map_approach_to_template("Crisis"), "Crisis Support")
        self.assertEqual(map_approach_to_template("Workplace"), "Workplace Trauma")
        self.assertEqual(map_approach_to_template("Heartbreak"), "Heartbreak")
        self.assertEqual(map_approach_to_template("Information"), "Information")
        self.assertEqual(map_approach_to_template("Acceptance"), "Acceptance and Commitment Therapy (ACT)")
        self.assertEqual(map_approach_to_template("DBT"), "Dialectical Behavior Therapy (DBT)")
        self.assertEqual(map_approach_to_template("SFBT"), "Solution-Focused Brief Therapy (SFBT)")
        self.assertEqual(map_approach_to_template("Motivational"), "Motivational Interviewing")

    def test_null_or_empty_values(self):
        """Test behavior with null or empty approach types."""
        self.assertEqual(map_approach_to_template(None), "Empathy and Validation")
        self.assertEqual(map_approach_to_template(""), "Empathy and Validation")

    def test_partial_matches(self):
        """Test partial matching behavior."""
        # These should match based on the partial matching logic
        self.assertEqual(map_approach_to_template("cbt_approach"), "Cognitive Behavioral Therapy (CBT)")
        self.assertEqual(map_approach_to_template("mindfulness_approach"), "Mindfulness and Relaxation")
        self.assertEqual(map_approach_to_template("acceptance_approach"), "Acceptance and Commitment Therapy (ACT)")
        self.assertEqual(map_approach_to_template("dbt_approach"), "Dialectical Behavior Therapy (DBT)")

    def test_fallback_for_unknown_approaches(self):
        """Test that unknown approaches fall back to the default template."""
        self.assertEqual(map_approach_to_template("nonexistent_approach"), "Empathy and Validation")
        self.assertEqual(map_approach_to_template("random_text"), "Empathy and Validation")

    def test_mock_objects(self):
        """Test handling of mock objects."""
        mock_approach = MagicMock()
        self.assertEqual(map_approach_to_template(mock_approach), "dynamic_rag_therapy")

    def test_case_insensitivity(self):
        """Test that matching is case-insensitive."""
        self.assertEqual(map_approach_to_template("CBT"), "Cognitive Behavioral Therapy (CBT)")
        self.assertEqual(map_approach_to_template("cbt"), "Cognitive Behavioral Therapy (CBT)")
        self.assertEqual(map_approach_to_template("Cbt"), "Cognitive Behavioral Therapy (CBT)")

    def test_integration_with_pain_point_results(self):
        """Test mapping approaches from typical pain_point_results dictionary."""
        pain_point_results = {
            "pain_point": "anxiety",
            "pain_point_detected": True,
            "approach_type": "CBT"
        }
        template = map_approach_to_template(pain_point_results.get("approach_type"))
        self.assertEqual(template, "Cognitive Behavioral Therapy (CBT)")

        # Test with common approach_type format from the codebase
        pain_point_results = {
            "pain_point": "depression",
            "pain_point_detected": True,
            "approach_type": "Behavioral_Activation"
        }
        template = map_approach_to_template(pain_point_results.get("approach_type"))
        self.assertEqual(template, "Depression")


if __name__ == "__main__":
    unittest.main()