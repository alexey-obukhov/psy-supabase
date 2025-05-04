import json
import os
from typing import Any, Dict, List

import pytest

from psy_supabase.utilities.semantic_emotion_detector import SemanticEmotionDetector
from psy_supabase.utilities.therapeutic_mappings import TherapeuticMappings

# Import your mapping dictionaries
from psy_supabase.utilities.utils_mapping import map_approach_to_template

# Import test data for expected values
from test_real_data.test_pain_point_detection_detailed import PainPointDetailedTester


class TestMappingConsistency:
    """Test suite to verify the consistency of mapping dictionaries across the codebase."""

    def test_approach_to_template_consistency(self):
        """Test that approach mappings are consistent across files."""
        # Get approaches from database.py (extracted to a local dict for testing)
        db_approaches = {
            "anxiety": "cbt",
            "worried": "cbt",
            "relationship": "interpersonal_therapy",
            "alone": "attachment_based_therapy",
            "sad": "behavioral_activation",
            "grief": "grief_processing",
            "failure": "compassion_focused_therapy",
            "trauma": "trauma",
        }

        # Check if each approach in database.py can be mapped in utils_mapping.py
        for keyword, technique in db_approaches.items():
            template = map_approach_to_template(technique)
            assert template is not None, f"Approach '{technique}' from keyword '{keyword}' has no template mapping"

            # Print the mapping for debugging
            print(f"Keyword: {keyword} → Technique: {technique} → Template: {template}")

    def test_pain_point_to_topic_consistency(self):
        """Test that pain point to topic mappings are consistent with test expectations."""
        # Load test expectations
        tester = PainPointDetailedTester()
        test_cases = tester.load_test_cases()

        # Extract expected mappings from test cases
        expected_pain_to_topic = {}
        for case in test_cases:
            if "expected_pain_points" in case and "expected_topic" in case:
                for pain in case["expected_pain_points"]:
                    expected_pain_to_topic[pain] = case["expected_topic"]

        # Now check if our implementation maps these correctly
        for pain, expected_topic in expected_pain_to_topic.items():
            # Get the actual topic our code would produce
            actual_topic = self._get_topic_from_pain_point(pain)

            # Print the mapping for debugging
            print(f"Pain point: {pain} → Expected topic: {expected_topic} → Actual: {actual_topic}")

            # Check for match - including substrings as partial matches
            assert (
                actual_topic == expected_topic or expected_topic in actual_topic or actual_topic in expected_topic
            ), f"Topic mismatch for pain '{pain}': expected '{expected_topic}', got '{actual_topic}'"

    def test_topic_to_emotion_consistency(self):
        """Test that topic to emotion mappings are consistent."""
        # Get emotion mappings from semantic_emotion_detector
        detector = SemanticEmotionDetector()
        emotion_mappings = detector.get_emotion_mappings()

        # Get topic to emotion mappings from therapeutic_mappings
        therapeutic_themes = TherapeuticMappings.THERAPEUTIC_THEMES

        for topic, data in therapeutic_themes.items():
            if "emotions" in data:
                for emotion in data["emotions"]:
                    # Check the emotion is in our emotion mappings
                    assert emotion in emotion_mappings or any(
                        emotion in e for e in emotion_mappings
                    ), f"Emotion '{emotion}' for topic '{topic}' not found in emotion mappings"

    def test_approach_visualization(self):
        """Visualize the complete mapping chain to help identify inconsistencies."""
        results = []

        # Get all the primary mappings
        approaches = {
            "anxiety": "cbt",
            "worried": "cbt",
            "relationship": "interpersonal_therapy",
            "alone": "attachment_based_therapy",
            "sad": "behavioral_activation",
            "grief": "grief_processing",
            "failure": "compassion_focused_therapy",
            "trauma": "trauma",
        }

        # Map through the full chain
        for keyword, technique in approaches.items():
            result = {
                "keyword": keyword,
                "technique": technique,
                "template": map_approach_to_template(technique),
                "topic": self._get_topic_from_keyword(keyword),
                "expected_topic": None,  # Will be filled from test cases
                "expected_approach": None,  # Will be filled from test cases
            }
            results.append(result)

        # Get expected values from test cases
        tester = PainPointDetailedTester()
        test_cases = tester.load_test_cases()

        # Print the complete mapping chain
        print("\nComplete Mapping Chain:")
        print("----------------------")
        for result in results:
            print(
                f"Keyword: {result['keyword']} → "
                + f"Technique: {result['technique']} → "
                + f"Template: {result['template']} → "
                + f"Topic: {result['topic']}"
            )

        # Save to file for easier analysis
        with open("mapping_analysis.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nFull mapping analysis saved to mapping_analysis.json")

    def _get_topic_from_pain_point(self, pain_point: str) -> str:
        """Convert a pain point to a topic using the system's logic."""
        # First check if it's a direct mapping
        therapeutic_themes = TherapeuticMappings.THERAPEUTIC_THEMES

        # Check direct match
        if pain_point in therapeutic_themes:
            return pain_point

        # Check for keyword match
        for topic, data in therapeutic_themes.items():
            if "keywords" in data and pain_point in data["keywords"]:
                return topic

        # Try partial matches
        for topic, data in therapeutic_themes.items():
            if "keywords" in data:
                for keyword in data["keywords"]:
                    if keyword in pain_point or pain_point in keyword:
                        return topic

        # No match found
        return "general_support"  # Default fallback

    def _get_topic_from_keyword(self, keyword: str) -> str:
        """Convert a keyword to a topic using the system's logic."""
        return self._get_topic_from_pain_point(keyword)
