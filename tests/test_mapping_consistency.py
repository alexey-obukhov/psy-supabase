import json
import os  # Add os import for path joining if needed for template file checks

from psy_supabase.utilities.semantic_emotion_detector import SemanticEmotionDetector
from psy_supabase.utilities.therapeutic_mappings import TherapeuticMappings

# Import your mapping function that USES the centralized TherapeuticMappings.APPROACH_TO_TEMPLATE
from psy_supabase.utilities.utils_mapping import map_approach_to_template

# Import test data for expected values
from test_real_data.test_pain_point_detection_detailed import PainPointDetailedTester


class TestMappingConsistency:
    """Test suite to verify the consistency of mapping dictionaries across the codebase."""

    def test_approach_to_template_consistency(self):
        """Test that approach mappings are consistent across files."""
        # Get approaches from database.py (extracted to a local dict for testing)
        # This db_approaches dict simulates keywords/terms that should map to certain *approaches*
        db_keywords_to_approaches = {
            # Trauma related
            "trauma": "trauma",
            "flashback": "trauma",
            "ptsd": "trauma",
            # Anxiety related
            "anxiety": "cognitive_behavioral",  # This is an *approach*
            "worry": "cognitive_behavioral",
            "stress": "cognitive_behavioral",
            # Depression related
            "depression": "behavioral_activation",  # This is an *approach*
            "sad": "behavioral_activation",
            "hopeless": "behavioral_activation",
            # Relationship related
            "relationship": "interpersonal_therapy",  # This is an *approach*
            "partner": "interpersonal_therapy",
            "breakup": "interpersonal_therapy",
            # Grief related
            "grief": "grief_processing",  # This is an *approach*
            "loss": "grief_processing",
            "death": "grief_processing",
            # Shame related
            "shame": "compassion_focused_therapy",  # This is an *approach*
            "embarrassment": "compassion_focused_therapy",
            "humiliation": "compassion_focused_therapy",
            # Guilt related
            "guilt": "cognitive_behavioral",  # This is an *approach*
            "regret": "cognitive_behavioral",
            "remorse": "cognitive_behavioral",
            # Work related
            "work": "cognitive_behavioral",  # This is an *approach*
            "job": "cognitive_behavioral",
            "career": "cognitive_behavioral",
            "boss": "cognitive_behavioral",
            "dbt": "dbt",
            "act": "act",
            "mindfulness": "mindfulness",
            "crisis": "crisis",
            "workplace": "workplace",
            "stress_management": "stress_management",
            "attachment_based": "attachment_based",
            "supportive_listening": "supportive_listening",
            "sfbt": "sfbt",
            "motivational": "motivational",
            "ocd": "ocd",
            "suicidality": "suicidality",
        }

        missing_templates_info = []
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # For template path

        # Check if each approach maps to a template defined in TherapeuticMappings.APPROACH_TO_TEMPLATE
        # And if that template file exists
        for keyword, approach_name in db_keywords_to_approaches.items():
            # 1. Get the template name that this approach_name *should* map to
            #    according to the centralized TherapeuticMappings.APPROACH_TO_TEMPLATE
            expected_template_name = TherapeuticMappings.APPROACH_TO_TEMPLATE.get(approach_name.lower().strip())

            # 2. Get the template name that the map_approach_to_template function *actually* returns
            #    This function itself uses TherapeuticMappings.get_template_for_approach which uses TherapeuticMappings.APPROACH_TO_TEMPLATE
            actual_template_name_from_func = map_approach_to_template(approach_name)

            if not expected_template_name:
                assert False, (
                    f"Approach '{approach_name}' (from keyword '{keyword}') is not defined in "
                    f"TherapeuticMappings.APPROACH_TO_TEMPLATE. Please add it."
                )

            assert actual_template_name_from_func == expected_template_name, (
                f"Consistency issue for approach '{approach_name}' (from keyword '{keyword}'):\n"
                f"  map_approach_to_template() returned: '{actual_template_name_from_func}'\n"
                f"  Direct lookup in TherapeuticMappings.APPROACH_TO_TEMPLATE expected: '{expected_template_name}'\n"
                f"  This might indicate an issue in map_approach_to_template's logic or its use of get_template_for_approach."
            )

            # 3. Check if the template file for the expected_template_name exists
            template_file_path = os.path.join(project_root, "templates", f"{expected_template_name}.j2")
            if not os.path.exists(template_file_path):
                missing_templates_info.append(
                    f"Approach '{approach_name}': Template file '{template_file_path}' (mapped from '{expected_template_name}')"
                )

        if missing_templates_info:
            assert False, f"Missing template files: {'; '.join(missing_templates_info)}"

    def test_pain_point_to_topic_consistency(self):
        """Test that pain point to topic mappings are consistent with test expectations."""
        tester = PainPointDetailedTester()
        test_cases = tester.load_test_cases()

        # Create mapping dictionary
        CANONICAL_TOPICS = {
            # Trauma-related
            "trauma": "trauma",
            "flashback": "trauma",
            "ptsd": "trauma",
            # Anxiety-related
            "anxiety": "anxiety",
            "worry": "anxiety",
            "stress": "anxiety",
            "health_anxiety": "anxiety",
            # Depression-related
            "depression": "depression",
            "sad": "depression",
            "hopeless": "depression",
            # Grief-related
            "grief": "grief_loss",
            "loss": "grief_loss",
            "death": "grief_loss",
            "bereavement": "grief_loss",
            "mourning": "grief_loss",
            # Relationship-related
            "relationship": "relationship_issues",
            "partner": "relationship_issues",
            "breakup": "relationship_issues",
            "family_conflict": "relationship_issues",
            "jealousy": "relationship_issues",
            "marriage": "relationship_issues",
            "divorce": "relationship_issues",
            "dating": "relationship_issues",
            "interpersonal": "relationship_issues",
            "couple": "relationship_issues",
            "romantic": "relationship_issues",
            # Self-worth related
            "shame": "shame",
            "embarrassment": "shame",
            "humiliation": "shame",
            "self_doubt": "self_compassion",
            "worthlessness": "self_compassion",
            "insecurity": "self_compassion",
            "impostor_syndrome": "self_compassion",
            # Guilt-related
            "guilt": "guilt",
            "regret": "guilt",
            "remorse": "guilt",
            # Work-related
            "work": "workplace_stress",
            "job": "workplace_stress",
            "career": "workplace_stress",
            "boss": "workplace_stress",
            "workplace_trauma": "workplace_stress",
            # Additional specialized mappings
            "ocd": "obsessive_compulsive_disorder",
            "obsession": "obsessive_compulsive_disorder",
            "loneliness": "loneliness",
            "childhood_issues": "trauma",
            "approval_seeking": "self_compassion",
        }

        for case in test_cases:
            if "expected_pain_points" in case and "expected_topic" in case:
                for pain in case["expected_pain_points"]:
                    actual_topic = self._get_topic_from_pain_point(pain)
                    expected_topic = CANONICAL_TOPICS.get(case["expected_topic"], case["expected_topic"])

                    print(f"Pain point: {pain} → Expected topic: {expected_topic} → Actual: {actual_topic}")

                    assert (
                        actual_topic == expected_topic
                    ), f"Topic mismatch for pain '{pain}': expected '{expected_topic}', got '{actual_topic}'"

    def test_topic_to_emotion_consistency(self):
        """Test that topic to emotion mappings are consistent."""
        # Get the actual defined emotion categories (which are the keys of the patterns dictionary)
        defined_emotion_categories = TherapeuticMappings.get_all_patterns().keys()

        missing_emotions_info = []

        for topic_name, topic_data in TherapeuticMappings.THERAPEUTIC_THEMES.items():
            if "emotions" in topic_data:
                for emotion_in_theme in topic_data["emotions"]:
                    if emotion_in_theme not in defined_emotion_categories:
                        missing_emotions_info.append(
                            f"Emotion '{emotion_in_theme}' for topic '{topic_name}' is listed in THERAPEUTIC_THEMES "
                            f"but not found as a defined emotion category in TherapeuticMappings.get_all_patterns()."
                        )

        if missing_emotions_info:
            assert False, f"Emotion consistency errors:\n" + "\n".join(missing_emotions_info)

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

    def test_grief_related_mappings(self):
        """Test specifically grief-related mappings for consistency."""
        grief_keywords = ["grief", "loss", "death", "bereavement", "mourning", "passed away", "deceased"]

        for keyword in grief_keywords:
            topic = self._get_topic_from_pain_point(keyword)
            template = map_approach_to_template(topic)

            print(f"Grief keyword: {keyword} → Topic: {topic} → Template: {template}")

            assert (
                "grief" in topic.lower() or "loss" in topic.lower()
            ), f"Expected grief-related topic for keyword '{keyword}', got '{topic}'"
            assert template == "grief_loss", f"Expected 'grief_loss' template for topic '{topic}', got '{template}'"

    def _get_topic_from_pain_point(self, pain_point: str) -> str:
        """Convert a pain point to a topic using the system's logic."""
        CANONICAL_TOPICS = {
            # Trauma-related
            "trauma": "trauma",
            "flashback": "trauma",
            "ptsd": "trauma",
            "childhood_issues": "trauma",
            "abuse": "trauma",
            # Anxiety-related
            "anxiety": "anxiety",
            "worry": "anxiety",
            "stress": "anxiety",
            "health_anxiety": "anxiety",
            "panic": "anxiety",
            "nervous": "anxiety",
            "overthinking": "anxiety",
            # Depression-related
            "depression": "depression",
            "sad": "depression",
            "hopeless": "depression",
            "unmotivated": "depression",
            "exhausted": "depression",
            # Grief-related
            "grief": "grief_loss",
            "loss": "grief_loss",
            "death": "grief_loss",
            "bereavement": "grief_loss",
            "mourning": "grief_loss",
            "passed away": "grief_loss",
            "deceased": "grief_loss",
            "gone": "grief_loss",
            "passing": "grief_loss",
            "died": "grief_loss",
            "lost someone": "grief_loss",
            "missing someone": "grief_loss",
            "funeral": "grief_loss",
            "memorial": "grief_loss",
            # Relationship-related
            "relationship": "relationship_issues",
            "partner": "relationship_issues",
            "breakup": "relationship_issues",
            "marriage": "relationship_issues",
            "divorce": "relationship_issues",
            "dating": "relationship_issues",
            "couple": "relationship_issues",
            "romantic": "relationship_issues",
            "interpersonal": "relationship_issues",
            "family_conflict": "relationship_issues",
            "jealousy": "relationship_issues",
            # Self-worth related
            "shame": "shame",
            "embarrassment": "shame",
            "humiliation": "shame",
            "self_doubt": "self_compassion",
            "worthlessness": "self_compassion",
            "insecurity": "self_compassion",
            "impostor_syndrome": "self_compassion",
            "not good enough": "self_compassion",
            "approval_seeking": "self_compassion",
            # Work-related (updated)
            "work": "workplace_anxiety",
            "job": "workplace_anxiety",
            "career": "workplace_anxiety",
            "boss": "workplace_anxiety",
            "workplace": "workplace_anxiety",
            "workplace_trauma": "workplace_anxiety",
            "office": "workplace_anxiety",
            "work_stress": "workplace_anxiety",
            "workplace_stress": "workplace_anxiety",
            # Additional specialized mappings
            "ocd": "obsessive_compulsive_disorder",
            "obsession": "obsessive_compulsive_disorder",
            "compulsion": "obsessive_compulsive_disorder",
            "loneliness": "loneliness",
            "alone": "loneliness",
            "isolated": "loneliness",
            # Guilt-related
            "guilt": "guilt",
            "regret": "guilt",
            "remorse": "guilt",
            "mistake": "guilt",
        }

        # Normalize input and check for match
        pain_point_lower = pain_point.lower()
        return CANONICAL_TOPICS.get(pain_point_lower, "general_support")

    def _get_topic_from_keyword(self, keyword: str) -> str:
        """Convert a keyword to a topic using the system's logic."""
        return self._get_topic_from_pain_point(keyword)
