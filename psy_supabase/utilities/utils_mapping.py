"""
Mapping Utilities
=================

This module provides centralized mapping functions used across the psy_supabase package.
It standardizes naming conventions between different parts of the system:

1. Pain point themes to therapeutic approach types
2. Approach types to template names
3. Themes to human-readable names
4. Emotion categories to response strategies

By centralizing these mappings, we ensure consistency across the entire application.
"""

from typing import Dict, Optional

from prismalog.log import get_logger

from psy_supabase.config import DEFAULT_TOPIC
from psy_supabase.utilities.therapeutic_mappings import TherapeuticMappings

logger = get_logger(__name__)


def map_approach_to_template(approach: Optional[str] = None) -> str:
    """
    Map therapeutic approach to template filename.

    Args:
        approach: The therapeutic approach name

    Returns:
        Template name (without .j2 extension)
    """
    # Default to empathy_validation if no approach is provided
    approach = approach.lower() if approach else "empathy_validation"

    # Core mapping of approaches to templates
    APPROACH_TO_TEMPLATE: Dict[str, str] = {
        # Primary approaches
        "cbt": "cognitive_behavioral_therapy",
        "dbt": "dialectical_behavior_therapy",
        "act": "acceptance_commitment_therapy",
        "self-compassion": "empathy_validation",
        "interpersonal_therapy": "relationship_issues",
        "trauma": "trauma",
        "grief_processing": "grief_loss",
        "behavioral_activation": "depression",
        "supportive_listening": "empathy_validation",
        "connection_building": "loneliness",
        # Map themes directly (for backward compatibility)
        "abandonment": "attachment_based_therapy",
        "rejection": "empathy_validation",
        "crisis": "crisis_support",
        "control": "cognitive_behavioral_therapy",
        "anxiety": "anxiety",
        "depression": "depression",
        "motivational": "motivational_interviewing",
        "workplace": "workplace_trauma",
        "information": "information",
        "acceptance": "acceptance_commitment_therapy",
        "sfbt": "solution_focused_brief_therapy",
        "heartbreak": "heartbreak",
        "relationship": "relationship_issues",
        "mindfulness": "mindfulness_relaxation",
        "self-worth": "empathy_validation",
        "trauma": "trauma",
        "loneliness": "loneliness",
        "grief": "grief_loss",
        "trust": "relationship_issues",
        "emotional_regulation": "dialectical_behavior_therapy",
        "guilt": "guilt",
        "shame": "shame",
    }

    # Check direct mapping first
    if approach in APPROACH_TO_TEMPLATE:
        return APPROACH_TO_TEMPLATE[approach]

    # If not found, try to identify theme from keyword
    for theme, data in TherapeuticMappings.THERAPEUTIC_THEMES.items():
        keywords = data.get("keywords", [])
        if approach in keywords:
            # Get approach from theme
            theme_approach = data.get("approach", "supportive_listening")
            # Map approach to template
            return APPROACH_TO_TEMPLATE.get(theme_approach, "empathy_validation")

    # Default fallback
    return "empathy_validation"


def get_template_for_theme(theme: str) -> str:
    """Helper function to get template for a specific theme"""
    # Get approach from theme
    approach = TherapeuticMappings.get_approach_for_theme(theme)
    # Map approach to template
    return map_approach_to_template(approach)


def get_template_for_keyword(keyword: str) -> str:
    """Helper function to map a keyword directly to a template"""
    # First find which theme this keyword belongs to
    for theme, data in TherapeuticMappings.THERAPEUTIC_THEMES.items():
        if keyword in data.get("keywords", []):
            return get_template_for_theme(theme)

    # Default fallback
    return DEFAULT_TOPIC


def map_theme_to_approach_type(theme: str) -> str:
    """Map a theme to its corresponding therapeutic approach type."""
    return TherapeuticMappings.get_approach_for_theme(theme)


# def map_approach_to_template(approach_type: str) -> str:
#     """Map an approach type to its corresponding template name."""
#     return TherapeuticMappings.get_template_for_approach(approach_type)


def map_approach_name(theme: str) -> str:
    """Map a theme to its human-readable name."""
    return TherapeuticMappings.get_human_readable_name(theme)


def map_emotion_to_response_strategy(emotion: str) -> Dict:
    """
    Maps an emotion to appropriate response strategies.

    Args:
        emotion: The detected emotion (e.g., 'anxiety', 'sadness')

    Returns:
        Dict with response strategies
    """
    emotion = emotion.lower() if emotion else "neutral"

    emotion_strategies = {
        "anxiety": {
            "primary_approach": "grounding",
            "tone": "calm",
            "response_length": "moderate",
            "template_preference": "anxiety",
        },
        "sadness": {
            "primary_approach": "validation",
            "tone": "warm",
            "response_length": "moderate",
            "template_preference": "depression",
        },
        "anger": {
            "primary_approach": "reflection",
            "tone": "neutral",
            "response_length": "brief",
            "template_preference": "empathy_validation",
        },
        "confusion": {
            "primary_approach": "clarification",
            "tone": "clear",
            "response_length": "moderate",
            "template_preference": "information",
        },
        "hope": {
            "primary_approach": "reinforcement",
            "tone": "encouraging",
            "response_length": "moderate",
            "template_preference": "solution_focused_brief_therapy",
        },
        "loneliness": {
            "primary_approach": "connection",
            "tone": "warm",
            "response_length": "moderate",
            "template_preference": "others",
        },
        "shame": {
            "primary_approach": "self-compassion",
            "tone": "gentle",
            "response_length": "moderate",
            "template_preference": "self-disclosure",
        },
        "neutral": {
            "primary_approach": "exploration",
            "tone": "curious",
            "response_length": "moderate",
            "template_preference": "Question",
        },
    }

    for key, strategy in emotion_strategies.items():
        if key in emotion:
            return strategy

    # Default to neutral
    return emotion_strategies["neutral"]


# def get_therapeutic_approach_for_pain_point(pain_point_data: Dict) -> Dict:
#     """
#     Get therapeutic approach recommendation based on pain point data.

#     Args:
#         pain_point_data: Pain point detection data

#     Returns:
#         Dict with approach information
#     """
#     # Default approach if no pain point detected
#     if not pain_point_data or not pain_point_data.get("detected"):
#         return {
#             "name": "general_support",
#             "approach_type": "supportive_listening",
#             "template": "empathy_validation",
#         }

#     # Get primary theme from recurring terms if available
#     recurring_terms = []
#     for pp in pain_point_data.get("pain_points", []):
#         if pp.get("recurring_terms"):
#             recurring_terms.extend(pp.get("recurring_terms"))

#     # Use the first recurring term as primary theme
#     primary_theme = recurring_terms[0] if recurring_terms else "unknown"

#     # Map theme to approach type
#     approach_type = map_theme_to_approach_type(primary_theme)

#     # Map approach type to template
#     template = map_approach_to_template(approach_type)

#     # Get human-readable name
#     name = map_approach_name(primary_theme)

#     return {"name": name, "approach_type": approach_type, "template": template}
