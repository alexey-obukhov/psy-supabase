"""
Therapeutic Response Mapping System
===================================

This module provides a comprehensive mapping system that forms the core of the therapeutic
response generation pipeline. It manages the conversion between different representations
of therapeutic concepts across the system.

Key Components
--------------
1. Approach to Template Mapping
   - Maps therapeutic approaches (e.g., "cognitive_behavioral") to template names
   - Handles specialized cases like grief counseling and workplace anxiety
   - Provides fallback to empathy_validation when specific mapping isn't found

2. Theme to Template Pipeline
   - Keywords → Themes → Approaches → Templates
   - Example: "feeling sad" → depression theme → behavioral_activation → depression template

3. Emotion to Response Strategy Mapping
   - Maps detected emotions to appropriate therapeutic responses
   - Controls tone, length, and approach based on emotional content
   - Provides default strategies for unknown emotions

Usage Flow
----------
1. User Input Processing:
   ```python
   keyword = "feeling sad"
   template = get_template_for_keyword(keyword)  # Returns "depression"
   ```

2. Theme-based Response:
   ```python
   theme = "grief_loss"
   template = get_template_for_theme(theme)  # Returns "grief_loss"
   ```

3. Emotion-Aware Response:
   ```python
   emotion = "anxiety"
   strategy = map_emotion_to_response_strategy(emotion)
   # Returns {
   #     "primary_approach": "grounding",
   #     "tone": "calm",
   #     "response_length": "moderate",
   #     "template_preference": "anxiety"
   # }
   ```

Template Hierarchy
------------------
1. Core Templates:
   - trauma.j2
   - cognitive_behavioral_therapy.j2
   - depression.j2
   - relationship_issues.j2

2. Specialized Templates:
   - grief_loss.j2
   - workplace_anxiety.j2
   - self_compassion.j2

3. Fallback Templates:
   - empathy_validation.j2
   - general_support.j2

Integration Points
------------------
1. RAG Processor:
   - Uses these mappings to select appropriate response templates
   - Combines with vector similarity for theme detection

2. Response Generator:
   - Uses template mappings for final response formatting
   - Applies emotion-based strategies to responses

3. Pain Point Detector:
   - Maps detected issues to therapeutic approaches
   - Uses theme mappings for categorization

Template Selection Logic
------------------------
1. Direct Mapping:
   approach → APPROACH_TO_TEMPLATE → template

2. Theme-based Mapping:
   theme → TherapeuticMappings.get_approach_for_theme() → approach → template

3. Keyword-based Mapping:
   keyword → theme → approach → template

4. Fallback Logic:
   unknown input → "empathy_validation"

Example Pipeline
----------------
User: "I'm feeling really sad about losing my job"
1. Keywords detected: ["sad", "losing", "job"]
2. Themes identified: ["depression", "loss", "workplace_anxiety"]
3. Primary approach selected: "behavioral_activation"
4. Template chosen: "depression.j2"
5. Emotion strategy: map_emotion_to_response_strategy("sadness")
"""

from typing import Dict, List, Optional, Union

from prismalog.log import get_logger

from psy_supabase.config import DEFAULT_TOPIC
from psy_supabase.utilities.therapeutic_mappings import TherapeuticMappings

logger = get_logger(__name__)


def map_approach_to_template(approach: Optional[Union[str, List[str]]] = None) -> str:
    """
    Maps a therapeutic approach (or a list of approaches) to its corresponding template name
    by utilizing TherapeuticMappings.get_template_for_approach.
    Defaults to "empathy_validation" if no specific mapping is found or no approach is given.
    """
    if not approach:
        return "empathy_validation"

    # Normalize input
    if isinstance(approach, str):
        logger.debug("Mapping single string approach: '%s'", approach)
        normalized: str = approach.lower().strip().replace("  ", " ")  # normalized is str
        logger.debug("Mapping approach: '%s' → normalized: '%s'", approach, normalized)

        # Direct lookup
        template = TherapeuticMappings.APPROACH_TO_TEMPLATE.get(normalized)
        if template:
            logger.debug(f"Direct match found: '{template}'")
            return template

        # Try matching without special characters
        normalized_clean = "".join(c for c in normalized if c.isalnum())
        for key, value in TherapeuticMappings.APPROACH_TO_TEMPLATE.items():
            key_clean = "".join(c for c in key if c.isalnum())
            if key_clean == normalized_clean:
                logger.debug(f"Clean match found: '{key}' → '{value}'")
                return value

    elif isinstance(approach, list):
        logger.debug("Mapping list of approaches: '%s'", approach)
        normalized_list = [a.lower().strip().replace("  ", " ") for a in approach]
        logger.debug("Mapping approaches: %s → normalized: %s", approach, normalized_list)
        for item_approach in normalized_list:
            template = TherapeuticMappings.APPROACH_TO_TEMPLATE.get(item_approach)
            if template:
                # Found a template for one of the approaches in the list
                logger.debug("Match found in list for '%s': '%s'", item_approach, template)
                return template

    logger.debug("No match found, using default: 'empathy_validation'")
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


def map_theme_to_approach_type(theme: str) -> Union[str, List[str]]:
    """Map a theme to its corresponding therapeutic approach type(s)."""
    normalized_theme = theme.lower().strip()
    approach = TherapeuticMappings.get_approach_for_theme(normalized_theme)  # This can return str or List[str]
    logger.debug(f"map_theme_to_approach_type: Theme '{normalized_theme}' mapped to approach(es) '{approach}'.")
    return approach


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
        "fear": {
            "primary_approach": "safety",
            "tone": "calm",
            "response_length": "moderate",
            "template_preference": "trauma",
        },
        "helplessness": {
            "primary_approach": "empowerment",
            "tone": "supportive",
            "response_length": "moderate",
            "template_preference": "trauma",
        },
        "emptiness": {
            "primary_approach": "validation",
            "tone": "warm",
            "response_length": "moderate",
            "template_preference": "grief_loss",
        },
        "longing": {
            "primary_approach": "connection",
            "tone": "gentle",
            "response_length": "moderate",
            "template_preference": "grief_loss",
        },
    }

    for key, strategy in emotion_strategies.items():
        if key in emotion:
            return strategy

    # Default to neutral
    return emotion_strategies["neutral"]
