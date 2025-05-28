"""
safety_handler.py

This module implements the `SafetyHandler` class, which is responsible for detecting and responding to potentially harmful content
in user input. It acts as middleware to process queries before they reach the main RAG system, ensuring that harmful or unsafe
content is identified and addressed appropriately.

Key Features:

- **Harmful Content Detection**: Detects suicide-related, self-harm-related, and violence-related content
  using regex patterns. Categorizes harmful content into predefined categories with associated severity levels.

- **Safety Interventions**: Generates tailored responses for each category of harmful content.
  Provides crisis helpline information and encourages users to seek professional help.

- **Metadata Generation**: Creates metadata for detected harmful content, including the category,
  severity, and matched pattern. Facilitates logging and tracking of safety interventions.

Classes:

- `SafetyHandler`: The main class that provides methods for detecting harmful content and generating appropriate responses.

Dependencies:

- `re`: Used for regex-based pattern matching to detect harmful content.
- `prismalog.log.ColoredLogger`: Provides enhanced logging for debugging and monitoring.

Usage:

.. code-block:: python

    from psy_supabase.utilities.safety_handler import SafetyHandler

    # Initialize the safety handler
    safety_handler = SafetyHandler()

    # Process user input
    user_input = "I feel like I want to end my life."
    is_harmful, response, metadata = safety_handler.process_input(user_input)

    if is_harmful:
        print("Harmful content detected!")
        print("Response:", response)
        print("Metadata:", metadata)
    else:
        print("No harmful content detected.")

"""

import re
from typing import Callable, Dict, List, Optional, Tuple, TypedDict

from psy_supabase import get_package_logger

# Set up logging
logger = get_package_logger(__name__)


class SafetyCategoryData(TypedDict):
    patterns: List[str]
    severity: str
    response: Callable[[], str]  # A function that takes no args and returns a string


class SafetyHandler:
    """
    Handles detection and response for potentially harmful content.
    Acts as middleware to process queries before they reach the RAG system.
    """

    def __init__(self) -> None:
        """Initialize SafetyHandler with detection patterns and response templates."""
        self.categories: Dict[str, SafetyCategoryData] = {
            "suicide": {
                "patterns": [
                    r"(?i)(suicide|kill myself|end my life|take my life|don\'t want to live|want to die)",
                    r"(?i)(no reason to live|can\'t go on|better off dead|life is too painful)",
                    r"(?i)(ending it all|my suicide note|planning to end|how to commit suicide)",
                ],
                "severity": "critical",
                "response": self._get_suicide_response,
            },
            "self_harm": {
                "patterns": [
                    r"(?i)(cut myself|hurt myself|self harm|self-harm|injure myself)",
                    r"(?i)(burning myself|hitting myself|starve myself)",
                ],
                "severity": "high",
                "response": self._get_self_harm_response,
            },
            "violence": {
                "patterns": [
                    r"(?i)(kill|murder|hurt|attack|bomb|shoot) (someone|people|them|him|her)",
                    r"(?i)(planning|want|going) to (kill|murder|hurt|attack)",
                ],
                "severity": "high",
                "response": self._get_violence_response,
            },
        }

    def process_input(self, user_input: str) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Process user input to detect harmful content.

        Args:
            user_input: The text input from the user

        Returns:
            Tuple containing:
            - is_harmful: Boolean indicating if harmful content was detected
            - response: Response text if harmful (None otherwise)
            - metadata: Dictionary with detection metadata
        """
        for category, data in self.categories.items():
            for pattern in data["patterns"]:
                if re.search(pattern, user_input):
                    logger.warning("Detected %s content: '%s'", category, user_input)

                    # Get appropriate response
                    response = data["response"]()

                    # Create metadata for logging and tracking
                    metadata = {
                        "detected_category": category,
                        "severity": data["severity"],
                        "matched_pattern": pattern,
                        "response_type": "safety_intervention",
                    }

                    return True, response, metadata

        # No harmful content detected
        return False, None, None

    def _get_suicide_response(self) -> str:
        """Generate a response for suicide-related content."""
        return """I'm deeply concerned about what you're sharing. These thoughts are serious, and I want you to know you're not alone.

Please contact a crisis helpline immediately:
• Samaritans: 116 123 (free, 24/7)
• CALM (Campaign Against Living Miserably): 0800 58 58 58 (5pm-midnight)
• Papyrus HOPELINEUK (for under 35s): 0800 068 4141

These services are available with trained counsellors ready to listen and help. They can provide immediate support and guidance.

If you're in immediate danger, please call NHS emergency services on 999 or go to your nearest A&E department.

Your life matters, and these difficult feelings can improve with the right support. Please reach out for help now."""

    def _get_self_harm_response(self) -> str:
        """Generate a response for self-harm related content."""
        return """I'm concerned about what you're sharing about harming yourself. These feelings are difficult, but support is available.

Please consider these resources:
• Samaritans: 116 123 (free, 24/7)
• MIND helpline: 0300 123 3393 (Mon-Fri, 9am-6pm)
• Self-Injury Support: 0808 800 8088 (Tues-Thurs, 7-9:30pm)

A trained counsellor can help you navigate these feelings and find healthier coping strategies. If you're in immediate danger, please call 999 or go to your nearest A&E department.

You deserve support and care during difficult times."""

    def _get_violence_response(self) -> str:
        """Generate a response for violent content."""
        return """I notice you're expressing thoughts about harming others. These are serious concerns that require proper support and intervention.

If you're having thoughts about harming others:
• Call Samaritans: 116 123 (free, 24/7)
• Contact emergency services: 999 (emergency) or 101 (non-emergency)
• NHS Mental Health Crisis Line: 111, option 2
• Remove yourself from triggering situations if possible

If someone is in immediate danger, please contact emergency services right away.

A mental health professional can help you work through these feelings safely."""
