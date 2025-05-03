"""Lightweight mock for TextGenerator that doesn't use GPU memory."""


class MockTextGenerator:
    """Lightweight mock for TextGenerator that doesn't use GPU memory."""

    def __init__(self, *args, **kwargs):
        self.device = "cpu"

    def generate_text(self, prompt, max_length=100):
        # Simple deterministic mock response based on prompt
        if "anxiety" in prompt.lower():
            return "Response about managing anxiety..."
        if "depression" in prompt.lower():
            return "Response about depression support..."
        return "General therapeutic response..."
