# filepath: /home/vertok/git-projects/psy_supabase/tests/helpers/model_mocks.py

class MockTextGenerator:
    """Lightweight mock for TextGenerator that doesn't use GPU memory."""

    def __init__(self, *args, **kwargs):
        self.device = "cpu"

    def generate_text(self, prompt, max_length=100):
        # Simple deterministic mock response based on prompt
        if "anxiety" in prompt.lower():
            return "Response about managing anxiety..."
        elif "depression" in prompt.lower():
            return "Response about depression support..."
        else:
            return "General therapeutic response..."
