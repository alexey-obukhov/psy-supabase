import pytest
import torch
import logging
import traceback
from unittest.mock import Mock, patch, MagicMock, mock_open
from jinja2 import Template

from psy_supabase.core.text_generator import TextGenerator
from tests.conftest import mock_model, mock_tokenizer, text_generator
from tests.conftest import create_text_generator_mocks, setup_text_generator_for_testing

# Configure logging for this test file
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestTemplateDebugging:
    """
    Specialized test class to diagnose template rendering issues in TextGenerator.
    """

    def test_identify_iteration_point(self, text_generator):
        """Test to identify exactly where the iteration happens."""
        # Create a special dict to track access
        class AccessTrackingDict(dict):
            def __init__(self, *args, **kwargs):
                self.access_log = []
                super().__init__(*args, **kwargs)

            def __getitem__(self, key):
                self.access_log.append(f"Accessed key: {key}")
                return super().__getitem__(key)

        # Create a context with tracking
        context = AccessTrackingDict({
            'user_question': "Test question for iteration tracking"
        })

        # Create a special mock template that logs render calls
        class DebugTemplate:
            def __init__(self):
                self.render_called = False

            def render(self, **kwargs):
                self.render_called = True
                logger.debug(f"Template.render() called with kwargs: {list(kwargs.keys())}")
                return "Rendered template content"

        debug_template = DebugTemplate()

        try:
            # Patch _load_template to return our debug template
            with patch.object(text_generator, '_load_template', return_value=debug_template):
                # Patch generate_text to return a simple string
                with patch.object(text_generator, 'generate_text', return_value="Generated response"):
                    # Run the method
                    result = text_generator.generate_therapeutic_response(
                        user_question="Test question for iteration tracking",
                        template_name="debug_template",
                        context=context
                    )

                    # If we get here without error, show successful flow
                    logger.info(f"Success! Result: {result}")
                    logger.info(f"Context accesses: {context.access_log}")
                    logger.info(f"Template render called: {debug_template.render_called}")
        except Exception as e:
            # Capture detailed info about the error
            logger.error(f"Exception caught: {str(e)}")
            logger.error(traceback.format_exc())

    def test_debug_prompt_splitting(self, text_generator):
        """Test specifically focused on the prompt splitting code."""
        # Mock a template that returns a string
        mock_template = Mock()
        mock_template.render.return_value = "First part\n\nUSER'S CURRENT MESSAGE: testing\n\nLast part"

        # Track where the failure happens
        checkpoint = {"value": 0}

        # Test the specific code that might fail
        try:
            with patch.object(text_generator, '_load_template', return_value=mock_template):
                # Call generate_therapeutic_response
                checkpoint["value"] = 1
                result = text_generator.generate_therapeutic_response(
                    "testing",
                    "test_template",
                    {'user_question': "testing"}
                )
                checkpoint["value"] = 2

                logger.info(f"Success! Result: {result}")
        except Exception as e:
            logger.error(f"Failure at checkpoint {checkpoint['value']}: {str(e)}")
            logger.error(traceback.format_exc())

    def test_isolate_mock_issues(self, text_generator):
        """Test to identify issues with mock objects specifically."""
        # Create real string outputs for mocks
        mock_template = Mock()
        mock_template.render.return_value = "Template content with USER'S CURRENT MESSAGE: test"

        # Create a special mock for tokenizer.encode that prints its argument
        def mock_encode(text, *args, **kwargs):
            logger.debug(f"tokenizer.encode called with text: {repr(text)[:100]}...")
            # Check if text is actually a Mock
            if isinstance(text, Mock):
                logger.error("tokenizer.encode received a Mock object instead of a string!")
            return torch.tensor(list(range(1, 11)))  # Return 10 tokens

        # Apply our mocks
        with patch.object(text_generator, '_load_template', return_value=mock_template):
            # Replace tokenizer.encode with our debug version
            text_generator.tokenizer.encode = mock_encode

            try:
                result = text_generator.generate_therapeutic_response(
                    "test question",
                    "test_template",
                    {'user_question': "test question"}
                )
                logger.info(f"Success! Result: {result}")
            except Exception as e:
                logger.error(f"Exception: {str(e)}")
                logger.error(traceback.format_exc())

    def test_bypass_template_rendering(self, text_generator):
        """Test that completely bypasses template rendering to isolate the issue."""
        # Make a version of the function that skips template rendering
        original_load_template = text_generator._load_template

        class BypassTemplate:
            def render(self, **kwargs):
                return "Pre-rendered template content"

        def bypass_load_template(name):
            logger.debug(f"Bypassing template loading for: {name}")
            return BypassTemplate()

        # Apply patch
        text_generator._load_template = bypass_load_template

        try:
            result = text_generator.generate_therapeutic_response(
                "test question",
                "test_template",
                {'user_question': "test question"}
            )
            logger.info(f"Success with bypassed template! Result: {result}")
        except Exception as e:
            logger.error(f"Exception with bypassed template: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            # Restore original method
            text_generator._load_template = original_load_template

    def test_reproduce_issue_step_by_step(self, text_generator):
        """Test that reproduces the issue step by step to pinpoint exact failure."""
        # 1. Create a mock template
        mock_template = Mock()
        mock_template.render.return_value = "Rendered template content"

        # 2. Intercept each step
        steps = []

        def step_logger(step_name):
            steps.append(step_name)
            logger.debug(f"Executing step: {step_name}")

        try:
            with patch.object(text_generator, '_load_template', return_value=mock_template):
                # Detailed step-by-step execution
                step_logger("Start")

                user_question = "Test question"
                context = {'user_question': user_question}
                template_name = "test_template"

                step_logger("Loading template")
                template = text_generator._load_template(template_name)

                step_logger("Rendering template")
                prompt = template.render(**context)

                step_logger("Encoding tokens")
                token_count = len(text_generator.tokenizer.encode(prompt))

                step_logger("Token count check")
                logger.debug(f"Token count: {token_count}")

                step_logger("Generating text")
                response = text_generator.generate_text(prompt)

                step_logger("Complete")
                logger.info(f"Success! Response: {response}")

        except Exception as e:
            logger.error(f"Failed at step: {steps[-1]}")
            logger.error(f"Exception: {str(e)}")
            logger.error(traceback.format_exc())

    def test_real_template_with_mock_functions(self, text_generator):
        """Test using a real Jinja2 template but with mock functions."""
        # Create a real template
        template_str = """
        <|system|>
        You are a helpful assistant responding to a user with emotion: {{psychological_context.emotion}}
        </|system|>

        <|user|>
        {{user_question}}
        </|user|>

        <|assistant|>
        """

        real_template = Template(template_str)

        try:
            # Create context with all required fields
            context = {
                'user_question': "How are you today?",
                'psychological_context': {
                    'emotion': 'happy',
                    'topic': 'greeting',
                    'confidence': 0.9
                }
            }

            # Manually render template
            rendered = real_template.render(**context)
            logger.info(f"Template rendered successfully: {rendered}")

            # Now try the whole function with a patched template
            with patch.object(text_generator, '_load_template', return_value=real_template):
                with patch.object(text_generator, 'generate_text', return_value="I'm doing well, thank you!"):
                    result = text_generator.generate_therapeutic_response(
                        "How are you today?",
                        "test_template",
                        context
                    )
                    logger.info(f"Function result: {result}")

        except Exception as e:
            logger.error(f"Template test failed: {str(e)}")
            logger.error(traceback.format_exc())

    def test_inspect_template_loading_code(self, text_generator):
        """Test focused on the template loading process."""
        # Inspect _load_template method
        template_dir = getattr(text_generator, 'template_dir', 'unknown')
        logger.info(f"Template directory: {template_dir}")

        # Mock the filesystem operations
        mock_template_content = """
        <|system|>
        You are a helpful assistant responding to a user with emotion: {{psychological_context.emotion}}
        </|system|>

        <|user|>
        {{user_question}}
        </|user|>

        <|assistant|>
        """

        with patch('os.path.join', return_value='/mock/path/to/template.j2'):
            with patch('os.path.exists', return_value=True):
                with patch('builtins.open', mock_open(read_data=mock_template_content)):
                    try:
                        # Try to load a template directly
                        template = text_generator._load_template('any_template')
                        logger.info(f"Template loaded: {type(template)}")

                        # Try rendering it
                        rendered = template.render(user_question="test",
                                                psychological_context={"emotion": "happy"})
                        logger.info(f"Template rendered successfully: {rendered[:50]}...")

                    except Exception as e:
                        logger.error(f"Template loading failed: {str(e)}")
                        logger.error(traceback.format_exc())

    def test_pinpoint_template_render_issue(self, text_generator):
        """Test to pinpoint exactly where template rendering is failing."""
        logger.info("Starting targeted template rendering test")

        # CRITICAL: The issue is in how the template.render() method is mocked
        # Let's create a robust mock that actually returns a renderable string
        mock_template = Mock()

        # This is CRITICAL: mock_template.render() must return a string, not a Mock!
        mock_template.render.return_value = "This is an actual string that can be iterated"

        # Track operation stages
        stages = []

        try:
            stages.append("Setup mocks")

            # Mock the _load_template to return our proper mock
            with patch.object(text_generator, '_load_template', return_value=mock_template):
                stages.append("After template mock")

                # Mock generate_text to simply return a string
                with patch.object(text_generator, 'generate_text', return_value="Generated response"):
                    stages.append("After generate_text mock")

                    # Create minimal context
                    context = {'user_question': "Test question"}

                    # Now instrument each step of the function that might fail
                    original_encode = text_generator.tokenizer.encode

                    def instrumented_encode(text, *args, **kwargs):
                        # THIS IS THE KEY PROBLEM: text might be a Mock instead of a string
                        logger.info(f"tokenizer.encode called with type: {type(text)}")
                        if isinstance(text, Mock):
                            logger.error("FOUND THE BUG: encode received a Mock instead of string!")
                            # Return something valid to continue
                            return torch.tensor([1, 2, 3, 4, 5])
                        return original_encode(text, *args, **kwargs)

                    # Replace with instrumented version
                    text_generator.tokenizer.encode = instrumented_encode

                    # Now execute
                    stages.append("Before function call")
                    result = text_generator.generate_therapeutic_response(
                        "Test question",
                        "test_template",
                        context
                    )
                    stages.append("After function call")

                    logger.info(f"Result: {result}")
                    # Restore original
                    text_generator.tokenizer.encode = original_encode

        except Exception as e:
            logger.error(f"Failed at stage: {stages[-1]}")
            logger.error(f"Exception: {str(e)}")
            logger.error(traceback.format_exc())

            # Check for the specific iteration error
            if "'Mock' object is not iterable" in str(e):
                logger.error("ITERATION ERROR CONFIRMED: This is our target bug")

    def test_fix_template_implementation(self, text_generator):
        """Test a potential fix for the template rendering issues."""
        logger.info("Testing template rendering fix")

        # Implement the fix directly in this test
        # 1. Create a wrapper class that ensures strings are returned
        class SafeTemplateMock:
            def __init__(self):
                self.calls = []

            def render(self, **kwargs):
                self.calls.append(kwargs)
                # Always return a real string
                return f"Rendered template with keys: {list(kwargs.keys())}"

        # Create our safe template
        safe_template = SafeTemplateMock()

        # Set up the test
        with patch.object(text_generator, '_load_template', return_value=safe_template):
            with patch.object(text_generator, 'generate_text', return_value="Generated response"):
                # Use a complete context
                context = {
                    'user_question': "Test question",
                    'psychological_context': {
                        'emotion': 'curious',
                        'topic': 'testing',
                        'confidence': 0.9
                    }
                }

                # Call the method
                result = text_generator.generate_therapeutic_response(
                    "Test question",
                    "test_template",
                    context
                )

                # Check success
                logger.info(f"SUCCESS! Result: {result}")
                logger.info(f"Template was rendered with context keys: {list(safe_template.calls[0].keys())}")

    def test_example_with_clean_setup(self, mock_model, mock_tokenizer):
        """Test example with clean TextGenerator setup."""
        # Create a fresh TextGenerator with our helper
        with patch('psy_supabase.core.text_generator.AutoModelForCausalLM.from_pretrained',
                  return_value=mock_model), \
             patch('psy_supabase.core.text_generator.AutoTokenizer.from_pretrained',
                   return_value=mock_tokenizer):

            fresh_generator = TextGenerator(
                model_name="test-model",
                device="cpu"
            )

            # Apply our testing setup
            setup_text_generator_for_testing(fresh_generator)
