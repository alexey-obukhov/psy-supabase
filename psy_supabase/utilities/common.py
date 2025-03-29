"""
Common utility functions shared across modules.
This module should NOT import from model_manager or text_generator.
"""

import os
import logging
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)

def get_project_root():
    """Return the absolute path to the project root directory."""
    return Path(__file__).parent.parent.parent.absolute()

def ensure_dir_exists(directory_path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        logger.info(f"Created directory: {directory_path}")
    return directory_path

def get_models_dir():
    """Get the models directory path."""
    root_dir = get_project_root()
    models_dir = os.path.join(root_dir, "models")
    return ensure_dir_exists(models_dir)

def is_github_actions():
    """Check if running in GitHub Actions environment."""
    return os.environ.get('GITHUB_ACTIONS') == 'true'

def is_test_environment():
    """Detect if we're running in a test environment."""
    import sys
    return any('pytest' in arg for arg in sys.argv) or any('test_' in arg for arg in sys.argv)
