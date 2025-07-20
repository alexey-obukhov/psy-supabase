"""
2025, Dresden Alexey Obukhov, alexey.obukhov@hotmail.com

PSY Supabase package initialization.
Configures logging for the entire package.
"""

import logging
import os
from typing import Union

from prismalog.config import LoggingConfig
from prismalog.log import ColoredLogger, get_logger

# Initialize logging when package is imported
if not LoggingConfig.is_initialized():
    config_path = os.path.join(os.path.dirname(__file__), "config_logging.yaml")
    if os.path.exists(config_path):
        LoggingConfig.initialize(config_file=config_path)

# Export commonly used items
__all__ = ["get_logger"]


def get_package_logger(name: str) -> Union[ColoredLogger, logging.Logger]:
    """Get a logger for this package."""
    return get_logger(name)
