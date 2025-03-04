"""
Logging configuration for psy_supabase.
"""
import logging

def configure_logging(level=logging.INFO):
    """
    Configure logging levels for the application and its dependencies.
    
    Args:
        level: The logging level for the application (default: INFO)
    """
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set levels for noisy third-party libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('hpack').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('h2').setLevel(logging.WARNING)
    
    # Supabase-related libraries
    logging.getLogger('supabase').setLevel(logging.INFO)
    logging.getLogger('postgrest').setLevel(logging.WARNING)
    logging.getLogger('gotrue').setLevel(logging.WARNING)
    logging.getLogger('realtime').setLevel(logging.WARNING)
    logging.getLogger('storage3').setLevel(logging.WARNING)
    
    # ML-related libraries
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('spacy').setLevel(logging.INFO)
