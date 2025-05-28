#!/usr/bin/env python3
"""
Entry point for the psy-supabase package.
"""
import os

from psy_supabase import get_package_logger

logger = get_package_logger(__name__)


def main() -> None:
    """Run the application."""
    logger.info("Application entry point executing...")  # Example log
    try:
        # Import the main application logic *after* logging is configured
        from . import main as main_module

        logger.debug("Imported main application module.")

        # If main has a run function, use it
        if hasattr(main_module, "run"):
            logger.info("Found run() function in main module. Executing...")
            main_module.run()
        # Otherwise, look for app and run it
        elif hasattr(main_module, "app"):
            logger.info("Found app object in main module. Running app...")
            # Consider getting host/port from config instead of env vars directly here
            host = os.environ.get("HOST", "0.0.0.0")
            port = int(os.environ.get("PORT", 5008))
            logger.info(f"Running Flask/FastAPI app on {host}:{port}")
            main_module.app.run(host=host, port=port)
        else:
            logger.critical("Could not find a run() function or app object in main.py")
    except ImportError as e:
        logger.critical(f"Could not import main module: {e}", exc_info=True)
    except Exception as e:
        logger.critical(f"Error running application: {e}", exc_info=True)


# The __name__ == "__main__" check is needed for when running as a module
if __name__ == "__main__":
    main()
