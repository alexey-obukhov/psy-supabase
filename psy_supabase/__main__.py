#!/usr/bin/env python3
"""
Entry point for the psy-supabase package.
"""
import os
import sys
from dotenv import load_dotenv

def main():
    """Run the application."""
    # Load environment variables
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        env_path = os.path.join(os.getcwd(), '.env')
        if os.path.exists(env_path):
            load_dotenv(env_path)
    
    try:
        from . import main as main_module
        
        # If main has a run function, use it
        if hasattr(main_module, 'run'):
            main_module.run()
        # Otherwise, look for app and run it
        elif hasattr(main_module, 'app'):
            main_module.app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5008)))
        else:
            print("Error: Could not find a run() function or app object in main.py")
            sys.exit(1)
    except ImportError as e:
        print(f"Error: Could not import main module: {e}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Python path: {sys.path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error running application: {e}")
        sys.exit(1)

# The __name__ == "__main__" check is needed for when running as a module
if __name__ == "__main__":
    main()
