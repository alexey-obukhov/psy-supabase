#!/usr/bin/env python3
import subprocess
import sys


def run_script(script_path: str) -> int:
    """Run a python script and return its exit code"""
    print(f"Running: {script_path}")
    result = subprocess.run([sys.executable, script_path], check=False)
    print(f"Finished: {script_path} with exit code: {result.returncode}")
    return result.returncode


def main() -> None:
    """Main function to run multiple scripts in order"""
    # List of scripts to run in order
    scripts = [
        "/home/vertok/git-projects/psy_supabase/test_real_data/test_pain_point_detection_cousine_only.py",
        "/home/vertok/git-projects/psy_supabase/test_real_data/test_pain_point_detection_detailed.py",
    ]

    # Run each script sequentially
    for script in scripts:
        exit_code = run_script(script)
        if exit_code != 0:
            print(f"Script {script} failed with exit code {exit_code}")
            # Uncomment the next line if you want to stop on first failure
            # sys.exit(exit_code)

    print("All scripts executed successfully!")


if __name__ == "__main__":
    main()
