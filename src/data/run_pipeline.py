import subprocess
import os
import sys

def run_script(script_name):
    """Run a Python script and handle its execution."""
    print(f"Running {script_name}...")
    if not os.path.isfile(script_name):
        print(f"Error: {script_name} not found in the current directory.")
        sys.exit(1)
    
    try:

        result = subprocess.run([sys.executable, script_name], check=True, text=True, capture_output=True)
        print(result.stdout)
        print(f"{script_name} completed successfully.")
        print("------------------------")
    except subprocess.CalledProcessError as e:
        print(f"Error: {script_name} failed.")
        print(f"Exit code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)

def main():
    """Execute the pipeline scripts in sequence."""
    if not sys.executable:
        print("Error: Python interpreter not found.")
        sys.exit(1)

    scripts = [
        "data_splitting.py",
        "data_norm.py",
        "grid_search_ridge.py",
        "model_eval.py",

    ]

    print("Starting pipeline execution...")
    print("------------------------")

    for script in scripts:
        run_script(script)

    print("Pipeline executed successfully!")

if __name__ == "__main__":
    main()