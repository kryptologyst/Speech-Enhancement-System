"""Setup script for speech enhancement system."""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False


def main():
    """Setup the speech enhancement system."""
    print("Setting up Speech Enhancement System...")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8+ is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("Failed to install dependencies")
        sys.exit(1)
    
    # Create directories
    directories = [
        "data/raw",
        "data/processed", 
        "checkpoints",
        "outputs",
        "logs",
        "results",
        "assets",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Generate synthetic dataset
    if not run_command(
        "python scripts/generate_data.py --data_dir data --n_samples_train 100 --n_samples_val 20 --n_samples_test 20",
        "Generating synthetic dataset"
    ):
        print("Failed to generate dataset")
        sys.exit(1)
    
    # Run basic tests
    if not run_command("python -m pytest tests/test_basic.py -v", "Running basic tests"):
        print("Some tests failed, but setup can continue")
    
    print("\n" + "=" * 50)
    print("Setup completed successfully!")
    print("\nNext steps:")
    print("1. Train a model: python scripts/train.py --model_type conv_tasnet --epochs 10")
    print("2. Run the demo: streamlit run demo/streamlit_demo.py")
    print("3. Check the README.md for more information")
    print("\nRemember: This is for research and educational purposes only!")


if __name__ == "__main__":
    main()
