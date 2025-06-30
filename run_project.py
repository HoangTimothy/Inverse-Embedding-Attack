#!/usr/bin/env python3
"""
Run Inverse Embedding Attack project in correct order
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_command(command, description, check=True):
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"Success! (Took {duration:.1f} seconds)")
            if result.stdout:
                print("Output:", result.stdout[-500:])
            return True
        else:
            print(f"Failed! (Took {duration:.1f} seconds)")
            print("Error output:", result.stderr)
            return False
            
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"Error! (Took {duration:.1f} seconds)")
        print("Error output:", e.stderr)
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def check_environment():
    print("Checking environment...")
    
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("Python 3.8+ required")
        return False
    
    print(f"Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available - will use CPU")
        return True
    except ImportError:
        print("PyTorch not installed")
        return False

def main():
    print("Inverse Embedding Attack - Project Runner")
    print("=" * 60)
    
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    print(f"Working directory: {os.getcwd()}")
    
    if not check_environment():
        print("Environment check failed. Please install dependencies first.")
        return False
    
    print("\nStep 1: Installing dependencies...")
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("Failed to install dependencies")
        return False
    
    print("\nStep 2: Preparing datasets...")
    if not run_command("python prepare_all_datasets.py", "Preparing all datasets"):
        print("Failed to prepare datasets")
        return False
    
    print("\nStep 3: Training attackers...")
    if not run_command("python train_all_attackers.py", "Training all attackers"):
        print("Failed to train attackers")
        return False
    
    print("\nStep 4: Verifying trained models...")
    if not run_command("python train_all_attackers.py --verify-only", "Verifying models"):
        print("Failed to verify models")
        return False
    
    print("\nStep 5: Running evaluation...")
    if not run_command("python src/evaluation/test_blackbox.py --dataset sst2 --split test --blackbox_model all-mpnet-base-v2", "Running black-box evaluation"):
        print("Failed to run evaluation")
        return False
    
    print("\nStep 6: Checking results...")
    results_dir = Path("experiments/results")
    if results_dir.exists():
        print("Results directory exists")
        for result_file in results_dir.glob("*.json"):
            print(f"   {result_file.name}")
    else:
        print("Results directory not found")
    
    print(f"\n{'='*60}")
    print("Project completed successfully!")
    print("Check results in: experiments/results/")
    print("Read RUNNING_GUIDE.md for detailed information")
    print(f"{'='*60}")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\nProject failed! Check the errors above.")
        sys.exit(1) 