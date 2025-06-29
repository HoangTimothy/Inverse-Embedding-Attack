#!/usr/bin/env python3
"""
Main script to run the complete Inverse Embedding Attack experiment
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Error!")
        print("Error output:", e.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description='Run complete Inverse Embedding Attack experiment')
    parser.add_argument('--dataset', type=str, default='sst2', 
                       choices=['sst2', 'personachat', 'abcd'],
                       help='Dataset to use')
    parser.add_argument('--blackbox_model', type=str, default='all-mpnet-base-v2',
                       help='Black-box model to test against')
    parser.add_argument('--skip_prepare', action='store_true',
                       help='Skip data preparation step')
    parser.add_argument('--skip_train', action='store_true',
                       help='Skip training step')
    parser.add_argument('--skip_test', action='store_true',
                       help='Skip testing step')
    
    args = parser.parse_args()
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    print("🚀 Starting Inverse Embedding Attack Experiment")
    print(f"Dataset: {args.dataset}")
    print(f"Black-box model: {args.blackbox_model}")
    
    success = True
    
    # Step 1: Prepare embeddings
    if not args.skip_prepare:
        print("\n📊 Step 1: Preparing embeddings...")
        for split in ['train', 'dev', 'test']:
            cmd = f"python src/data_processing/prepare_embeddings.py --dataset {args.dataset} --split {split} --create_alignment"
            if not run_command(cmd, f"Preparing embeddings for {split} split"):
                success = False
                break
    
    # Step 2: Train attackers
    if not args.skip_train and success:
        print("\n🎯 Step 2: Training attackers...")
        
        # Train on different embedding models (excluding black-box)
        embedding_models = ['all-mpnet-base-v2', 'stsb-roberta-base', 'all-MiniLM-L6-v2', 'paraphrase-MiniLM-L6-v2']
        attacker_models = ['gpt2', 'opt', 't5']
        
        for embedding_model in embedding_models:
            if embedding_model != args.blackbox_model:  # Don't train on black-box model
                for attacker in attacker_models:
                    cmd = f"python src/attackers/train_attackers.py --dataset {args.dataset} --split train --attacker {attacker} --embedding_model {embedding_model}"
                    if not run_command(cmd, f"Training {attacker} on {embedding_model}"):
                        success = False
                        break
                if not success:
                    break
    
    # Step 3: Test on black-box
    if not args.skip_test and success:
        print("\n🔍 Step 3: Testing on black-box model...")
        cmd = f"python src/evaluation/test_blackbox.py --dataset {args.dataset} --split test --blackbox_model {args.blackbox_model}"
        if not run_command(cmd, f"Testing attackers on black-box model {args.blackbox_model}"):
            success = False
    
    # Summary
    print(f"\n{'='*50}")
    if success:
        print("🎉 Experiment completed successfully!")
        print(f"Results saved in: {os.path.join('experiments', 'results')}")
    else:
        print("❌ Experiment failed!")
    print(f"{'='*50}")

if __name__ == "__main__":
    main() 