#!/usr/bin/env python3
"""
Prepare all 4 embedding datasets for inverse embedding attack
Creates datasets for all 4 embedding models as required by the report
"""

import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from data_processing.prepare_embeddings import EmbeddingPreparer
from config import EMBEDDING_MODELS, PATHS

def prepare_all_datasets():
    """Prepare all 4 embedding datasets for all 3 datasets"""
    
    print("=" * 60)
    print("PREPARING ALL EMBEDDING DATASETS")
    print("=" * 60)
    
    # Initialize preparer
    preparer = EmbeddingPreparer()
    
    # Datasets to process
    datasets = ['sst2', 'personachat', 'abcd']
    splits = ['train', 'dev', 'test']
    
    # Track statistics
    stats = {}
    
    for dataset_name in datasets:
        print(f"\n{'='*40}")
        print(f"Processing dataset: {dataset_name.upper()}")
        print(f"{'='*40}")
        
        stats[dataset_name] = {}
        
        for split in splits:
            print(f"\n--- Processing {split} split ---")
            
            try:
                # Load dataset using GEIA
                sentences = preparer.load_dataset(dataset_name, split)
                print(f"Loaded {len(sentences)} sentences from {dataset_name} {split}")
                
                # Create embeddings for all 4 models
                for model_name in EMBEDDING_MODELS.keys():
                    print(f"  Creating embeddings for {model_name}...")
                    
                    save_path = os.path.join(
                        PATHS['embeddings_dir'], 
                        f"{dataset_name}_{split}_{model_name}.json"
                    )
                    
                    embeddings = preparer.create_embeddings(sentences, model_name, save_path)
                    
                    # Store statistics
                    if model_name not in stats[dataset_name]:
                        stats[dataset_name][model_name] = {}
                    
                    stats[dataset_name][model_name][split] = {
                        'num_samples': len(sentences),
                        'embedding_dim': embeddings.shape[1],
                        'file_path': save_path
                    }
                    
                    print(f"    ✅ Saved {len(sentences)} samples, dim: {embeddings.shape[1]}")
                
            except Exception as e:
                print(f"❌ Error processing {dataset_name} {split}: {e}")
                continue
    
    # Save statistics
    stats_path = os.path.join(PATHS['embeddings_dir'], 'dataset_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'='*60}")
    print("DATASET PREPARATION COMPLETED")
    print(f"{'='*60}")
    print(f"Statistics saved to: {stats_path}")
    
    # Print summary
    print("\n📊 DATASET SUMMARY:")
    for dataset_name, dataset_stats in stats.items():
        print(f"\n{dataset_name.upper()}:")
        for model_name, model_stats in dataset_stats.items():
            print(f"  {model_name}:")
            for split, split_stats in model_stats.items():
                print(f"    {split}: {split_stats['num_samples']} samples, dim: {split_stats['embedding_dim']}")
    
    return stats

def verify_requirements(stats):
    """Verify that all requirements are met"""
    print(f"\n{'='*60}")
    print("VERIFYING REQUIREMENTS")
    print(f"{'='*60}")
    
    requirements_met = True
    
    # Check 1: 4 embedding models
    print("\n1. Checking 4 embedding models...")
    expected_models = ['all-mpnet-base-v2', 'stsb-roberta-base', 'all-MiniLM-L6-v2', 'paraphrase-MiniLM-L6-v2']
    for model in expected_models:
        if model in EMBEDDING_MODELS:
            print(f"  ✅ {model}")
        else:
            print(f"  ❌ {model} - MISSING")
            requirements_met = False
    
    # Check 2: 3 datasets
    print("\n2. Checking 3 datasets...")
    expected_datasets = ['sst2', 'personachat', 'abcd']
    for dataset in expected_datasets:
        if dataset in stats:
            print(f"  ✅ {dataset}")
        else:
            print(f"  ❌ {dataset} - MISSING")
            requirements_met = False
    
    # Check 3: 10,000 samples for training
    print("\n3. Checking 10,000 samples for training...")
    for dataset_name, dataset_stats in stats.items():
        for model_name, model_stats in dataset_stats.items():
            if 'train' in model_stats:
                num_samples = model_stats['train']['num_samples']
                if num_samples >= 10000:
                    print(f"  ✅ {dataset_name}_{model_name}: {num_samples} samples")
                else:
                    print(f"  ⚠️  {dataset_name}_{model_name}: {num_samples} samples (less than 10,000)")
    
    # Check 4: Black-box model (all-mpnet-base-v2)
    print("\n4. Checking black-box model...")
    blackbox_model = 'all-mpnet-base-v2'
    if blackbox_model in EMBEDDING_MODELS:
        print(f"  ✅ Black-box model: {blackbox_model}")
        print(f"  ✅ Dimension: {EMBEDDING_MODELS[blackbox_model]['dim']}")
    else:
        print(f"  ❌ Black-box model {blackbox_model} - MISSING")
        requirements_met = False
    
    # Check 5: All 4 datasets created
    print("\n5. Checking all 4 embedding datasets...")
    total_datasets = 0
    for dataset_name, dataset_stats in stats.items():
        for model_name, model_stats in dataset_stats.items():
            if 'train' in model_stats:
                total_datasets += 1
                print(f"  ✅ {dataset_name}_{model_name}")
    
    if total_datasets >= 12:  # 3 datasets × 4 models
        print(f"  ✅ Total datasets: {total_datasets}")
    else:
        print(f"  ❌ Total datasets: {total_datasets} (expected 12)")
        requirements_met = False
    
    print(f"\n{'='*60}")
    if requirements_met:
        print("🎉 ALL REQUIREMENTS MET!")
        print("✅ Ready for attacker training")
    else:
        print("⚠️  SOME REQUIREMENTS NOT MET")
        print("Please check the issues above")
    
    return requirements_met

def create_training_config():
    """Create training configuration for the 3 attackers"""
    print(f"\n{'='*60}")
    print("CREATING TRAINING CONFIGURATION")
    print(f"{'='*60}")
    
    # Black-box model (for prediction)
    blackbox_model = 'all-mpnet-base-v2'
    
    # 3 attackers (excluding black-box model)
    attacker_models = ['stsb-roberta-base', 'all-MiniLM-L6-v2', 'paraphrase-MiniLM-L6-v2']
    
    # Datasets to use
    datasets = ['sst2', 'personachat', 'abcd']
    
    training_config = {
        'blackbox_model': blackbox_model,
        'attackers': {},
        'datasets': datasets
    }
    
    # Create configuration for each attacker
    for i, attacker_model in enumerate(attacker_models):
        # Assign dataset (round-robin)
        dataset = datasets[i % len(datasets)]
        
        training_config['attackers'][attacker_model] = {
            'dataset': dataset,
            'embedding_dim': EMBEDDING_MODELS[attacker_model]['dim'],
            'blackbox_dim': EMBEDDING_MODELS[blackbox_model]['dim'],
            'needs_projection': EMBEDDING_MODELS[attacker_model]['dim'] != EMBEDDING_MODELS[blackbox_model]['dim']
        }
    
    # Save configuration
    config_path = os.path.join(PATHS['embeddings_dir'], 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(training_config, f, indent=2)
    
    print("Training configuration:")
    print(f"Black-box model: {blackbox_model}")
    print("\nAttackers:")
    for attacker_model, config in training_config['attackers'].items():
        print(f"  {attacker_model}:")
        print(f"    Dataset: {config['dataset']}")
        print(f"    Embedding dim: {config['embedding_dim']}")
        print(f"    Needs projection: {config['needs_projection']}")
    
    print(f"\nConfiguration saved to: {config_path}")
    
    return training_config

def main():
    parser = argparse.ArgumentParser(description='Prepare all embedding datasets')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify existing datasets without creating new ones')
    
    args = parser.parse_args()
    
    if args.verify_only:
        # Load existing statistics
        stats_path = os.path.join(PATHS['embeddings_dir'], 'dataset_statistics.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            verify_requirements(stats)
            create_training_config()
        else:
            print("❌ No existing statistics found. Run without --verify-only to create datasets.")
    else:
        # Prepare all datasets
        stats = prepare_all_datasets()
        verify_requirements(stats)
        create_training_config()
    
    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print("1. Train attackers using the created datasets")
    print("2. Use black-box model (all-mpnet-base-v2) for prediction")
    print("3. Evaluate attack performance")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 