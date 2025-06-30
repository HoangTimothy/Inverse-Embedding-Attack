#!/usr/bin/env python3
"""
Train all 3 attackers for inverse embedding attack
Uses the prepared datasets and training configuration
"""

import os
import sys
import json
import torch
import argparse
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from attackers.train_attackers import InverseEmbeddingAttacker
from config import PATHS, TRAIN_CONFIG

def load_training_config():
    """Load training configuration"""
    config_path = os.path.join(PATHS['embeddings_dir'], 'training_config.json')
    
    if not os.path.exists(config_path):
        print(f"❌ Training config not found: {config_path}")
        print("Please run prepare_all_datasets.py first")
        return None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def train_attacker(attacker_model, config, training_params):
    """Train a single attacker"""
    print(f"\n{'='*50}")
    print(f"TRAINING ATTACKER: {attacker_model}")
    print(f"{'='*50}")
    
    attacker_config = config['attackers'][attacker_model]
    dataset_name = attacker_config['dataset']
    embedding_dim = attacker_config['embedding_dim']
    
    print(f"Dataset: {dataset_name}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Black-box dimension: {attacker_config['blackbox_dim']}")
    print(f"Needs projection: {attacker_config['needs_projection']}")
    
    try:
        # Map embedding models to attacker models
        attacker_model_mapping = {
            'stsb-roberta-base': 'gpt2',
            'all-MiniLM-L6-v2': 'opt', 
            'paraphrase-MiniLM-L6-v2': 't5'
        }
        
        attacker_model_name = attacker_model_mapping.get(attacker_model)
        if attacker_model_name is None:
            raise ValueError(f"Unknown embedding model: {attacker_model}")
        
        print(f"Using attacker model: {attacker_model_name}")
        
        # Initialize attacker
        attacker = InverseEmbeddingAttacker(
            attacker_model_name=attacker_model_name,  # gpt2, opt, t5
            embedding_model_name=attacker_model,      # embedding model name
            embedding_dim=embedding_dim
        )
        
        # Train attacker
        attacker.train(
            dataset_name=dataset_name,
            split='train'
        )
        
        print(f"✅ Training completed for {attacker_model}")
        return True
        
    except Exception as e:
        print(f"❌ Error training {attacker_model}: {e}")
        import traceback
        traceback.print_exc()
        return False

def train_all_attackers():
    """Train all 3 attackers"""
    print("=" * 60)
    print("TRAINING ALL ATTACKERS")
    print("=" * 60)
    
    # Load training configuration
    config = load_training_config()
    if config is None:
        return False
    
    # Training parameters
    training_params = {
        'num_epochs': 5,  # Adjust based on your needs
        'batch_size': 8,  # Adjust based on GPU memory
        'learning_rate': 2e-5
    }
    
    print(f"Training parameters:")
    print(f"  Epochs: {training_params['num_epochs']}")
    print(f"  Batch size: {training_params['batch_size']}")
    print(f"  Learning rate: {training_params['learning_rate']}")
    
    # Get attacker models
    attacker_models = list(config['attackers'].keys())
    print(f"\nAttacker models: {attacker_models}")
    
    # Train each attacker
    results = {}
    
    for attacker_model in attacker_models:
        success = train_attacker(attacker_model, config, training_params)
        results[attacker_model] = success
    
    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    
    for attacker_model, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{attacker_model}: {status}")
    
    print(f"\nOverall: {successful}/{total} attackers trained successfully")
    
    if successful == total:
        print("🎉 All attackers trained successfully!")
        print("✅ Ready for evaluation")
    else:
        print("⚠️  Some attackers failed to train")
    
    return successful == total

def verify_trained_models():
    """Verify that all trained models exist"""
    print(f"\n{'='*60}")
    print("VERIFYING TRAINED MODELS")
    print(f"{'='*60}")
    
    # Load training configuration
    config = load_training_config()
    if config is None:
        return False
    
    # Map embedding models to attacker models
    attacker_model_mapping = {
        'stsb-roberta-base': 'gpt2',
        'all-MiniLM-L6-v2': 'opt', 
        'paraphrase-MiniLM-L6-v2': 't5'
    }
    
    all_exist = True
    
    for embedding_model in config['attackers'].keys():
        attacker_type = attacker_model_mapping.get(embedding_model)
        if attacker_type is None:
            print(f"❌ {embedding_model}: Unknown attacker type")
            all_exist = False
            continue
        
        # Look for models with pattern: attacker_{attacker_type}_{embedding_model}_epoch_{epoch}
        model_pattern = f"attacker_{attacker_type}_{embedding_model}_epoch_"
        models_dir = PATHS['models_dir']
        
        # Check if any model with this pattern exists
        model_found = False
        if os.path.exists(models_dir):
            for item in os.listdir(models_dir):
                if item.startswith(model_pattern):
                    print(f"✅ {embedding_model}: {os.path.join(models_dir, item)}")
                    model_found = True
                    break
        
        if not model_found:
            print(f"❌ {embedding_model}: No model found with pattern '{model_pattern}*'")
            all_exist = False
    
    if all_exist:
        print("\n🎉 All trained models verified!")
    else:
        print("\n⚠️  Some trained models are missing")
    
    return all_exist

def main():
    parser = argparse.ArgumentParser(description='Train all attackers')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify existing trained models')
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_trained_models()
    else:
        success = train_all_attackers()
        if success:
            verify_trained_models()
    
    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print("1. Run evaluation on trained attackers")
    print("2. Test black-box attack performance")
    print("3. Generate final results")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 