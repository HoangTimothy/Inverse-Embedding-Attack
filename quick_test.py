#!/usr/bin/env python3
"""
Quick test script for Inverse Embedding Attack
"""

import os
import sys
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

# Add project to path
sys.path.append(os.path.dirname(__file__))
from config import EMBEDDING_MODELS, TRAIN_CONFIG, PATHS

def create_sample_data():
    """Create sample data for testing"""
    sample_sentences = [
        "This is a great movie!",
        "I love this film.",
        "The weather is nice today.",
        "I'm going to the store.",
        "This restaurant is amazing.",
        "The book was interesting.",
        "I enjoyed the concert.",
        "The food was delicious.",
        "The movie was terrible.",
        "I didn't like it."
    ]
    return sample_sentences

def test_embedding_generation():
    """Test embedding generation with sample data"""
    print("🔍 Testing embedding generation...")
    
    # Create sample data
    sentences = create_sample_data()
    print(f"Created {len(sentences)} sample sentences")
    
    # Test first embedding model
    model_name = list(EMBEDDING_MODELS.keys())[0]
    config = EMBEDDING_MODELS[model_name]
    
    try:
        # Load model
        device = TRAIN_CONFIG['device']
        model = SentenceTransformer(config['path'], device=device)
        print(f"✅ Loaded {model_name}")
        
        # Generate embeddings
        embeddings = model.encode(sentences, convert_to_tensor=True)
        print(f"✅ Generated embeddings, shape: {embeddings.shape}")
        
        # Save embeddings
        save_path = os.path.join(PATHS['embeddings_dir'], f"test_{model_name}.json")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        data = {
            'embeddings': embeddings.cpu().numpy().tolist(),
            'sentences': sentences,
            'model_name': model_name,
            'embedding_dim': config['dim']
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Saved embeddings to {save_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_alignment():
    """Test alignment functionality"""
    print("\n🔄 Testing alignment...")
    
    try:
        from src.utils.alignment import EmbeddingAlignment
        
        # Create test embeddings
        source_embeddings = torch.randn(10, 384)
        target_embeddings = torch.randn(10, 768)
        
        # Create alignment
        alignment = EmbeddingAlignment(384, 768, method='linear')
        alignment.fit(source_embeddings.numpy(), target_embeddings.numpy())
        
        # Test transform
        aligned = alignment.transform(source_embeddings)
        print(f"✅ Successfully aligned embeddings, shape: {aligned.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_attacker_components():
    """Test attacker components"""
    print("\n⚔️ Testing attacker components...")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.attackers.attacker_models import SequenceCrossEntropyLoss
        
        # Load small model for testing
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        
        # Test loss function
        loss_fn = SequenceCrossEntropyLoss()
        
        print("✅ Successfully loaded attacker components")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_complete_pipeline():
    """Test complete pipeline with sample data"""
    print("\n🚀 Testing complete pipeline...")
    
    try:
        # 1. Generate embeddings
        if not test_embedding_generation():
            return False
        
        # 2. Test alignment
        if not test_alignment():
            return False
        
        # 3. Test attacker components
        if not test_attacker_components():
            return False
        
        print("✅ All pipeline components working!")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        return False

def main():
    print("🎯 Quick Test for Inverse Embedding Attack")
    print("=" * 50)
    
    # Test complete pipeline
    success = test_complete_pipeline()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed!")
        print("\nNext steps:")
        print("1. Run: python test_simple.py")
        print("2. Run: python src/data_processing/prepare_embeddings.py --dataset sst2 --split train")
        print("3. Run: python run_experiment.py --dataset sst2 --blackbox_model all-mpnet-base-v2")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    print("=" * 50)

if __name__ == "__main__":
    main() 