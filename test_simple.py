#!/usr/bin/env python3
"""
Simple test script to check if all components work
"""

import os
import sys
import torch
from sentence_transformers import SentenceTransformer

# Add project to path
sys.path.append(os.path.dirname(__file__))
from config import EMBEDDING_MODELS, TRAIN_CONFIG

def test_embedding_models():
    """Test if embedding models can be loaded"""
    print("Testing embedding models...")
    
    device = TRAIN_CONFIG['device']
    
    # Test first model only
    model_name = list(EMBEDDING_MODELS.keys())[0]
    config = EMBEDDING_MODELS[model_name]
    
    try:
        print(f"Loading {model_name}...")
        model = SentenceTransformer(config['path'], device=device)
        print(f"✅ Successfully loaded {model_name}")
        
        # Test encoding
        test_sentences = ["This is a test sentence.", "Another test sentence."]
        embeddings = model.encode(test_sentences, convert_to_tensor=True)
        print(f"✅ Successfully encoded sentences, shape: {embeddings.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")
        return False

def test_attacker_models():
    """Test if attacker models can be loaded"""
    print("\nTesting attacker models...")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.attackers.attacker_models import SequenceCrossEntropyLoss
        
        # Test GPT-2
        print("Loading GPT-2...")
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        print("✅ Successfully loaded GPT-2")
        
        # Test loss function
        loss_fn = SequenceCrossEntropyLoss()
        print("✅ Successfully loaded SequenceCrossEntropyLoss")
        
        return True
    except Exception as e:
        print(f"❌ Error loading attacker models: {e}")
        return False

def test_alignment():
    """Test alignment functionality"""
    print("\nTesting alignment...")
    
    try:
        from src.utils.alignment import EmbeddingAlignment
        
        # Create test data
        source_embeddings = torch.randn(10, 384)
        target_embeddings = torch.randn(10, 768)
        
        # Test alignment
        alignment = EmbeddingAlignment(384, 768, method='linear')
        alignment.fit(source_embeddings.numpy(), target_embeddings.numpy())
        
        # Test transform
        aligned = alignment.transform(source_embeddings)
        print(f"✅ Successfully aligned embeddings, shape: {aligned.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing alignment: {e}")
        return False

def test_dataset_loading():
    """Test dataset loading"""
    print("\nTesting dataset loading...")
    
    try:
        from datasets import load_dataset
        
        # Try to load a small dataset without cache_dir to avoid path issues
        dataset = load_dataset('glue', 'sst2', split='train[:10]')
        sentences = [item['sentence'] for item in dataset]
        print(f"✅ Successfully loaded {len(sentences)} sentences from SST-2")
        
        return True
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("Using sample data instead...")
        return False

def main():
    print("🧪 Testing Inverse Embedding Attack Components")
    print("=" * 50)
    
    tests = [
        ("Embedding Models", test_embedding_models),
        ("Attacker Models", test_attacker_models),
        ("Alignment", test_alignment),
        ("Dataset Loading", test_dataset_loading)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! You can now run the full experiment.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main() 