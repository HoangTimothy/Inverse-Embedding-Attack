#!/usr/bin/env python3
"""
Demo script for Inverse Embedding Attack
"""

import os
import sys
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

# Add project to path
sys.path.append(os.path.dirname(__file__))
from config import EMBEDDING_MODELS, ATTACKER_MODELS

def demo_embedding_extraction():
    """Demo: Extract embeddings from different models"""
    print("Demo: Embedding Extraction")
    print("="*50)
    
    # Sample sentences
    sentences = [
        "I love this movie!",
        "This is terrible.",
        "The weather is nice today.",
        "I'm going to the store."
    ]
    
    # Test different embedding models
    for model_name, config in list(EMBEDDING_MODELS.items())[:2]:  # Test first 2 models
        print(f"\nTesting {model_name}:")
        
        try:
            model = SentenceTransformer(config['path'])
            embeddings = model.encode(sentences, convert_to_tensor=True)
            
            print(f"  Embedding dimension: {embeddings.shape[1]}")
            print(f"  Sample embedding (first sentence): {embeddings[0][:5].tolist()}...")
            
            # Calculate similarity between first two sentences
            similarity = torch.cosine_similarity(embeddings[0:1], embeddings[1:2])
            print(f"  Similarity between 'I love this movie!' and 'This is terrible.': {similarity.item():.4f}")
            
        except Exception as e:
            print(f"  Error: {e}")

def demo_alignment():
    """Demo: Alignment between different embedding spaces"""
    print("\nDemo: Embedding Alignment")
    print("="*50)
    
    # Create sample embeddings with different dimensions
    np.random.seed(42)
    source_embeddings = np.random.randn(100, 384)  # 384-dim
    target_embeddings = np.random.randn(100, 768)  # 768-dim
    
    print(f"Source embeddings shape: {source_embeddings.shape}")
    print(f"Target embeddings shape: {target_embeddings.shape}")
    
    # Import alignment utility
    from src.utils.alignment import EmbeddingAlignment
    
    # Create and fit alignment
    alignment = EmbeddingAlignment(384, 768, method='linear')
    alignment.fit(source_embeddings, target_embeddings)
    
    # Transform embeddings
    aligned_embeddings = alignment.transform(source_embeddings)
    print(f"Aligned embeddings shape: {aligned_embeddings.shape}")
    
    # Test with new data
    test_embeddings = np.random.randn(10, 384)
    test_aligned = alignment.transform(test_embeddings)
    print(f"Test aligned shape: {test_aligned.shape}")

def demo_attack_simulation():
    """Demo: Simulate attack process"""
    print("\nDemo: Attack Simulation")
    print("="*50)
    
    # Simulate the attack process
    print("1. Input: Black-box embedding")
    print("   [0.3, -0.9, 0.7, 0.1, -0.5, ...]")
    
    print("\n2. Alignment (if needed)")
    print("   [0.3, -0.9, 0.7, 0.1, -0.5, ...] -> [0.2, -0.8, 0.6, ...]")
    
    print("\n3. Attacker Model")
    print("   GPT-2/OPT/T5 generates text from embedding")
    
    print("\n4. Output: Recovered text")
    print("   'I love this movie!'")
    
    print("\n5. Evaluation")
    print("   - Embedding similarity: 0.85")
    print("   - Exact match: False")
    print("   - BLEU score: 0.72")

def demo_complete_flow():
    """Demo: Complete attack flow"""
    print("\nDemo: Complete Attack Flow")
    print("="*50)
    
    # Step 1: Prepare data
    print("Step 1: Data Preparation")
    print("  - Load SST-2 dataset")
    print("  - Create embeddings with 4 different models")
    print("  - Save embeddings to data/embeddings/")
    
    # Step 2: Train attackers
    print("\nStep 2: Train Attackers")
    print("  - Train GPT-2 on all-mpnet-base-v2 embeddings")
    print("  - Train OPT on stsb-roberta-base embeddings") 
    print("  - Train T5 on all-MiniLM-L6-v2 embeddings")
    print("  - Save models to models/")
    
    # Step 3: Test on black-box
    print("\nStep 3: Test on Black-box")
    print("  - Load paraphrase-MiniLM-L6-v2 as black-box")
    print("  - Generate embeddings from test sentences")
    print("  - Use trained attackers to recover text")
    print("  - Evaluate performance")
    
    # Expected results
    print("\nExpected Results:")
    print("  - Embedding similarity: 0.7-0.9")
    print("  - Exact match rate: 0.1-0.3")
    print("  - BLEU score: 0.5-0.8")

def main():
    print("Inverse Embedding Attack Demo")
    print("="*60)
    print("This demo shows the key components of our solution:")
    print("- Embedding extraction from multiple models")
    print("- Alignment between different embedding spaces")
    print("- Attack simulation and evaluation")
    print("- Complete experimental flow")
    
    # Run demos
    demo_embedding_extraction()
    demo_alignment()
    demo_attack_simulation()
    demo_complete_flow()
    
    print("\n" + "="*60)
    print("Demo completed!")
    print("\nTo run the full experiment:")
    print("python run_experiment.py --dataset sst2 --blackbox_model all-mpnet-base-v2")
    print("\nTo prepare data only:")
    print("python src/data_processing/prepare_embeddings.py --dataset sst2 --split train")

if __name__ == "__main__":
    main() 