#!/usr/bin/env python3
"""
Test script to evaluate improved text generation for inverse embedding attack
"""

import os
import sys
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from utils.beam_search import beam_decode_sentence

def test_improved_generation():
    """Test improved generation with current models"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test sentences
    test_sentences = [
        "This movie is absolutely fantastic!",
        "I really enjoyed watching this film.",
        "The acting was superb and the plot was engaging."
    ]
    
    # Black-box model
    blackbox_model = SentenceTransformer('all-mpnet-base-v2')
    blackbox_embeddings = blackbox_model.encode(test_sentences, convert_to_tensor=True)
    
    print("Testing improved generation...")
    print("=" * 50)
    
    # Test different attacker models
    attackers = {
        'gpt2_paraphrase-MiniLM-L6-v2': {
            'model_path': 'models/attacker_gpt2_large_personachat_paraphrase-MiniLM-L6-v2',
            'tokenizer_path': 'microsoft/DialoGPT-medium',
            'embedding_model': 'paraphrase-MiniLM-L6-v2',
            'model_type': 'causal'
        },
        't5_paraphrase-MiniLM-L6-v2': {
            'model_path': 'models/attacker_t5_large_personachat_paraphrase-MiniLM-L6-v2', 
            'tokenizer_path': 't5-base',
            'embedding_model': 'paraphrase-MiniLM-L6-v2',
            'model_type': 't5'
        },
        'opt_paraphrase-MiniLM-L6-v2': {
            'model_path': 'models/attacker_opt_large_personachat_paraphrase-MiniLM-L6-v2',
            'tokenizer_path': 'facebook/opt-350m',
            'embedding_model': 'paraphrase-MiniLM-L6-v2', 
            'model_type': 'causal'
        }
    }
    
    results = {}
    
    for attacker_name, config in attackers.items():
        print(f"\nTesting {attacker_name}...")
        
        try:
            # Load model and tokenizer
            if config['model_type'] == 't5':
                model = T5ForConditionalGeneration.from_pretrained(config['model_path'])
                tokenizer = T5Tokenizer.from_pretrained(config['tokenizer_path'])
            else:
                model = AutoModelForCausalLM.from_pretrained(config['model_path'])
                tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])
            
            model.to(device)
            model.eval()
            
            # Set pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load projection if needed
            projection = None
            proj_path = config['model_path'] + '/projection.pt'
            if os.path.exists(proj_path):
                from src.attackers.train_attackers import LinearProjection
                embedding_dim = 384  # paraphrase-MiniLM-L6-v2 dimension
                projection = LinearProjection(embedding_dim, model.config.hidden_size)
                projection.load_state_dict(torch.load(proj_path))
                projection.to(device)
                print(f"Loaded projection: {embedding_dim} -> {model.config.hidden_size}")
            
            # Generate text for each embedding
            generated_texts = []
            for i, embedding in enumerate(blackbox_embeddings):
                print(f"  Original: {test_sentences[i]}")
                
                # Apply projection if needed
                embedding_tensor = embedding.unsqueeze(0).to(device)
                if projection is not None:
                    embedding_tensor = projection(embedding_tensor)
                
                # Generate text
                config_dict = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'device': device
                }
                
                generated_text = beam_decode_sentence(
                    hidden_X=embedding_tensor.squeeze(0),
                    config=config_dict,
                    num_generate=1,
                    beam_size=5
                )
                
                if isinstance(generated_text, list):
                    generated_text = generated_text[0]
                
                generated_texts.append(generated_text)
                print(f"  Generated: {generated_text}")
                print()
            
            # Calculate similarity
            generated_embeddings = blackbox_model.encode(generated_texts, convert_to_tensor=True)
            from sentence_transformers import util
            cosine_scores = util.cos_sim(blackbox_embeddings, generated_embeddings)
            similarity_scores = [cosine_scores[i][i].item() for i in range(len(test_sentences))]
            avg_similarity = np.mean(similarity_scores)
            
            results[attacker_name] = {
                'avg_similarity': avg_similarity,
                'generated_texts': generated_texts,
                'similarity_scores': similarity_scores
            }
            
            print(f"  Average similarity: {avg_similarity:.4f}")
            
        except Exception as e:
            print(f"  Error testing {attacker_name}: {e}")
            results[attacker_name] = {
                'error': str(e),
                'avg_similarity': 0.0
            }
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    for attacker_name, result in results.items():
        if 'error' not in result:
            print(f"{attacker_name}: {result['avg_similarity']:.4f}")
        else:
            print(f"{attacker_name}: ERROR - {result['error']}")
    
    # Save results
    with open('test_improved_generation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to test_improved_generation_results.json")

if __name__ == "__main__":
    test_improved_generation() 