#!/usr/bin/env python3
"""
GEIA-based evaluation framework for inverse embedding attack
Uses GEIA's evaluation metrics and methods
"""

import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# Add GEIA to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'GEIA'))
from eval_generation import eval_on_batch
from eval_classification import eval_classification
from eval_ppl import eval_ppl

# Add local path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import ATTACKER_MODELS, EMBEDDING_MODELS, PATHS, EVAL_CONFIG

class GEIAEvaluator:
    def __init__(self, blackbox_model_name='all-mpnet-base-v2'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.blackbox_model_name = blackbox_model_name
        self.blackbox_model = SentenceTransformer(blackbox_model_name)
        
        # Load all embedding models for evaluation
        self.embedding_models = {}
        for model_name, config in EMBEDDING_MODELS.items():
            self.embedding_models[model_name] = SentenceTransformer(config['path'])
    
    def load_test_data(self, dataset_name, split='test'):
        """Load test data using GEIA data pipeline"""
        # Import GEIA data processing
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'GEIA'))
        from data_process import get_sent_list
        
        config = {
            'dataset': dataset_name,
            'data_type': split
        }
        
        sentences = get_sent_list(config)
        
        # Use reasonable test size
        if len(sentences) > 1000:
            sentences = sentences[:1000]
        
        print(f"Loaded {len(sentences)} test sentences from {dataset_name}")
        return sentences
    
    def evaluate_embedding_similarity(self, original_sentences, generated_sentences):
        """Evaluate embedding similarity using GEIA approach"""
        print("Evaluating embedding similarity...")
        
        # Get embeddings from black-box model
        original_embeddings = self.blackbox_model.encode(original_sentences, convert_to_tensor=True)
        generated_embeddings = self.blackbox_model.encode(generated_sentences, convert_to_tensor=True)
        
        # Calculate cosine similarity
        cosine_scores = util.cos_sim(original_embeddings, generated_embeddings)
        similarity_scores = [cosine_scores[i][i].item() for i in range(len(original_sentences))]
        
        avg_similarity = np.mean(similarity_scores)
        std_similarity = np.std(similarity_scores)
        
        return {
            'avg_similarity': avg_similarity,
            'std_similarity': std_similarity,
            'similarity_scores': similarity_scores
        }
    
    def evaluate_text_quality(self, generated_sentences):
        """Evaluate text quality using GEIA metrics"""
        print("Evaluating text quality...")
        
        # Calculate perplexity using GEIA's eval_ppl
        try:
            perplexity_scores = []
            for sentence in generated_sentences:
                if sentence.strip():
                    ppl = eval_ppl([sentence])
                    perplexity_scores.append(ppl)
            
            avg_perplexity = np.mean(perplexity_scores) if perplexity_scores else float('inf')
        except:
            avg_perplexity = float('inf')
        
        # Calculate text length statistics
        lengths = [len(sentence.split()) for sentence in generated_sentences if sentence.strip()]
        avg_length = np.mean(lengths) if lengths else 0
        
        # Calculate diversity (unique words ratio)
        all_words = []
        for sentence in generated_sentences:
            if sentence.strip():
                all_words.extend(sentence.lower().split())
        
        unique_words = len(set(all_words))
        total_words = len(all_words)
        diversity = unique_words / total_words if total_words > 0 else 0
        
        return {
            'avg_perplexity': avg_perplexity,
            'avg_length': avg_length,
            'diversity': diversity,
            'num_generated': len([s for s in generated_sentences if s.strip()])
        }
    
    def evaluate_attack_performance(self, attacker_name, dataset_name, split='test'):
        """Comprehensive attack evaluation using GEIA framework"""
        print(f"Evaluating {attacker_name} on {dataset_name}...")
        
        # Load test data
        test_sentences = self.load_test_data(dataset_name, split)
        
        # Get black-box embeddings
        blackbox_embeddings = self.blackbox_model.encode(test_sentences, convert_to_tensor=True)
        
        # Load attacker model
        attacker_info = ATTACKER_MODELS[attacker_name.split('_')[0]]
        model_path = os.path.join(PATHS['models_dir'], f"attacker_{attacker_name}")
        
        if not os.path.exists(model_path):
            print(f"Attacker model not found: {model_path}")
            return None
        
        # Load model and generate text
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
            
            if attacker_name.startswith('t5'):
                model = T5ForConditionalGeneration.from_pretrained(model_path)
                tokenizer = T5Tokenizer.from_pretrained(attacker_info['tokenizer_path'])
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(attacker_info['tokenizer_path'])
            
            model.to(self.device)
            model.eval()
            
            # Set pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load projection if needed
            projection = None
            proj_path = os.path.join(model_path, "projection.pt")
            if os.path.exists(proj_path):
                from src.attackers.train_attackers import LinearProjection
                embedding_model = attacker_name.split('_')[1]
                embedding_dim = EMBEDDING_MODELS[embedding_model]['dim']
                projection = LinearProjection(embedding_dim, model.config.hidden_size)
                projection.load_state_dict(torch.load(proj_path))
                projection.to(self.device)
            
            # Generate text using GEIA's eval_on_batch
            generated_texts = []
            batch_size = 8
            
            for i in tqdm(range(0, len(blackbox_embeddings), batch_size)):
                batch_embeddings = blackbox_embeddings[i:i+batch_size]
                
                # Apply projection if needed
                if projection is not None:
                    batch_embeddings = projection(batch_embeddings)
                
                # Use GEIA's eval_on_batch
                config = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'device': self.device,
                    'decode': 'beam',
                    'beam_size': EVAL_CONFIG['beam_size']
                }
                
                batch_generated, _ = eval_on_batch(
                    batch_X=batch_embeddings,
                    batch_D=test_sentences[i:i+batch_size],
                    model=model,
                    tokenizer=tokenizer,
                    device=self.device,
                    config=config
                )
                
                generated_texts.extend(batch_generated)
            
            # Evaluate using GEIA metrics
            similarity_results = self.evaluate_embedding_similarity(test_sentences, generated_texts)
            quality_results = self.evaluate_text_quality(generated_texts)
            
            # Combine results
            results = {
                'attacker_name': attacker_name,
                'dataset_name': dataset_name,
                'num_samples': len(test_sentences),
                'embedding_similarity': similarity_results,
                'text_quality': quality_results,
                'generated_texts': generated_texts[:10],  # Save first 10 for inspection
                'original_texts': test_sentences[:10]
            }
            
            return results
            
        except Exception as e:
            print(f"Error evaluating {attacker_name}: {e}")
            return None
    
    def evaluate_all_attackers(self, dataset_name, split='test'):
        """Evaluate all attackers on the dataset"""
        print(f"Evaluating all attackers on {dataset_name}...")
        
        all_results = {}
        
        # Get list of available attackers
        available_attackers = []
        for model_name in ['gpt2', 'opt', 't5']:
            for embedding_model in EMBEDDING_MODELS.keys():
                attacker_name = f"{model_name}_{embedding_model}"
                model_path = os.path.join(PATHS['models_dir'], f"attacker_{attacker_name}")
                if os.path.exists(model_path):
                    available_attackers.append(attacker_name)
        
        print(f"Found {len(available_attackers)} available attackers")
        
        # Evaluate each attacker
        for attacker_name in available_attackers:
            print(f"\nEvaluating {attacker_name}...")
            results = self.evaluate_attack_performance(attacker_name, dataset_name, split)
            
            if results:
                all_results[attacker_name] = results
                
                # Print summary
                similarity = results['embedding_similarity']['avg_similarity']
                perplexity = results['text_quality']['avg_perplexity']
                print(f"  Similarity: {similarity:.4f}")
                print(f"  Perplexity: {perplexity:.2f}")
        
        # Save results
        results_path = os.path.join(
            PATHS['results_dir'], 
            f"geia_evaluation_{dataset_name}_{self.blackbox_model_name}.json"
        )
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nResults saved to {results_path}")
        return all_results

def main():
    parser = argparse.ArgumentParser(description='GEIA-based evaluation')
    parser.add_argument('--dataset', type=str, default='sst2',
                       choices=['sst2', 'personachat', 'abcd'],
                       help='Dataset to evaluate on')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'dev', 'test'],
                       help='Dataset split')
    parser.add_argument('--blackbox_model', type=str, default='all-mpnet-base-v2',
                       help='Black-box embedding model')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = GEIAEvaluator(args.blackbox_model)
    
    # Run evaluation
    results = evaluator.evaluate_all_attackers(args.dataset, args.split)
    
    print("GEIA evaluation completed!")

if __name__ == "__main__":
    main()