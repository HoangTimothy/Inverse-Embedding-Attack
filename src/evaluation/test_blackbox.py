import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import EVAL_CONFIG, PATHS, TRAIN_CONFIG

# Import local beam search
from src.utils.beam_search import beam_decode_sentence

class BlackBoxTester:
    def __init__(self, blackbox_model_name, target_dim=768):
        self.device = TRAIN_CONFIG['device']
        self.blackbox_model_name = blackbox_model_name
        self.target_dim = target_dim
        
        # Load black-box embedding model
        self.load_blackbox_model()
        
        # Load trained attackers
        self.attackers = {}
        self.load_attackers()
    
    def load_blackbox_model(self):
        """Load black-box embedding model"""
        print(f"Loading black-box model: {self.blackbox_model_name}")
        self.blackbox_model = SentenceTransformer(self.blackbox_model_name, device=self.device)
        self.blackbox_model.eval()
    
    def load_attackers(self):
        """Load all trained attackers"""
        models_dir = PATHS['models_dir']
        
        # Find all attacker models
        for item in os.listdir(models_dir):
            if item.startswith('attacker_') and os.path.isdir(os.path.join(models_dir, item)):
                attacker_path = os.path.join(models_dir, item)
                
                # Parse model info from path
                parts = item.split('_')
                if len(parts) >= 4:
                    attacker_type = parts[1]  # gpt2, opt, t5
                    embedding_model = parts[2]  # embedding model name
                    
                    print(f"Loading attacker: {attacker_type} trained on {embedding_model}")
                    
                    # Load attacker (simplified - you'll need to implement full loading)
                    self.attackers[f"{attacker_type}_{embedding_model}"] = {
                        'path': attacker_path,
                        'type': attacker_type,
                        'embedding_model': embedding_model
                    }
    
    def load_test_data(self, dataset_name, split='test'):
        """Load test data"""
        # Use sample data for testing
        if dataset_name == 'sst2':
            sample_sentences = [
                "This movie is absolutely fantastic!",
                "I really enjoyed watching this film.",
                "The acting was superb and the plot was engaging.",
                "This is one of the best movies I've ever seen.",
                "The cinematography was beautiful and the story was compelling."
            ]
        elif dataset_name == 'personachat':
            sample_sentences = [
                "Hello, how are you today?",
                "What's your favorite hobby?",
                "Do you like traveling?",
                "What's your favorite food?",
                "How was your weekend?"
            ]
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        return sample_sentences[:5]  # Use 5 sentences for testing
    
    def get_blackbox_embeddings(self, sentences):
        """Get embeddings from black-box model"""
        print("Getting embeddings from black-box model...")
        embeddings = []
        
        batch_size = 32
        for i in tqdm(range(0, len(sentences), batch_size)):
            batch_sentences = sentences[i:i+batch_size]
            with torch.no_grad():
                batch_embeddings = self.blackbox_model.encode(batch_sentences, convert_to_tensor=True)
                embeddings.append(batch_embeddings.cpu().numpy())
        
        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings
    
    def align_embeddings(self, embeddings, target_dim):
        """Align embeddings to target dimension"""
        if embeddings.shape[1] != target_dim:
            # Simple linear projection (you might want to use learned alignment)
            import torch.nn as nn
            projection = nn.Linear(embeddings.shape[1], target_dim)
            embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
            aligned_embeddings = projection(embeddings_tensor).detach().numpy()
            return aligned_embeddings
        return embeddings
    
    def generate_text(self, embeddings, attacker_info):
        """Generate text using attacker model"""
        attacker_path = attacker_info['path']
        attacker_type = attacker_info['type']
        
        # Load attacker model
        if attacker_type == 't5':
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            model = T5ForConditionalGeneration.from_pretrained(attacker_path)
            tokenizer = T5Tokenizer.from_pretrained(attacker_path)
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(attacker_path)
            tokenizer = AutoTokenizer.from_pretrained(attacker_path)
            
        model.to(self.device)
        model.eval()
        
        # Load projection if exists
        projection = None
        proj_path = os.path.join(attacker_path, "projection.pt")
        if os.path.exists(proj_path):
            import torch.nn as nn
            embedding_dim = embeddings.shape[1]
            projection = nn.Linear(embedding_dim, model.config.hidden_size)
            projection.load_state_dict(torch.load(proj_path))
            projection.to(self.device)
        
        generated_texts = []
        
        for embedding in tqdm(embeddings):
            # Convert embedding to tensor
            embedding_tensor = torch.tensor(embedding, dtype=torch.float32).to(self.device)
            
            # Apply projection if needed
            if projection is not None:
                embedding_tensor = projection(embedding_tensor)
            
            # Generate text using beam search
            try:
                config = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'device': self.device
                }
                
                generated_text = beam_decode_sentence(
                    hidden_X=embedding_tensor,
                    config=config,
                    num_generate=1,
                    beam_size=EVAL_CONFIG['beam_size']
                )
                
                if isinstance(generated_text, list):
                    generated_texts.append(generated_text[0])
                else:
                    generated_texts.append(generated_text)
                    
            except Exception as e:
                print(f"Error generating text: {e}")
                generated_texts.append("")
        
        return generated_texts
    
    def evaluate_attack(self, original_sentences, generated_sentences):
        """Evaluate attack performance"""
        from sentence_transformers import util
        
        print("Evaluating attack performance...")
        
        # Calculate embedding similarity
        original_embeddings = self.blackbox_model.encode(original_sentences, convert_to_tensor=True)
        generated_embeddings = self.blackbox_model.encode(generated_sentences, convert_to_tensor=True)
        
        cosine_scores = util.cos_sim(original_embeddings, generated_embeddings)
        similarity_scores = [cosine_scores[i][i].item() for i in range(len(original_sentences))]
        
        avg_similarity = np.mean(similarity_scores)
        
        # Calculate exact match
        exact_matches = sum(1 for orig, gen in zip(original_sentences, generated_sentences) if orig.lower() == gen.lower())
        exact_match_rate = exact_matches / len(original_sentences)
        
        # Calculate BLEU score (simplified)
        try:
            import nltk
            from nltk.translate.bleu_score import sentence_bleu
            
            bleu_scores = []
            for orig, gen in zip(original_sentences, generated_sentences):
                if gen.strip():
                    bleu = sentence_bleu([orig.split()], gen.split())
                    bleu_scores.append(bleu)
            
            avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
        except:
            avg_bleu = 0
        
        results = {
            'avg_embedding_similarity': avg_similarity,
            'exact_match_rate': exact_match_rate,
            'avg_bleu_score': avg_bleu,
            'num_samples': len(original_sentences)
        }
        
        return results
    
    def test_all_attackers(self, dataset_name, split='test'):
        """Test all attackers on black-box model"""
        print(f"Testing attackers on {dataset_name} dataset...")
        
        # Load test data
        test_sentences = self.load_test_data(dataset_name, split)
        print(f"Loaded {len(test_sentences)} test sentences")
        
        # Get black-box embeddings
        blackbox_embeddings = self.get_blackbox_embeddings(test_sentences)
        
        all_results = {}
        
        # Test each attacker
        for attacker_key, attacker_info in self.attackers.items():
            print(f"\nTesting attacker: {attacker_key}")
            
            try:
                # Generate text using attacker
                generated_texts = self.generate_text(blackbox_embeddings, attacker_info)
                
                # Debug: show some examples
                print(f"Generated {len(generated_texts)} texts")
                for i in range(min(3, len(test_sentences))):
                    print(f"Original: {test_sentences[i]}")
                    print(f"Generated: {generated_texts[i]}")
                    print("---")
                
                # Evaluate attack
                results = self.evaluate_attack(test_sentences, generated_texts)
                all_results[attacker_key] = results
                
                print(f"Results for {attacker_key}:")
                print(f"  Embedding Similarity: {results['avg_embedding_similarity']:.4f}")
                print(f"  Exact Match Rate: {results['exact_match_rate']:.4f}")
                print(f"  BLEU Score: {results['avg_bleu_score']:.4f}")
                
            except Exception as e:
                print(f"Error testing attacker {attacker_key}: {e}")
                all_results[attacker_key] = {
                    'avg_embedding_similarity': 0.0,
                    'exact_match_rate': 0.0,
                    'avg_bleu_score': 0.0,
                    'error': str(e)
                }
        
        # Save results
        results_path = os.path.join(PATHS['results_dir'], f"blackbox_test_{dataset_name}_{self.blackbox_model_name}.json")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nResults saved to {results_path}")
        print("Black-box testing completed!")
        
        return all_results

def main():
    parser = argparse.ArgumentParser(description='Test attackers on black-box model')
    parser.add_argument('--dataset', type=str, default='sst2',
                       choices=['sst2', 'personachat', 'abcd'],
                       help='Dataset to test on')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'dev', 'test'],
                       help='Dataset split')
    parser.add_argument('--blackbox_model', type=str, required=True,
                       help='Black-box embedding model to test against')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = BlackBoxTester(args.blackbox_model)
    
    # Run tests
    results = tester.test_all_attackers(args.dataset, args.split)
    
    print("Black-box testing completed!")

if __name__ == "__main__":
    main() 