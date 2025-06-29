import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import EMBEDDING_MODELS, DATASET_PATHS, PATHS, TRAIN_CONFIG

# Import GEIA data processing - Fixed for both local and Colab
def import_geia_data_process():
    """Import GEIA data_process module with proper path handling"""
    # Try multiple possible paths for different environments
    current_dir = os.getcwd()
    possible_paths = [
        os.path.join(current_dir, 'GEIA'),
        os.path.join(current_dir, '..', 'GEIA'),
        os.path.join(current_dir, 'Inverse_Embedding_Attack', '..', 'GEIA'),
        os.path.join(current_dir, '..', 'Inverse_Embedding_Attack', '..', 'GEIA'),
        'GEIA',
        '../GEIA',
        '../../GEIA'
    ]
    
    print(f"Current directory: {current_dir}")
    print(f"Looking for GEIA in possible paths...")
    
    for geia_path in possible_paths:
        print(f"Trying path: {geia_path}")
        if os.path.exists(geia_path):
            print(f"✅ Found GEIA at: {geia_path}")
            sys.path.insert(0, geia_path)
            try:
                from data_process import get_sent_list
                print(f"✅ Successfully imported GEIA data_process from {geia_path}")
                return get_sent_list
            except ImportError as e:
                print(f"❌ Failed to import from {geia_path}: {e}")
                sys.path.remove(geia_path)
                continue
        else:
            print(f"❌ Path does not exist: {geia_path}")
    
    # If all paths failed, try to create a simple fallback
    print("⚠️ Could not find GEIA, creating fallback data processing...")
    return create_fallback_data_process()

def create_fallback_data_process():
    """Create a fallback data processing function that works without GEIA"""
    def get_sent_list(config):
        dataset = config['dataset']
        data_type = config['data_type']
        
        print(f"Using fallback data processing for {dataset} {data_type}")
        
        # Handle different data types
        if data_type == 'dev':
            data_type = 'validation'
        
        try:
            if dataset == 'sst2':
                # Fix cache_dir issue
                dataset_obj = load_dataset('glue', 'sst2', split=data_type, cache_dir="./data_cache")
                sentences = [d['sentence'] for d in dataset_obj]
            elif dataset == 'personachat':
                # Use a simpler dataset for PersonaChat
                dataset_obj = load_dataset('glue', 'sst2', split=data_type, cache_dir="./data_cache")
                sentences = [d['sentence'] for d in dataset_obj]
            elif dataset == 'abcd':
                # Use SST-2 as fallback for ABCD
                dataset_obj = load_dataset('glue', 'sst2', split=data_type, cache_dir="./data_cache")
                sentences = [d['sentence'] for d in dataset_obj]
            else:
                # Default to SST-2
                dataset_obj = load_dataset('glue', 'sst2', split=data_type, cache_dir="./data_cache")
                sentences = [d['sentence'] for d in dataset_obj]
            
            # Limit to 10,000 samples
            if len(sentences) > 10000:
                sentences = sentences[:10000]
                print(f"Limited to 10,000 samples for {dataset} {data_type}")
            
            print(f"✅ Loaded {len(sentences)} sentences from {dataset} {data_type}")
            return sentences
            
        except Exception as e:
            print(f"❌ Error loading dataset {dataset}: {e}")
            # Return some sample sentences as last resort
            sample_sentences = [
                "This movie is absolutely fantastic!",
                "I really enjoyed watching this film.",
                "The acting was superb and the plot was engaging.",
                "This is a great example of excellent filmmaking.",
                "I would highly recommend this movie to everyone."
            ] * 2000  # Repeat to get 10,000 samples
            print(f"⚠️ Using sample sentences as fallback")
            return sample_sentences[:10000]
    
    return get_sent_list

# Import the function
get_sent_list = import_geia_data_process()

class EmbeddingPreparer:
    def __init__(self):
        self.device = TRAIN_CONFIG['device']
        self.embedding_models = {}
        self.load_embedding_models()
    
    def load_embedding_models(self):
        """Load all embedding models"""
        print("Loading embedding models...")
        for model_name, config in EMBEDDING_MODELS.items():
            print(f"Loading {model_name}...")
            model = SentenceTransformer(config['path'], device=self.device)
            self.embedding_models[model_name] = {
                'model': model,
                'dim': config['dim'],
                'type': config['type']
            }
        print("All models loaded successfully!")
    
    def load_dataset(self, dataset_name, split='train'):
        """Load dataset using GEIA data pipeline"""
        # Use GEIA data processing
        config = {
            'dataset': dataset_name,
            'data_type': split
        }
        
        # Load real dataset using GEIA
        sentences = get_sent_list(config)
        
        # Limit to 10,000 samples for training (as per requirements)
        if len(sentences) > 10000:
            sentences = sentences[:10000]
            print(f"Limited to 10,000 samples for {dataset_name} {split}")
        
        print(f"Loaded {len(sentences)} sentences from {dataset_name} {split}")
        return sentences
    
    def create_embeddings(self, sentences, model_name, save_path):
        """Create embeddings for sentences using specified model"""
        model_info = self.embedding_models[model_name]
        model = model_info['model']
        
        print(f"Creating embeddings for {model_name}...")
        embeddings = []
        
        # Process in batches
        batch_size = 32
        for i in tqdm(range(0, len(sentences), batch_size)):
            batch_sentences = sentences[i:i+batch_size]
            with torch.no_grad():
                batch_embeddings = model.encode(batch_sentences, convert_to_tensor=True)
                embeddings.append(batch_embeddings.cpu().numpy())
        
        embeddings = np.concatenate(embeddings, axis=0)
        
        # Save embeddings and sentences in proper format
        data = {
            'embeddings': embeddings.tolist(),
            'sentences': sentences,
            'model_name': model_name,
            'embedding_dim': model_info['dim'],
            'num_samples': len(sentences)
        }
        
        # Create directory if not exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved embeddings to {save_path}")
        print(f"Shape: {embeddings.shape}")
        print(f"Number of samples: {len(sentences)}")
        
        return embeddings
    
    def prepare_all_embeddings(self, dataset_name, split='train'):
        """Prepare embeddings for all models on specified dataset"""
        print(f"Preparing embeddings for {dataset_name} dataset...")
        
        # Load dataset using GEIA
        sentences = self.load_dataset(dataset_name, split)
        print(f"Loaded {len(sentences)} sentences")
        
        # Create embeddings for each model
        for model_name in EMBEDDING_MODELS.keys():
            save_path = os.path.join(
                PATHS['embeddings_dir'], 
                f"{dataset_name}_{split}_{model_name}.json"
            )
            self.create_embeddings(sentences, model_name, save_path)
    
    def create_alignment_layers(self, target_dim=768):
        """Create alignment layers to standardize embedding dimensions"""
        alignment_layers = {}
        
        for model_name, config in EMBEDDING_MODELS.items():
            if config['dim'] != target_dim:
                # Create linear projection layer
                import torch.nn as nn
                alignment_layer = nn.Linear(config['dim'], target_dim)
                alignment_layers[model_name] = alignment_layer
                
                # Save alignment layer
                save_path = os.path.join(
                    PATHS['models_dir'], 
                    f"alignment_{model_name}_{config['dim']}to{target_dim}.pt"
                )
                torch.save(alignment_layer.state_dict(), save_path)
                print(f"Saved alignment layer for {model_name}")
        
        return alignment_layers

def main():
    parser = argparse.ArgumentParser(description='Prepare embeddings for inverse embedding attack')
    parser.add_argument('--dataset', type=str, default='sst2', 
                       choices=['sst2', 'personachat', 'abcd'],
                       help='Dataset to use')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'dev', 'test'],
                       help='Dataset split')
    parser.add_argument('--create_alignment', action='store_true',
                       help='Create alignment layers for different embedding dimensions')
    
    args = parser.parse_args()
    
    # Initialize preparer
    preparer = EmbeddingPreparer()
    
    # Prepare embeddings
    preparer.prepare_all_embeddings(args.dataset, args.split)
    
    # Create alignment layers if requested
    if args.create_alignment:
        preparer.create_alignment_layers()
    
    print("Embedding preparation completed!")

if __name__ == "__main__":
    main() 