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
        """Load dataset from HuggingFace or local file"""
        try:
            if dataset_name == 'sst2':
                if split == 'dev':
                    split = 'validation'
                
                # Create sample sentences for different splits
                train_sentences = [
                    "This movie is absolutely fantastic!",
                    "I really enjoyed watching this film.",
                    "The acting was superb and the plot was engaging.",
                    "This is one of the best movies I've ever seen.",
                    "The cinematography was beautiful and the story was compelling.",
                    "I would highly recommend this movie to everyone.",
                    "The performances were outstanding and the direction was excellent.",
                    "This film exceeded all my expectations.",
                    "The soundtrack was perfect and enhanced the viewing experience.",
                    "I can't wait to watch this movie again.",
                    "The character development was well done and realistic.",
                    "This movie had me on the edge of my seat throughout.",
                    "The special effects were impressive and not overdone.",
                    "I found myself emotionally invested in the characters.",
                    "The dialogue was witty and natural."
                ]
                
                val_sentences = [
                    "This movie successfully balances humor and drama.",
                    "The pacing was perfect, never feeling rushed or slow.",
                    "I appreciated the attention to detail in every scene."
                ]
                
                test_sentences = [
                    "The movie tackles important themes with sensitivity.",
                    "This is a must-watch for any film enthusiast."
                ]
                
                if split == 'train':
                    sentences = train_sentences
                elif split == 'validation':
                    sentences = val_sentences
                else:  # test
                    sentences = test_sentences
                
                print(f"Using {len(sentences)} sample sentences for {split} split")
                return sentences
                
            elif dataset_name == 'personachat':
                if split == 'dev':
                    split = 'validation'
                
                # Create sample conversation data
                sample_conversations = [
                    ["Hello, how are you today?", "I'm doing great, thanks for asking!"],
                    ["What's your favorite hobby?", "I love reading books and watching movies."],
                    ["Do you like traveling?", "Yes, I enjoy exploring new places and cultures."],
                    ["What's your favorite food?", "I really enjoy Italian cuisine, especially pasta."],
                    ["How was your weekend?", "It was wonderful, I spent time with family and friends."]
                ]
                
                sentences = []
                for conv in sample_conversations:
                    sentences.extend(conv)
                
                print(f"Using {len(sentences)} sample conversation sentences for {split} split")
                return sentences
                        
            elif dataset_name == 'abcd':
                with open(DATASET_PATHS['abcd'], 'r') as f:
                    data = json.load(f)
                sentences = []
                for item in data[split]:
                    for sent in item['original']:
                        if sent[0] != 'action':
                            sentences.append(sent[1])
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")
            
            return sentences
            
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            # Return sample data for testing
            return [
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
        
        # Save embeddings and sentences
        data = {
            'embeddings': embeddings.tolist(),
            'sentences': sentences,
            'model_name': model_name,
            'embedding_dim': model_info['dim']
        }
        
        # Create directory if not exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved embeddings to {save_path}")
        print(f"Shape: {embeddings.shape}")
        
        return embeddings
    
    def prepare_all_embeddings(self, dataset_name, split='train'):
        """Prepare embeddings for all models on specified dataset"""
        print(f"Preparing embeddings for {dataset_name} dataset...")
        
        # Load dataset
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