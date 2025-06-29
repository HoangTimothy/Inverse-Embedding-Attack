import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import ATTACKER_MODELS, TRAIN_CONFIG, PATHS, EVAL_CONFIG

# Import from local files
from attacker_models import SequenceCrossEntropyLoss
from decode_beam_search import beam_decode_sentence

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, sentences):
        self.embeddings = embeddings
        self.sentences = sentences
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.sentences[idx]
    
    def collate(self, batch):
        embeddings, sentences = zip(*batch)
        return torch.stack(embeddings), sentences

class LinearProjection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearProjection, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        return self.fc(x)

class InverseEmbeddingAttacker:
    def __init__(self, attacker_model_name, embedding_model_name, embedding_dim):
        self.device = TRAIN_CONFIG['device']
        self.attacker_model_name = attacker_model_name
        self.embedding_model_name = embedding_model_name
        self.embedding_dim = embedding_dim
        
        # Load attacker model first
        self.load_attacker_model()
        
        # Get the correct embedding dimension for the attacker model
        target_dim = self.model.config.hidden_size  # This will be 1024 for GPT-2
        
        # Initialize projection layer if needed
        self.projection = None
        if embedding_dim != target_dim:
            self.projection = LinearProjection(embedding_dim, target_dim)
            self.projection.to(self.device)
            print(f"Created projection layer: {embedding_dim} -> {target_dim}")
    
    def load_attacker_model(self):
        """Load attacker model from GEIA framework"""
        model_config = ATTACKER_MODELS[self.attacker_model_name]
        
        print(f"Loading attacker model: {self.attacker_model_name}")
        
        # Use smaller models for Colab compatibility
        if self.attacker_model_name == 'gpt2':
            model_path = 'microsoft/DialoGPT-medium'  # Use medium instead of large
        else:
            model_path = model_config['path']
            
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Debug: print model info
        print(f"Model type: {type(self.model).__name__}")
        print(f"Model hidden size: {self.model.config.hidden_size}")
        print(f"Model vocab size: {self.model.config.vocab_size}")
        
        # Check model structure
        if hasattr(self.model, 'transformer'):
            print("Model has 'transformer' attribute (GPT-2 style)")
        elif hasattr(self.model, 'model'):
            print("Model has 'model' attribute (OPT style)")
        else:
            print("Model has different structure (T5 style)")
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(self.device)
        self.model.train()
    
    def load_embedding_data(self, dataset_name, split='train'):
        """Load embedding data for training"""
        embedding_path = os.path.join(
            PATHS['embeddings_dir'],
            f"{dataset_name}_{split}_{self.embedding_model_name}.json"
        )
        
        with open(embedding_path, 'r') as f:
            data = json.load(f)
        
        embeddings = torch.tensor(data['embeddings'], dtype=torch.float32)
        sentences = data['sentences']
        
        return embeddings, sentences
    
    def train_on_batch(self, batch_embeddings, batch_sentences):
        """Training step for one batch"""
        # Move embeddings to device first
        batch_embeddings = batch_embeddings.to(self.device)
        
        # Tokenize sentences
        inputs = self.tokenizer(
            batch_sentences, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True, 
            max_length=TRAIN_CONFIG['max_length']
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        labels = input_ids.clone()
        
        # Get embeddings from attacker model (handle different model types)
        if hasattr(self.model, 'transformer'):
            # GPT-2 style models
            input_emb = self.model.transformer.wte(input_ids)
        elif hasattr(self.model, 'model'):
            # OPT style models
            input_emb = self.model.model.decoder.embed_tokens(input_ids)
        else:
            # T5 style models
            input_emb = self.model.shared(input_ids)
        
        # Apply projection if needed
        if self.projection is not None:
            batch_embeddings = self.projection(batch_embeddings)
        
        # Concatenate embeddings with input embeddings
        batch_embeddings_unsqueeze = torch.unsqueeze(batch_embeddings, 1)
        inputs_embeds = torch.cat((batch_embeddings_unsqueeze, input_emb), dim=1)
        
        # Forward pass
        outputs = self.model(inputs_embeds=inputs_embeds, return_dict=True)
        logits = outputs.logits[:, :-1].contiguous()
        target = labels.contiguous()
        
        # Calculate loss
        criterion = SequenceCrossEntropyLoss()
        target_mask = torch.ones_like(target).float().to(self.device)
        loss = criterion(logits, target, target_mask, label_smoothing=0.02, reduce="batch")
        
        return loss
    
    def train(self, dataset_name, split='train'):
        """Train the attacker model"""
        print(f"Training attacker {self.attacker_model_name} on {self.embedding_model_name} embeddings...")
        
        # Load data
        embeddings, sentences = self.load_embedding_data(dataset_name, split)
        dataset = EmbeddingDataset(embeddings, sentences)
        dataloader = DataLoader(
            dataset, 
            batch_size=TRAIN_CONFIG['batch_size'], 
            shuffle=True,
            collate_fn=dataset.collate
        )
        
        # Setup optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'ln', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=TRAIN_CONFIG['learning_rate'], eps=1e-06)
        
        # Add projection parameters to optimizer if exists
        if self.projection is not None:
            optimizer.add_param_group({'params': self.projection.parameters()})
        
        # Setup scheduler
        num_train_steps = len(dataloader) * TRAIN_CONFIG['num_epochs']
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=100, 
            num_training_steps=num_train_steps
        )
        
        # Training loop
        for epoch in range(TRAIN_CONFIG['num_epochs']):
            total_loss = 0
            self.model.train()
            
            for batch_idx, (batch_embeddings, batch_sentences) in enumerate(tqdm(dataloader)):
                optimizer.zero_grad()
                
                loss = self.train_on_batch(batch_embeddings, batch_sentences)
                loss.backward()
                
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
            
            # Save checkpoint
            self.save_model(epoch)
    
    def save_model(self, epoch=None):
        """Save trained model"""
        model_name = f"attacker_{self.attacker_model_name}_{self.embedding_model_name}"
        if epoch is not None:
            model_name += f"_epoch_{epoch}"
        
        save_path = os.path.join(PATHS['models_dir'], model_name)
        
        # Save attacker model
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save projection if exists
        if self.projection is not None:
            proj_path = os.path.join(save_path, "projection.pt")
            torch.save(self.projection.state_dict(), proj_path)
        
        print(f"Model saved to {save_path}")
    
    def load_model(self, model_path):
        """Load trained model"""
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        
        # Load projection if exists
        proj_path = os.path.join(model_path, "projection.pt")
        if os.path.exists(proj_path):
            self.projection = LinearProjection(self.embedding_dim, self.model.config.hidden_size)
            self.projection.load_state_dict(torch.load(proj_path))
            self.projection.to(self.device)
        
        self.model.eval()

def main():
    parser = argparse.ArgumentParser(description='Train inverse embedding attackers')
    parser.add_argument('--dataset', type=str, default='sst2', 
                       choices=['sst2', 'personachat', 'abcd'],
                       help='Dataset to use')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'dev', 'test'],
                       help='Dataset split')
    parser.add_argument('--attacker', type=str, default='gpt2',
                       choices=['gpt2', 'opt', 't5'],
                       help='Attacker model type')
    parser.add_argument('--embedding_model', type=str, required=True,
                       choices=['all-mpnet-base-v2', 'stsb-roberta-base', 'all-MiniLM-L6-v2', 'paraphrase-MiniLM-L6-v2'],
                       help='Embedding model to attack')
    
    args = parser.parse_args()
    
    # Get embedding dimension
    from config import EMBEDDING_MODELS
    embedding_dim = EMBEDDING_MODELS[args.embedding_model]['dim']
    
    # Initialize attacker
    attacker = InverseEmbeddingAttacker(
        args.attacker, 
        args.embedding_model, 
        embedding_dim
    )
    
    # Train attacker
    attacker.train(args.dataset, args.split)
    
    print("Training completed!")

if __name__ == "__main__":
    main() 