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

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import ATTACKER_MODELS, TRAIN_CONFIG, PATHS, EVAL_CONFIG

current_dir = os.path.dirname(__file__)
sys.path.insert(0, current_dir)

try:
    from attacker_models import SequenceCrossEntropyLoss
    from decode_beam_search import beam_decode_sentence
    print("Successfully imported local modules")
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print(f"Current directory: {current_dir}")
    print(f"Files in directory: {os.listdir(current_dir)}")
    raise

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
        
        self.load_attacker_model()
        
        target_dim = self.model.config.hidden_size
        
        self.projection = None
        if embedding_dim != target_dim:
            self.projection = LinearProjection(embedding_dim, target_dim)
            self.projection.to(self.device)
            print(f"Created projection layer: {embedding_dim} -> {target_dim}")
    
    def load_attacker_model(self):
        model_config = ATTACKER_MODELS[self.attacker_model_name]
        
        print(f"Loading attacker model: {self.attacker_model_name}")
        
        if self.attacker_model_name == 'gpt2':
            model_path = 'microsoft/DialoGPT-medium'
        else:
            model_path = model_config['path']
        
        if self.attacker_model_name == 't5':
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print(f"Model type: {type(self.model).__name__}")
        print(f"Model hidden size: {self.model.config.hidden_size}")
        print(f"Model vocab size: {self.model.config.vocab_size}")
        
        if hasattr(self.model, 'transformer'):
            print("Model has 'transformer' attribute (GPT-2 style)")
        elif hasattr(self.model, 'model'):
            print("Model has 'model' attribute (OPT style)")
        else:
            print("Model has different structure (T5 style)")
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(self.device)
        self.model.train()
    
    def load_embedding_data(self, dataset_name, split='train'):
        embedding_path = os.path.join(
            PATHS['embeddings_dir'],
            f"{dataset_name}_{split}_{self.embedding_model_name}.json"
        )
        
        if not os.path.exists(embedding_path):
            print(f"Embedding file not found: {embedding_path}")
            print("Please run prepare_embeddings.py first to create embedding datasets")
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
        
        with open(embedding_path, 'r') as f:
            data = json.load(f)
        
        embeddings = torch.tensor(data['embeddings'], dtype=torch.float32)
        sentences = data['sentences']
        
        print(f"Loaded {len(sentences)} samples from {embedding_path}")
        print(f"Embedding shape: {embeddings.shape}")
        
        return embeddings, sentences
    
    def train_on_batch(self, batch_embeddings, batch_sentences):
        batch_embeddings = batch_embeddings.to(self.device)
        
        inputs = self.tokenizer(
            batch_sentences, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True, 
            max_length=TRAIN_CONFIG['max_length']
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        labels = input_ids.clone()
        
        if self.attacker_model_name == 't5':
            return self._train_t5_batch(batch_embeddings, input_ids, labels)
        else:
            return self._train_causal_batch(batch_embeddings, input_ids, labels)
    
    def _train_causal_batch(self, batch_embeddings, input_ids, labels):
        if hasattr(self.model, 'transformer'):
            input_emb = self.model.transformer.wte(input_ids)
        elif hasattr(self.model, 'model'):
            input_emb = self.model.model.decoder.embed_tokens(input_ids)
        
        if self.projection is not None:
            batch_embeddings = self.projection(batch_embeddings)
        
        batch_embeddings_unsqueeze = torch.unsqueeze(batch_embeddings, 1)
        inputs_embeds = torch.cat((batch_embeddings_unsqueeze, input_emb), dim=1)
        
        outputs = self.model(inputs_embeds=inputs_embeds, return_dict=True)
        logits = outputs.logits[:, :-1].contiguous()
        target = labels.contiguous()
        
        criterion = SequenceCrossEntropyLoss()
        target_mask = torch.ones_like(target).float().to(self.device)
        loss = criterion(logits, target, target_mask, label_smoothing=0.02, reduce="batch")
        
        return loss
    
    def _train_t5_batch(self, batch_embeddings, input_ids, labels):
        if self.projection is not None:
            batch_embeddings = self.projection(batch_embeddings)
        
        batch_size = batch_embeddings.shape[0]
        
        outputs = self.model(input_ids=input_ids, labels=labels, return_dict=True)
        loss = outputs.loss
        
        return loss
    
    def train(self, dataset_name, split='train'):
        print(f"Training attacker {self.attacker_model_name} on {self.embedding_model_name} embeddings...")
        
        embeddings, sentences = self.load_embedding_data(dataset_name, split)
        dataset = EmbeddingDataset(embeddings, sentences)
        dataloader = DataLoader(
            dataset, 
            batch_size=TRAIN_CONFIG['batch_size'], 
            shuffle=True,
            collate_fn=dataset.collate
        )
        
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'ln', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=TRAIN_CONFIG['learning_rate'], eps=1e-06)
        
        if self.projection is not None:
            optimizer.add_param_group({'params': self.projection.parameters()})
        
        num_train_steps = len(dataloader) * TRAIN_CONFIG['num_epochs']
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=100, 
            num_training_steps=num_train_steps
        )
        
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
            
            self.save_model(epoch)
    
    def save_model(self, epoch=None):
        model_name = f"attacker_{self.attacker_model_name}_{self.embedding_model_name}"
        if epoch is not None:
            model_name += f"_epoch_{epoch}"
        
        save_path = os.path.join(PATHS['models_dir'], model_name)
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        if self.projection is not None:
            proj_path = os.path.join(save_path, "projection.pt")
            torch.save(self.projection.state_dict(), proj_path)
        
        print(f"Model saved to {save_path}")
    
    def load_model(self, model_path):
        if self.attacker_model_name == 't5':
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
        self.model.to(self.device)
        
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
    
    from config import EMBEDDING_MODELS
    embedding_dim = EMBEDDING_MODELS[args.embedding_model]['dim']
    
    attacker = InverseEmbeddingAttacker(
        args.attacker, 
        args.embedding_model, 
        embedding_dim
    )
    
    attacker.train(args.dataset, args.split)
    
    print("Training completed!")

if __name__ == "__main__":
    main() 