import os
import torch

# Dataset paths
DATASET_PATHS = {
    'sst2': 'data/sst2',
    'personachat': 'data/personachat', 
    'abcd': 'data/abcd/abcd_v1.1.json'
}

# Embedding models configuration
EMBEDDING_MODELS = {
    'all-mpnet-base-v2': {
        'path': 'all-mpnet-base-v2',
        'dim': 768,
        'type': 'sentence_transformers'
    },
    'stsb-roberta-base': {
        'path': 'stsb-roberta-base',
        'dim': 768, 
        'type': 'sentence_transformers'
    },
    'all-MiniLM-L6-v2': {
        'path': 'all-MiniLM-L6-v2',
        'dim': 384,
        'type': 'sentence_transformers'
    },
    'paraphrase-MiniLM-L6-v2': {
        'path': 'paraphrase-MiniLM-L6-v2',
        'dim': 384,
        'type': 'sentence_transformers'
    }
}

# Attacker models configuration
ATTACKER_MODELS = {
    'gpt2': {
        'path': 'microsoft/DialoGPT-medium',
        'type': 'causal_lm'
    },
    'opt': {
        'path': 'facebook/opt-125m',
        'type': 'causal_lm'
    },
    't5': {
        'path': 't5-small',
        'type': 'seq2seq'
    }
}

# Training configuration
TRAIN_CONFIG = {
    'batch_size': 8,
    'num_epochs': 5,
    'learning_rate': 3e-5,
    'max_length': 40,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'save_dir': 'models/',
    'log_dir': 'logs/'
}

# Evaluation configuration
EVAL_CONFIG = {
    'beam_size': 5,
    'max_length': 50,
    'temperature': 0.9,
    'top_p': 0.9,
    'top_k': -1
}

# Paths
PATHS = {
    'data_dir': 'data/',
    'embeddings_dir': 'data/embeddings/',
    'models_dir': 'models/',
    'logs_dir': 'logs/',
    'results_dir': 'experiments/results/'
}

# Create directories if not exist
for path in PATHS.values():
    os.makedirs(path, exist_ok=True) 