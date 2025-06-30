# Inverse Embedding Attack

## Realistic Inverse Embedding Attack Solution

### Main Idea
Instead of training attackers directly on black-box models (unrealistic), we:
1. Train attackers on similar embedding models
2. Use transfer learning to attack black-box models
3. No need for original black-box model data

### Main Flow
```
Dataset → Multiple Embedding Models → Train Attackers → Test on Black-box
```

### Project Structure
```
Inverse_Embedding_Attack/
├── src/
│   ├── data_processing/     # Dataset processing and embedding creation
│   ├── attackers/          # Attacker models
│   ├── evaluation/         # Result evaluation
│   └── utils/             # Utilities and helpers
├── data/                  # Datasets and embeddings
├── models/               # Trained models
├── experiments/          # Configs and results
└── configs/             # Configuration files
```

### Supported Models
- **Embedding Models**: all-mpnet-base-v2, stsb-roberta-base, all-MiniLM-L6-v2, paraphrase-MiniLM-L6-v2
- **Attacker Models**: GPT-2, OPT, T5 (from GEIA)
- **Datasets**: SST-2, PersonaChat, ABCD

### Usage
```bash
# 1. Prepare data and embeddings
python src/data_processing/prepare_embeddings.py

# 2. Train attackers
python src/attackers/train_attackers.py

# 3. Test on black-box
python src/evaluation/test_blackbox.py
``` 