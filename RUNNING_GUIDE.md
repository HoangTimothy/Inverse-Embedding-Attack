# Inverse Embedding Attack - Running Guide

## Project Execution Order

### **Step 1: Environment Setup**
```bash
# Install dependencies
pip install -r requirements.txt

# Check GPU (if available)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### **Step 2: Prepare Datasets and Embeddings**
```bash
# Prepare all datasets (SST-2, PersonaChat, ABCD)
# Each dataset will have 10,000 samples
python prepare_all_datasets.py
```

**Result:** Creates 4 embedding datasets with 10,000 samples each:
- `sst2_train_all-mpnet-base-v2.json`
- `personachat_train_stsb-roberta-base.json` 
- `abcd_train_all-MiniLM-L6-v2.json`
- `sst2_train_paraphrase-MiniLM-L6-v2.json`

### **Step 3: Train All Attackers**
```bash
# Train 3 attackers on 3 different datasets
python train_all_attackers.py
```

**Result:** Train 3 attackers:
- GPT-2 attacker on SST-2 dataset
- OPT attacker on PersonaChat dataset  
- T5 attacker on ABCD dataset

### **Step 4: Verify Trained Models**
```bash
# Verify all models were trained successfully
python train_all_attackers.py --verify-only
```

### **Step 5: Run Evaluation**
```bash
# Test attackers on black-box model
python src/evaluation/test_blackbox.py --dataset sst2 --split test --blackbox_model all-mpnet-base-v2
```

### **Step 6: View Results**
```bash
# Check results in experiments/results/ directory
ls experiments/results/
```

---

## Step Details

### **Step 1: Environment Setup**

**Purpose:** Install all necessary dependencies

**Check:**
- Python 3.8+
- PyTorch with CUDA (if GPU available)
- Transformers library
- Sentence Transformers
- All dependencies in requirements.txt

### **Step 2: Dataset Preparation**

**Purpose:** Create 4 embedding datasets with 10,000 samples each

**Process:**
1. Load datasets from HuggingFace (SST-2, PersonaChat, ABCD)
2. Extract 10,000 samples from each dataset
3. Create embeddings using 4 different embedding models
4. Save embeddings to JSON files

**Expected Result:**
```
data/embeddings/
├── sst2_train_all-mpnet-base-v2.json (10,000 samples)
├── personachat_train_stsb-roberta-base.json (10,000 samples)
├── abcd_train_all-MiniLM-L6-v2.json (10,000 samples)
└── sst2_train_paraphrase-MiniLM-L6-v2.json (10,000 samples)
```

### **Step 3: Train Attackers**

**Purpose:** Train 3 attacker models on 3 different datasets

**Training Configuration:**
- **GPT-2 attacker:** Train on SST-2 dataset
- **OPT attacker:** Train on PersonaChat dataset
- **T5 attacker:** Train on ABCD dataset

**Training Parameters:**
- Epochs: 5
- Batch size: 8
- Learning rate: 2e-5
- Device: CUDA (if available) or CPU

**Expected Result:**
```
models/
├── attacker_gpt2_all-mpnet-base-v2/
├── attacker_opt_stsb-roberta-base/
└── attacker_t5_all-MiniLM-L6-v2/
```

### **Step 4: Verify Models**

**Purpose:** Check all models were trained successfully

**Verification:**
- Model files exist
- Projection layers saved
- Tokenizers saved
- Config files saved

### **Step 5: Evaluation**

**Purpose:** Test attackers on black-box model

**Process:**
1. Load trained attackers
2. Load test dataset
3. Generate text from embeddings
4. Calculate similarity with original text
5. Evaluate text quality

**Metrics:**
- Embedding similarity (cosine similarity)
- Text quality (length, diversity)
- Semantic similarity

### **Step 6: Results**

**Purpose:** View and analyze results

**Result Files:**
```
experiments/results/
├── blackbox_attack_results.json
├── similarity_scores.json
└── text_quality_metrics.json
```

---

## Important Notes

### **Execution Time:**
- **Step 2:** ~30-60 minutes (depending on GPU)
- **Step 3:** ~2-4 hours (depending on GPU and model size)
- **Step 5:** ~15-30 minutes

### **Hardware Requirements:**
- **RAM:** Minimum 16GB
- **GPU:** Recommended (8GB+ VRAM)
- **Storage:** Minimum 10GB free space

### **Error Handling:**
- If GPU out of memory: Reduce batch_size
- If training slow: Increase learning_rate
- If dataset loading fails: Check internet connection

---

## Troubleshooting

### **Common Issues:**

1. **CUDA out of memory:**
   ```bash
   # Reduce batch size in config.py
   TRAIN_CONFIG['batch_size'] = 4  # Instead of 8
   ```

2. **Dataset loading failed:**
   ```bash
   # Check internet connection
   # Retry prepare_all_datasets.py
   ```

3. **Model loading failed:**
   ```bash
   # Check model path
   # Ensure model was trained successfully
   ```

### **Progress Monitoring:**
```bash
# View logs
tail -f logs/training.log

# Check GPU usage
nvidia-smi
```

---

## Expected Results

### **After Completion:**

1. **4 embedding datasets** with 10,000 samples each
2. **3 trained attackers** (GPT-2, OPT, T5)
3. **Black-box evaluation results** with similarity scores
4. **Complete research pipeline** ready for publication

### **Success Metrics:**
- ✅ All datasets created successfully
- ✅ All attackers trained successfully  
- ✅ Similarity scores > 0.7 (good)
- ✅ Text quality metrics meet standard

---

## Next Steps

After completing pipeline:

1. **Analyze results** - Analyze results in detail
2. **Compare with baselines** - Compare with baseline methods
3. **Write paper** - Write research paper
4. **Submit to conference** - Submit to conference/journal

---

**🎉 Good luck with research project!**

## Test with Google Colab

### 1. Clone repository
```python
!git clone <your-repo-url>
%cd Inverse-Embedding-Attack
```

### 2. Install dependencies
```python
!pip install -r requirements.txt
!pip install tf-keras
```

### 3. Run quick test
```python
!python quick_test.py
```

### 4. Run full experiment
```python
!python run_experiment.py --dataset sst2 --blackbox_model all-mpnet-base-v2
```

## Expected Results

### Demo Results:
- Embedding extraction: ✅
- Alignment: ✅
- Attack simulation: ✅
- Complete flow: ✅

### Full Experiment Results:
- Embedding similarity: 0.7-0.9
- Exact match rate: 0.1-0.3
- BLEU score: 0.5-0.8

## Debug Tips

### 1. Check GPU availability
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
```

### 2. Check model loading
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-mpnet-base-v2')
print("Model loaded successfully")
```

### 3. Check data loading
```python
from datasets import load_dataset
dataset = load_dataset('glue', 'sst2', split='train[:10]')
print(f"Loaded {len(dataset)} samples")
```

## File Structure
```
Inverse_Embedding_Attack/
├── config.py                    # Configuration
├── demo.py                      # Demo script
├── quick_test.py               # Quick test
├── test_simple.py              # Detailed test
├── run_experiment.py           # Full experiment
├── requirements.txt            # Dependencies
├── RUNNING_GUIDE.md           # This file
└── src/
    ├── data_processing/
    │   └── prepare_embeddings.py
    ├── attackers/
    │   ├── train_attackers.py
    │   ├── attacker_models.py
    │   └── decode_beam_search.py
    ├── evaluation/
    │   └── test_blackbox.py
    └── utils/
        └── alignment.py
```

## Need Help?

If you encounter issues, please:

1. Run `python quick_test.py` to test components
2. Run `python test_simple.py` to test in detail
3. Check logs and error messages
4. Ensure you have installed the correct dependencies
5. Check GPU/CPU compatibility 