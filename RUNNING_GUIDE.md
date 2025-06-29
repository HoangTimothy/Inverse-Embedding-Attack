# Hướng dẫn chạy Inverse Embedding Attack

## 🚀 Quick Start

### Bước 1: Kiểm tra môi trường
```bash
# Test các components cơ bản
python quick_test.py
```

### Bước 2: Test chi tiết
```bash
# Test từng component riêng biệt
python test_simple.py
```

### Bước 3: Chạy demo
```bash
# Chạy demo để xem flow
python demo.py
```

## 🔧 Troubleshooting

### Lỗi thường gặp:

#### 1. Lỗi dataset loading
```
ValueError: Invalid pattern: '**' can only be an entire path component
```

**Giải pháp:**
- Sử dụng cache_dir: `cache_dir="./data_cache"`
- Hoặc dùng sample data trong code

#### 2. Lỗi import AdamW
```
ImportError: cannot import name 'AdamW' from 'transformers'
```

**Giải pháp:**
- Đã sửa trong code: `from torch.optim import AdamW`

#### 3. Lỗi missing attacker_models.py
```
ModuleNotFoundError: No module named 'attacker_models'
```

**Giải pháp:**
- File đã được tạo trong `src/attackers/attacker_models.py`

#### 4. Lỗi Keras version
```
Your currently installed version of Keras is Keras 3, but this is not yet supported in Transformers
```

**Giải pháp:**
```bash
pip install tf-keras
```

## 📋 Các bước chạy đầy đủ

### 1. Chuẩn bị data (Optional - có thể skip)
```bash
# Nếu muốn dùng real dataset
python src/data_processing/prepare_embeddings.py --dataset sst2 --split train
python src/data_processing/prepare_embeddings.py --dataset sst2 --split dev
python src/data_processing/prepare_embeddings.py --dataset sst2 --split test
```

### 2. Train attackers (Optional - có thể skip)
```bash
# Train GPT-2 trên all-mpnet-base-v2
python src/attackers/train_attackers.py --dataset sst2 --split train --attacker gpt2 --embedding_model all-mpnet-base-v2

# Train OPT trên stsb-roberta-base
python src/attackers/train_attackers.py --dataset sst2 --split train --attacker opt --embedding_model stsb-roberta-base

# Train T5 trên all-MiniLM-L6-v2
python src/attackers/train_attackers.py --dataset sst2 --split train --attacker t5 --embedding_model all-MiniLM-L6-v2
```

### 3. Test trên black-box (Optional - có thể skip)
```bash
python src/evaluation/test_blackbox.py --dataset sst2 --split test --blackbox_model paraphrase-MiniLM-L6-v2
```

### 4. Chạy toàn bộ experiment
```bash
# Chạy tất cả các bước
python run_experiment.py --dataset sst2 --blackbox_model all-mpnet-base-v2

# Skip data preparation nếu đã có
python run_experiment.py --dataset sst2 --blackbox_model all-mpnet-base-v2 --skip_prepare

# Skip training nếu đã có models
python run_experiment.py --dataset sst2 --blackbox_model all-mpnet-base-v2 --skip_prepare --skip_train
```

## 🎯 Test với Google Colab

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

## 📊 Expected Results

### Demo Results:
- Embedding extraction: ✅
- Alignment: ✅
- Attack simulation: ✅
- Complete flow: ✅

### Full Experiment Results:
- Embedding similarity: 0.7-0.9
- Exact match rate: 0.1-0.3
- BLEU score: 0.5-0.8

## 🔍 Debug Tips

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

## 📁 File Structure
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

## 🆘 Need Help?

Nếu gặp lỗi, hãy:

1. Chạy `python quick_test.py` để kiểm tra components
2. Chạy `python test_simple.py` để test chi tiết
3. Kiểm tra logs và error messages
4. Đảm bảo đã cài đúng dependencies
5. Kiểm tra GPU/CPU compatibility 