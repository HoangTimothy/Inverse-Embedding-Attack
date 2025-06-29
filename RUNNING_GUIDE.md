# Inverse Embedding Attack - Running Guide

## 🚀 Thứ tự chạy project

### **Bước 1: Chuẩn bị môi trường**
```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Kiểm tra GPU (nếu có)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### **Bước 2: Chuẩn bị datasets và embeddings**
```bash
# Chuẩn bị tất cả datasets (SST-2, PersonaChat, ABCD)
# Mỗi dataset sẽ có 10,000 samples
python prepare_all_datasets.py
```

**Kết quả:** Tạo ra 4 embedding datasets với 10,000 samples mỗi dataset:
- `sst2_train_all-mpnet-base-v2.json`
- `personachat_train_stsb-roberta-base.json` 
- `abcd_train_all-MiniLM-L6-v2.json`
- `sst2_train_paraphrase-MiniLM-L6-v2.json`

### **Bước 3: Train tất cả attackers**
```bash
# Train 3 attackers trên 3 datasets khác nhau
python train_all_attackers.py
```

**Kết quả:** Train 3 attackers:
- GPT-2 attacker trên SST-2 dataset
- OPT attacker trên PersonaChat dataset  
- T5 attacker trên ABCD dataset

### **Bước 4: Kiểm tra models đã train**
```bash
# Verify tất cả models đã được train thành công
python train_all_attackers.py --verify-only
```

### **Bước 5: Chạy evaluation**
```bash
# Test attackers trên black-box model
python src/evaluation/test_blackbox.py --dataset sst2 --split test --blackbox_model all-mpnet-base-v2
```

### **Bước 6: Xem kết quả**
```bash
# Kiểm tra kết quả trong thư mục experiments/results/
ls experiments/results/
```

---

## 📋 Chi tiết từng bước

### **Bước 1: Chuẩn bị môi trường**

**Mục đích:** Cài đặt tất cả dependencies cần thiết

**Kiểm tra:**
- ✅ Python 3.8+
- ✅ PyTorch với CUDA (nếu có GPU)
- ✅ Transformers library
- ✅ Sentence Transformers
- ✅ Tất cả dependencies trong requirements.txt

### **Bước 2: Chuẩn bị datasets**

**Mục đích:** Tạo 4 embedding datasets với 10,000 samples mỗi dataset

**Quá trình:**
1. Load datasets từ HuggingFace (SST-2, PersonaChat, ABCD)
2. Lấy 10,000 samples từ mỗi dataset
3. Tạo embeddings bằng 4 embedding models khác nhau
4. Lưu embeddings vào file JSON

**Kết quả mong đợi:**
```
data/embeddings/
├── sst2_train_all-mpnet-base-v2.json (10,000 samples)
├── personachat_train_stsb-roberta-base.json (10,000 samples)
├── abcd_train_all-MiniLM-L6-v2.json (10,000 samples)
└── sst2_train_paraphrase-MiniLM-L6-v2.json (10,000 samples)
```

### **Bước 3: Train attackers**

**Mục đích:** Train 3 attacker models trên 3 datasets khác nhau

**Cấu hình training:**
- **GPT-2 attacker:** Train trên SST-2 dataset
- **OPT attacker:** Train trên PersonaChat dataset
- **T5 attacker:** Train trên ABCD dataset

**Tham số training:**
- Epochs: 5
- Batch size: 8
- Learning rate: 2e-5
- Device: CUDA (nếu có) hoặc CPU

**Kết quả mong đợi:**
```
models/
├── attacker_gpt2_all-mpnet-base-v2/
├── attacker_opt_stsb-roberta-base/
└── attacker_t5_all-MiniLM-L6-v2/
```

### **Bước 4: Verify models**

**Mục đích:** Kiểm tra tất cả models đã được train thành công

**Kiểm tra:**
- ✅ Model files tồn tại
- ✅ Projection layers được lưu
- ✅ Tokenizers được lưu
- ✅ Config files được lưu

### **Bước 5: Evaluation**

**Mục đích:** Test attackers trên black-box model

**Quá trình:**
1. Load trained attackers
2. Load test dataset
3. Generate text từ embeddings
4. Tính similarity với original text
5. Đánh giá chất lượng text

**Metrics:**
- Embedding similarity (cosine similarity)
- Text quality (length, diversity)
- Semantic similarity

### **Bước 6: Kết quả**

**Mục đích:** Xem và phân tích kết quả

**Files kết quả:**
```
experiments/results/
├── blackbox_attack_results.json
├── similarity_scores.json
└── text_quality_metrics.json
```

---

## ⚠️ Lưu ý quan trọng

### **Thời gian chạy:**
- **Bước 2:** ~30-60 phút (tùy GPU)
- **Bước 3:** ~2-4 giờ (tùy GPU và model size)
- **Bước 5:** ~15-30 phút

### **Yêu cầu phần cứng:**
- **RAM:** Tối thiểu 16GB
- **GPU:** Khuyến nghị có GPU (8GB+ VRAM)
- **Storage:** Tối thiểu 10GB free space

### **Xử lý lỗi:**
- Nếu GPU out of memory: Giảm batch_size
- Nếu training chậm: Tăng learning_rate
- Nếu dataset loading lỗi: Kiểm tra internet connection

---

## 🔧 Troubleshooting

### **Lỗi thường gặp:**

1. **CUDA out of memory:**
   ```bash
   # Giảm batch size trong config.py
   TRAIN_CONFIG['batch_size'] = 4  # Thay vì 8
   ```

2. **Dataset loading failed:**
   ```bash
   # Kiểm tra internet connection
   # Thử lại prepare_all_datasets.py
   ```

3. **Model loading failed:**
   ```bash
   # Kiểm tra model path
   # Đảm bảo model đã được train thành công
   ```

### **Kiểm tra tiến trình:**
```bash
# Xem logs
tail -f logs/training.log

# Kiểm tra GPU usage
nvidia-smi

# Kiểm tra disk space
df -h
```

---

## 📊 Expected Results

### **Sau khi hoàn thành:**

1. **4 embedding datasets** với 10,000 samples mỗi dataset
2. **3 trained attackers** (GPT-2, OPT, T5)
3. **Black-box evaluation results** với similarity scores
4. **Complete research pipeline** sẵn sàng cho publication

### **Success metrics:**
- ✅ Tất cả datasets được tạo thành công
- ✅ Tất cả attackers được train thành công  
- ✅ Similarity scores > 0.7 (tốt)
- ✅ Text quality metrics đạt chuẩn

---

## 🎯 Next Steps

Sau khi hoàn thành pipeline:

1. **Analyze results** - Phân tích kết quả chi tiết
2. **Compare with baselines** - So sánh với baseline methods
3. **Write paper** - Viết research paper
4. **Submit to conference** - Submit đến conference/journal

---

**🎉 Chúc bạn thành công với research project!**

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