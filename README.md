# Inverse Embedding Attack

## Giải pháp tấn công ngược embedding thực tế

### Ý tưởng chính
Thay vì train attacker trực tiếp trên black-box model (không thực tế), chúng ta:
1. Train attackers trên các embedding models tương tự
2. Sử dụng transfer learning để tấn công black-box model
3. Không cần data gốc của black-box model

### Flow chính
```
Dataset → Multiple Embedding Models → Train Attackers → Test on Black-box
```

### Cấu trúc dự án
```
Inverse_Embedding_Attack/
├── src/
│   ├── data_processing/     # Xử lý dataset và tạo embeddings
│   ├── attackers/          # Các model attacker
│   ├── evaluation/         # Đánh giá kết quả
│   └── utils/             # Utilities và helpers
├── data/                  # Dataset và embeddings
├── models/               # Trained models
├── experiments/          # Configs và results
└── configs/             # Configuration files
```

### Models được hỗ trợ
- **Embedding Models**: all-mpnet-base-v2, stsb-roberta-base, all-MiniLM-L6-v2, paraphrase-MiniLM-L6-v2
- **Attacker Models**: GPT-2, OPT, T5 (từ GEIA)
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