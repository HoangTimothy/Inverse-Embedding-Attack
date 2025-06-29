# Phân tích Flow - Inverse Embedding Attack

## 🎯 Ý tưởng chính

### Vấn đề của GEIA gốc:
- Cần data gốc để train attacker
- Không thực tế trong real-world scenarios
- Metrics cao nhưng không practical

### Giải pháp của chúng ta:
- **Transfer Learning**: Train trên models tương tự, test trên black-box
- **Cross-Model Attack**: Không cần data gốc của target model
- **Alignment Layers**: Xử lý khác biệt embedding dimensions

## 🔄 Flow chi tiết

### **Phase 1: Data Preparation**
```
Dataset → Multiple Embedding Models → Embeddings Storage
```

**Input:**
- Dataset: SST-2, PersonaChat, ABCD
- Embedding Models: 4 models khác nhau
  - all-mpnet-base-v2 (768d)
  - stsb-roberta-base (768d) 
  - all-MiniLM-L6-v2 (384d)
  - paraphrase-MiniLM-L6-v2 (384d)

**Process:**
1. Load dataset từ HuggingFace/local
2. Extract embeddings từ mỗi model
3. Save embeddings + sentences pairs
4. Create alignment layers cho different dimensions

**Output:**
```
data/embeddings/
├── sst2_train_all-mpnet-base-v2.json
├── sst2_train_stsb-roberta-base.json
├── sst2_train_all-MiniLM-L6-v2.json
└── sst2_train_paraphrase-MiniLM-L6-v2.json
```

### **Phase 2: Attacker Training**
```
Embeddings → Attacker Models → Trained Attackers
```

**Input:**
- Embeddings từ 3 models (không phải black-box)
- Attacker models: GPT-2, OPT, T5

**Process:**
1. Load embeddings cho từng model
2. Apply alignment nếu cần
3. Train attacker models (dựa trên GEIA framework)
4. Save trained models

**Output:**
```
models/
├── attacker_gpt2_all-mpnet-base-v2/
├── attacker_opt_stsb-roberta-base/
└── attacker_t5_all-MiniLM-L6-v2/
```

### **Phase 3: Black-box Testing**
```
Black-box Embeddings → Trained Attackers → Recovered Text
```

**Input:**
- Black-box model (model thứ 4)
- Test sentences
- Trained attackers

**Process:**
1. Generate embeddings từ black-box model
2. Apply alignment layers
3. Use trained attackers để generate text
4. Evaluate performance

**Output:**
```
experiments/results/
└── blackbox_test_sst2_paraphrase-MiniLM-L6-v2.json
```

## 📊 Key Components

### **1. EmbeddingPreparer (`src/data_processing/prepare_embeddings.py`)**
```python
class EmbeddingPreparer:
    def load_embedding_models(self):
        # Load 4 embedding models
    
    def create_embeddings(self, sentences, model_name):
        # Generate embeddings for sentences
    
    def create_alignment_layers(self):
        # Create projection layers for dimension alignment
```

### **2. InverseEmbeddingAttacker (`src/attackers/train_attackers.py`)**
```python
class InverseEmbeddingAttacker:
    def train_on_batch(self, batch_embeddings, batch_sentences):
        # Training step (based on GEIA)
    
    def train(self, dataset_name, split='train'):
        # Complete training loop
```

### **3. BlackBoxTester (`src/evaluation/test_blackbox.py`)**
```python
class BlackBoxTester:
    def get_blackbox_embeddings(self, sentences):
        # Extract embeddings from black-box model
    
    def generate_text(self, embeddings, attacker_info):
        # Generate text using trained attackers
    
    def evaluate_attack(self, original, generated):
        # Calculate metrics
```

### **4. EmbeddingAlignment (`src/utils/alignment.py`)**
```python
class EmbeddingAlignment:
    def fit(self, source_embeddings, target_embeddings):
        # Learn alignment transformation
    
    def transform(self, embeddings):
        # Apply alignment
```

## 🎯 Advantages over GEIA

### **1. Realistic Attack Scenario**
- ✅ Không cần data gốc của black-box
- ✅ Transfer learning approach
- ✅ Cross-model generalization

### **2. Practical Implementation**
- ✅ Reuse GEIA framework
- ✅ Modular design
- ✅ Easy to extend

### **3. Better Evaluation**
- ✅ Test on unseen models
- ✅ More realistic metrics
- ✅ Cross-domain evaluation

## 🚀 Usage Examples

### **Quick Start:**
```bash
# Install dependencies
pip install -r requirements.txt

# Run demo
python demo.py

# Run full experiment
python run_experiment.py --dataset sst2 --blackbox_model all-mpnet-base-v2
```

### **Step by Step:**
```bash
# 1. Prepare embeddings
python src/data_processing/prepare_embeddings.py --dataset sst2 --split train

# 2. Train attackers
python src/attackers/train_attackers.py --dataset sst2 --attacker gpt2 --embedding_model all-mpnet-base-v2

# 3. Test on black-box
python src/evaluation/test_blackbox.py --dataset sst2 --blackbox_model paraphrase-MiniLM-L6-v2
```

## 📈 Expected Results

### **Performance Metrics:**
- **Embedding Similarity**: 0.7-0.9
- **Exact Match Rate**: 0.1-0.3  
- **BLEU Score**: 0.5-0.8
- **Edit Distance**: 10-30 characters

### **Transfer Learning Effectiveness:**
- Better performance khi source và target models tương tự
- Degradation khi models khác biệt lớn
- Alignment layers giúp cải thiện performance

## 🔧 Technical Details

### **Alignment Methods:**
1. **Linear Regression**: Đơn giản, hiệu quả
2. **PCA-based**: Cho high-dimensional embeddings
3. **Neural Alignment**: Cho complex transformations

### **Attacker Models:**
- **GPT-2**: Causal LM, tốt cho text generation
- **OPT**: Open-source alternative
- **T5**: Seq2seq, có thể tốt hơn cho structured text

### **Evaluation Metrics:**
- **Embedding Similarity**: Cosine similarity
- **Exact Match**: String comparison
- **BLEU Score**: N-gram overlap
- **Edit Distance**: Character-level difference

## 🎯 Future Improvements

### **1. Advanced Alignment:**
- Multi-step alignment
- Adversarial alignment
- Domain-specific alignment

### **2. Better Attackers:**
- Ensemble methods
- Adversarial training
- Multi-modal attackers

### **3. Evaluation:**
- Human evaluation
- Semantic similarity
- Privacy metrics 