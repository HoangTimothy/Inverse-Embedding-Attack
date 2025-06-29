import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

class EmbeddingAlignment:
    """Utility class for aligning embeddings from different models"""
    
    def __init__(self, source_dim, target_dim, method='linear'):
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.method = method
        self.alignment_layer = None
        
    def fit(self, source_embeddings, target_embeddings):
        """Fit alignment layer using paired embeddings"""
        if self.method == 'linear':
            self._fit_linear(source_embeddings, target_embeddings)
        elif self.method == 'pca':
            self._fit_pca(source_embeddings, target_embeddings)
        else:
            raise ValueError(f"Unsupported alignment method: {self.method}")
    
    def _fit_linear(self, source_embeddings, target_embeddings):
        """Fit linear regression alignment"""
        # Use sklearn LinearRegression for better numerical stability
        reg = LinearRegression()
        reg.fit(source_embeddings, target_embeddings)
        
        # Create PyTorch layer
        self.alignment_layer = nn.Linear(self.source_dim, self.target_dim)
        self.alignment_layer.weight.data = torch.tensor(reg.coef_, dtype=torch.float32)
        self.alignment_layer.bias.data = torch.tensor(reg.intercept_, dtype=torch.float32)
    
    def _fit_pca(self, source_embeddings, target_embeddings):
        """Fit PCA-based alignment"""
        # Project to common space using PCA
        min_dim = min(self.source_dim, self.target_dim)
        
        # Fit PCA on concatenated embeddings
        combined = np.concatenate([source_embeddings, target_embeddings], axis=1)
        pca = PCA(n_components=min_dim)
        pca.fit(combined)
        
        # Create projection layers
        self.alignment_layer = nn.Sequential(
            nn.Linear(self.source_dim, min_dim),
            nn.Linear(min_dim, self.target_dim)
        )
    
    def transform(self, embeddings):
        """Transform embeddings using fitted alignment"""
        if self.alignment_layer is None:
            raise ValueError("Alignment layer not fitted. Call fit() first.")
        
        with torch.no_grad():
            if isinstance(embeddings, np.ndarray):
                embeddings = torch.tensor(embeddings, dtype=torch.float32)
            
            aligned_embeddings = self.alignment_layer(embeddings)
            return aligned_embeddings
    
    def save(self, path):
        """Save alignment layer"""
        if self.alignment_layer is not None:
            torch.save(self.alignment_layer.state_dict(), path)
    
    def load(self, path):
        """Load alignment layer"""
        if self.alignment_layer is None:
            if self.method == 'linear':
                self.alignment_layer = nn.Linear(self.source_dim, self.target_dim)
            elif self.method == 'pca':
                min_dim = min(self.source_dim, self.target_dim)
                self.alignment_layer = nn.Sequential(
                    nn.Linear(self.source_dim, min_dim),
                    nn.Linear(min_dim, self.target_dim)
                )
        
        self.alignment_layer.load_state_dict(torch.load(path))

class CrossModelAlignment:
    """Advanced alignment for multiple embedding models"""
    
    def __init__(self, target_model_name, target_dim):
        self.target_model_name = target_model_name
        self.target_dim = target_dim
        self.alignments = {}
    
    def create_alignments(self, embedding_models_info, dataset_embeddings):
        """Create alignment layers for all models to target dimension"""
        for model_name, embeddings in dataset_embeddings.items():
            if model_name != self.target_model_name:
                source_dim = embeddings.shape[1]
                
                if source_dim != self.target_dim:
                    print(f"Creating alignment for {model_name} ({source_dim} -> {self.target_dim})")
                    
                    # For now, use simple linear projection
                    # In practice, you might want to use more sophisticated methods
                    alignment = EmbeddingAlignment(source_dim, self.target_dim, method='linear')
                    
                    # Use target model embeddings as reference
                    target_embeddings = dataset_embeddings[self.target_model_name]
                    
                    # Fit alignment (this is simplified - you'd need proper paired data)
                    # In real scenario, you might use parallel corpora or other methods
                    alignment._fit_linear(embeddings[:len(target_embeddings)], target_embeddings)
                    
                    self.alignments[model_name] = alignment
    
    def align_embeddings(self, model_name, embeddings):
        """Align embeddings from specific model"""
        if model_name == self.target_model_name:
            return embeddings
        
        if model_name in self.alignments:
            return self.alignments[model_name].transform(embeddings)
        else:
            raise ValueError(f"No alignment found for model: {model_name}")
    
    def save_alignments(self, save_dir):
        """Save all alignment layers"""
        for model_name, alignment in self.alignments.items():
            save_path = f"{save_dir}/alignment_{model_name}.pt"
            alignment.save(save_path)
    
    def load_alignments(self, save_dir):
        """Load all alignment layers"""
        import os
        for model_name in self.alignments.keys():
            save_path = f"{save_dir}/alignment_{model_name}.pt"
            if os.path.exists(save_path):
                self.alignments[model_name].load(save_path) 