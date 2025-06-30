#!/usr/bin/env python3
"""
Analyze and visualize the 4 datasets and embedding vectors
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from sentence_transformers import SentenceTransformer

def load_embedding_data(file_path):
    """Load embedding data from JSON file"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return data

def analyze_dataset(data, dataset_name):
    """Analyze a single dataset"""
    if data is None:
        return None
    
    embeddings = np.array(data['embeddings'])
    sentences = data['sentences']
    model_name = data.get('model_name', 'unknown')
    embedding_dim = data.get('embedding_dim', embeddings.shape[1])
    
    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_name}")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Number of samples: {len(sentences)}")
    print(f"Embedding shape: {embeddings.shape}")
    
    # Basic statistics
    print(f"\nEmbedding Statistics:")
    print(f"  Mean: {embeddings.mean():.6f}")
    print(f"  Std: {embeddings.std():.6f}")
    print(f"  Min: {embeddings.min():.6f}")
    print(f"  Max: {embeddings.max():.6f}")
    
    # Sample sentences
    print(f"\nSample Sentences:")
    for i, sentence in enumerate(sentences[:5]):
        print(f"  {i+1}. {sentence}")
    
    if len(sentences) > 5:
        print(f"  ... and {len(sentences) - 5} more")
    
    return {
        'embeddings': embeddings,
        'sentences': sentences,
        'model_name': model_name,
        'embedding_dim': embedding_dim
    }

def visualize_embeddings(embeddings, title, method='pca'):
    """Visualize embeddings using PCA or t-SNE"""
    if method == 'pca':
        reducer = PCA(n_components=2)
        method_name = 'PCA'
    else:
        reducer = TSNE(n_components=2, random_state=42)
        method_name = 't-SNE'
    
    # Reduce dimensions
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7)
    plt.title(f'{title} - {method_name} Visualization')
    plt.xlabel(f'{method_name} Component 1')
    plt.ylabel(f'{method_name} Component 2')
    plt.grid(True, alpha=0.3)
    
    # Add sample points with labels
    for i in range(min(5, len(reduced_embeddings))):
        plt.annotate(f'S{i+1}', 
                    (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8)
    
    plt.tight_layout()
    return plt.gcf()

def compare_embeddings(datasets):
    """Compare embeddings across datasets"""
    print(f"\n{'='*60}")
    print("EMBEDDING COMPARISON")
    print(f"{'='*60}")
    
    comparison_data = []
    
    for name, data in datasets.items():
        if data is None:
            continue
            
        embeddings = data['embeddings']
        model_name = data['model_name']
        embedding_dim = data['embedding_dim']
        
        # Calculate statistics
        mean_norm = np.linalg.norm(embeddings, axis=1).mean()
        std_norm = np.linalg.norm(embeddings, axis=1).std()
        
        comparison_data.append({
            'Dataset': name,
            'Model': model_name,
            'Dimension': embedding_dim,
            'Samples': len(embeddings),
            'Mean Norm': mean_norm,
            'Std Norm': std_norm,
            'Mean': embeddings.mean(),
            'Std': embeddings.std()
        })
    
    # Create comparison table
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    
    return df

def analyze_vector_weights(embeddings, sentences, title):
    """Analyze vector weights and their distribution"""
    print(f"\n{'='*60}")
    print(f"VECTOR WEIGHTS ANALYSIS: {title}")
    print(f"{'='*60}")
    
    # Calculate weight statistics for each dimension
    mean_weights = embeddings.mean(axis=0)
    std_weights = embeddings.std(axis=0)
    
    # Find most important dimensions (highest variance)
    importance_scores = std_weights ** 2
    top_dimensions = np.argsort(importance_scores)[-10:]  # Top 10 dimensions
    
    print(f"Top 10 Most Important Dimensions:")
    for i, dim in enumerate(reversed(top_dimensions)):
        print(f"  Dim {dim}: std={std_weights[dim]:.6f}, mean={mean_weights[dim]:.6f}")
    
    # Visualize weight distribution
    plt.figure(figsize=(15, 5))
    
    # Histogram of all weights
    plt.subplot(1, 3, 1)
    plt.hist(embeddings.flatten(), bins=50, alpha=0.7, edgecolor='black')
    plt.title(f'{title} - Weight Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    
    # Mean weights across dimensions
    plt.subplot(1, 3, 2)
    plt.plot(mean_weights)
    plt.title(f'{title} - Mean Weights by Dimension')
    plt.xlabel('Dimension')
    plt.ylabel('Mean Weight')
    plt.grid(True, alpha=0.3)
    
    # Standard deviation across dimensions
    plt.subplot(1, 3, 3)
    plt.plot(std_weights)
    plt.title(f'{title} - Std Weights by Dimension')
    plt.xlabel('Dimension')
    plt.ylabel('Standard Deviation')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt.gcf()

def calculate_similarities(embeddings, sentences):
    """Calculate similarities between embeddings"""
    print(f"\nSimilarity Analysis:")
    
    # Calculate cosine similarities
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    
    # Show similarity matrix for first 10 samples
    n_show = min(10, len(similarity_matrix))
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix[:n_show, :n_show], 
                annot=True, fmt='.3f', cmap='viridis',
                xticklabels=range(1, n_show+1),
                yticklabels=range(1, n_show+1))
    plt.title('Cosine Similarity Matrix (First 10 Samples)')
    plt.xlabel('Sample Index')
    plt.ylabel('Sample Index')
    plt.tight_layout()
    
    # Show most similar pairs
    print(f"Most Similar Pairs:")
    for i in range(n_show):
        for j in range(i+1, n_show):
            sim = similarity_matrix[i, j]
            print(f"  ({i+1}, {j+1}): {sim:.3f}")
            print(f"    '{sentences[i][:50]}...'")
            print(f"    '{sentences[j][:50]}...'")
            print()

def main():
    print("EMBEDDING DATASET ANALYSIS")
    print("="*60)
    
    # Expected dataset files
    expected_datasets = {
        'SST-2 (GPT-2)': 'data/embeddings/sst2_train_stsb-roberta-base.json',
        'PersonaChat (OPT)': 'data/embeddings/personachat_train_all-MiniLM-L6-v2.json',
        'ABCD (T5)': 'data/embeddings/abcd_train_paraphrase-MiniLM-L6-v2.json',
        'Black-box Test': 'data/embeddings/test_all-mpnet-base-v2.json'
    }
    
    # Load and analyze datasets
    datasets = {}
    for name, file_path in expected_datasets.items():
        data = load_embedding_data(file_path)
        analysis = analyze_dataset(data, name)
        datasets[name] = analysis
    
    # Compare datasets
    comparison_df = compare_embeddings(datasets)
    
    # Analyze each dataset in detail
    for name, data in datasets.items():
        if data is None:
            continue
            
        # Vector weights analysis
        weights_fig = analyze_vector_weights(
            data['embeddings'], 
            data['sentences'], 
            name
        )
        weights_fig.savefig(f'analysis_{name.replace(" ", "_").replace("(", "").replace(")", "")}_weights.png', dpi=300, bbox_inches='tight')
        
        # Embedding visualization
        viz_fig = visualize_embeddings(data['embeddings'], name, method='pca')
        viz_fig.savefig(f'analysis_{name.replace(" ", "_").replace("(", "").replace(")", "")}_pca.png', dpi=300, bbox_inches='tight')
        
        # Similarity analysis
        calculate_similarities(data['embeddings'], data['sentences'])
    
    # Save comparison table
    comparison_df.to_csv('embedding_comparison.csv', index=False)
    print(f"\nComparison table saved to: embedding_comparison.csv")
    
    # Summary
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total datasets analyzed: {len([d for d in datasets.values() if d is not None])}")
    print(f"Visualization files saved with prefix: analysis_")
    print(f"Comparison table saved: embedding_comparison.csv")
    
    plt.show()

if __name__ == "__main__":
    main() 