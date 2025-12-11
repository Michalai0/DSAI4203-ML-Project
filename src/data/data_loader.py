"""
Data loading module
Used to load BERT embeddings and category labels, and perform preprocessing
"""
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

from config import DATA_CONFIG


class NewsDataset(Dataset):
    """News classification dataset"""
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def load_data(test_size=None, val_size=None, random_state=None, data_root=None):
    """
    Load BERT embeddings and category labels
    
    Args:
        test_size: Test set ratio
        val_size: Validation set ratio (split from training set)
        random_state: Random seed
        data_root: Data root directory path, if None will auto-detect
    
    Returns:
        train_loader, val_loader, test_loader, label_encoder, num_classes
    """
    # Resolve configuration defaults
    test_size = DATA_CONFIG.get('test_size', 0.2) if test_size is None else test_size
    val_size = DATA_CONFIG.get('val_size', 0.1) if val_size is None else val_size
    random_state = DATA_CONFIG.get('random_state', 42) if random_state is None else random_state

    embeddings, encoded_labels, label_encoder, num_classes, data_root = load_dataset_arrays(
        data_root=data_root
    )
    
    # Split dataset
    X_temp, X_test, y_temp, y_test = train_test_split(
        embeddings, encoded_labels, 
        test_size=test_size, 
        random_state=random_state,
        stratify=encoded_labels
    )
    
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=y_temp
    )
    
    print(f"\nDataset split:")
    print(f"  Training set: {len(X_train)}")
    print(f"  Validation set: {len(X_val)}")
    print(f"  Test set: {len(X_test)}")
    
    batch_size = DATA_CONFIG.get('batch_size', 64)
    print(f"Batch size: {batch_size}")
    
    # Create datasets and data loaders
    train_dataset = NewsDataset(X_train, y_train)
    val_dataset = NewsDataset(X_val, y_val)
    test_dataset = NewsDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, label_encoder, num_classes


def load_dataset_arrays(data_root=None):
    """
    Load embeddings and labels without splitting into loaders.
    
    Returns:
        embeddings (np.ndarray), encoded_labels (np.ndarray), label_encoder, num_classes, data_root
    """
    # Auto-detect data root directory
    if data_root is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_root = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'data')
    
    print("Loading data...")

    # Select embedding file based on config
    embedding_type = DATA_CONFIG.get('embedding_type', 'bert')
    embedding_files = DATA_CONFIG.get('embedding_files', {})
    embedding_filename = embedding_files.get(embedding_type, 'bert_embeddings_all.csv')
    embeddings_path = os.path.join(data_root, 'embeddings', embedding_filename)

    print(f"Loading {embedding_type} embeddings from: {embeddings_path}")
    
    # Check if this is the new format file (headlines_with_bert_embeddings.csv)
    # which uses pipe separator and contains category labels
    if embedding_filename == 'headlines_with_bert_embeddings.csv':
        embeddings_df = pd.read_csv(embeddings_path, sep='|')
        bert_cols = [col for col in embeddings_df.columns if col.startswith('bert_dim_')]
        embeddings = embeddings_df[bert_cols].values
        print("Loading category labels from embeddings file...")
        labels = embeddings_df['category'].values
    else:
        embeddings_df = pd.read_csv(embeddings_path)
        embeddings = embeddings_df.values
        print("Loading category labels...")
        labels_path = os.path.join(data_root, 'processed', 'headlines_preprocessed.csv')
        labels_df = pd.read_csv(labels_path, sep='|')
        labels = labels_df['category'].values
        min_len = min(len(embeddings), len(labels))
        embeddings = embeddings[:min_len]
        labels = labels[:min_len]
    
    print(f"Total samples (raw): {len(embeddings)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    allowed_categories = DATA_CONFIG.get('allowed_categories')
    if allowed_categories:
        allowed_set = set(allowed_categories)
        mask = np.isin(labels, list(allowed_set))
        filtered_embeddings = embeddings[mask]
        filtered_labels = labels[mask]
        print(f"Filtering to allowed categories ({len(allowed_set)} classes): {allowed_set}")
        print(f"  Samples kept: {len(filtered_embeddings)} / {len(embeddings)}")
        if len(filtered_embeddings) == 0:
            raise ValueError("No samples left after filtering by allowed_categories.")
        embeddings, labels = filtered_embeddings, filtered_labels

    max_samples_per_class = DATA_CONFIG.get('max_samples_per_class')
    if max_samples_per_class is not None:
        rng = np.random.default_rng(DATA_CONFIG.get('random_state', 42))
        kept_embeddings = []
        kept_labels = []
        unique_labels = np.unique(labels)
        for cls in unique_labels:
            cls_mask = labels == cls
            cls_indices = np.where(cls_mask)[0]
            if len(cls_indices) > max_samples_per_class:
                chosen = rng.choice(cls_indices, size=max_samples_per_class, replace=False)
            else:
                chosen = cls_indices
            kept_embeddings.append(embeddings[chosen])
            kept_labels.append(labels[chosen])
            print(f"  Class '{cls}': keeping {len(chosen)} samples (orig {len(cls_indices)})")
        embeddings = np.vstack(kept_embeddings)
        labels = np.concatenate(kept_labels)
    
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    
    print(f"Number of categories: {num_classes}")
    print(f"Category list: {label_encoder.classes_[:10]}...")
    
    return embeddings, encoded_labels, label_encoder, num_classes, data_root

