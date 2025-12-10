"""
Model Configuration File
All model hyperparameters can be adjusted here
"""

# Data Configuration
DATA_CONFIG = {
    # Choose which embedding to use: 'bert', 'word2vec' or 'bert_headlines'
    'embedding_type': 'bert_headlines',
    # Embedding dimension lookup (used to set model input_dim)
    'embedding_dims': {
        'bert': 768,
        'word2vec': 300,
    },
    # Embedding filename lookup
    'embedding_files': {
        'bert': 'bert_embeddings_all.csv',
        'word2vec': 'word2vec_embeddings_all.csv',
        'bert_headlines': 'headlines_with_bert_embeddings.csv',
    },
    'input_dim': 768,  # Default/fallback dimension
    'num_classes': 42,  # Number of news categories
    'batch_size': 1024,
    'test_size': 0.2,  # Test set ratio
    'val_size': 0.1,   # Validation set ratio (from training set)
    'random_state': 42
}

# MLP Model Configuration
MLP_CONFIG = {
    'enabled': True,  # Set to False to skip training this model
    'hidden_dims': [512, 256, 128],  # Hidden layer dimensions
    'dropout': 0.3,
    'learning_rate': 0.001,
    'epochs': 20,
    'weight_decay': 1e-5,
    'scheduler_step_size': 7,  # Learning rate scheduler step size
    'scheduler_gamma': 0.1  # Learning rate decay factor
}

# CNN Model Configuration
CNN_CONFIG = {
    'enabled': True,  # Set to False to skip training this model
    'num_filters': 128,  # Number of convolution filters
    'filter_sizes': [3, 4, 5],  # Convolution kernel sizes
    'dropout': 0.3,
    'learning_rate': 0.001,
    'epochs': 20,
    'weight_decay': 1e-5,
    'scheduler_step_size': 7,
    'scheduler_gamma': 0.1
}

# Transformer Model Configuration
TRANSFORMER_CONFIG = {
    'enabled': True,  # Set to False to skip training this model
    'd_model': 256,  # Model dimension
    'nhead': 8,  # Number of attention heads
    'num_layers': 3,  # Number of transformer encoder layers
    'seq_len': 3,  # Number of tokens to split 768-dim embedding into
    'dropout': 0.3,
    'learning_rate': 0.0005,
    'epochs': 20,
    'weight_decay': 1e-5,
    'scheduler_step_size': 7,
    'scheduler_gamma': 0.1
}

# Training Configuration
TRAINING_CONFIG = {
    'device': 'auto',  # 'auto', 'cuda', or 'cpu' - Use 'cuda' for GPU, 'cpu' for CPU, 'auto' to auto-detect
    # Note: RTX 5080 (sm_120) has compatibility warnings but CUDA is available
    # 'auto' will try GPU first, fallback to CPU if needed
    'save_models': True,  # Whether to save trained models
    'saved_models_dir': 'saved_models',  # Directory to save models
    # Early stopping settings
    'early_stopping_metric': 'val_acc',  # 'val_acc' or 'val_loss'
    'early_stopping_patience': 5,
    'early_stopping_min_delta': 0.0  # Minimum improvement to reset patience
}

