"""
Neural network model definitions
Contains three models: MLP, CNN, and Transformer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MLP(nn.Module):
    """
    Multi-Layer Perceptron model
    Uses BERT embeddings as input and classifies through fully connected layers
    """
    def __init__(self, input_dim=768, hidden_dims=[512, 256, 128], num_classes=42, dropout=0.3):
        super(MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class CNN(nn.Module):
    """
    Convolutional Neural Network model
    Reshapes BERT embeddings into sequence form and uses 1D convolution to extract features
    """
    def __init__(self, input_dim=768, num_filters=128, filter_sizes=[3, 4, 5], num_classes=42, dropout=0.3):
        super(CNN, self).__init__()
        
        # Reshape 768-dim embeddings to (batch, 1, 768) for 1D convolution
        # Or reshape to sequence form like (batch, 24, 32)
        # Here we use (batch, 1, 768) form
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        # Calculate dimension after convolution and global max pooling
        # Each conv outputs (batch, num_filters) after global max pooling
        # Concatenating all gives (batch, num_filters * len(filter_sizes))
        conv_output_dim = num_filters * len(filter_sizes)
        
        self.fc1 = nn.Linear(conv_output_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # x shape: (batch, 768)
        # Reshape to (batch, 1, 768) for 1D convolution
        x = x.unsqueeze(1)  # (batch, 1, 768)
        
        # Apply multiple convolution kernels
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))  # (batch, num_filters, seq_len)
            # Global max pooling
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))  # (batch, num_filters, 1)
            pooled = pooled.squeeze(2)  # (batch, num_filters)
            conv_outputs.append(pooled)
        
        # Concatenate all convolution outputs
        x = torch.cat(conv_outputs, dim=1)  # (batch, num_filters * len(filter_sizes))
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class CNNEnsemble(nn.Module):
    """
    Simple average-fusion over multiple CNN sub-models (logits average).
    Sub-models can vary kernels/channels to increase diversity.
    """
    def __init__(self, input_dim=768, variant_configs=None, num_classes=42):
        super().__init__()
        if not variant_configs:
            variant_configs = [{'num_filters': 128, 'filter_sizes': [3, 4, 5], 'dropout': 0.3}]
        self.sub_models = nn.ModuleList([
            CNN(
                input_dim=input_dim,
                num_filters=cfg.get('num_filters', 128),
                filter_sizes=cfg.get('filter_sizes', [3, 4, 5]),
                num_classes=num_classes,
                dropout=cfg.get('dropout', 0.3),
            )
            for cfg in variant_configs
        ])
    
    def forward(self, x):
        # Forward each sub-model, sum logits then average
        logits_sum = None
        for model in self.sub_models:
            logits = model(x)
            logits_sum = logits if logits_sum is None else logits_sum + logits
        return logits_sum / len(self.sub_models)


class TransformerClassifier(nn.Module):
    """
    Transformer classification model
    Uses Transformer encoder to process BERT embeddings
    Splits 768-dim embeddings into multiple tokens for processing
    """
    def __init__(self, input_dim=768, d_model=256, nhead=8, num_layers=3, num_classes=42, dropout=0.3, seq_len=3):
        super(TransformerClassifier, self).__init__()
        
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Split 768-dim into seq_len tokens, each token dimension is input_dim // seq_len
        # Then project to d_model
        token_dim = input_dim // seq_len
        self.token_projections = nn.ModuleList([
            nn.Linear(token_dim, d_model) for _ in range(seq_len)
        ])
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Classification head
        self.fc1 = nn.Linear(d_model, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # x shape: (batch, 768)
        batch_size = x.size(0)
        
        # Split 768-dim into seq_len tokens
        # Example: 768 -> 3 tokens of 256-dim each
        token_dim = x.size(1) // self.seq_len
        x_tokens = x.view(batch_size, self.seq_len, token_dim)  # (batch, seq_len, token_dim)
        
        # Project each token to d_model dimension
        projected_tokens = []
        for i, proj in enumerate(self.token_projections):
            token = x_tokens[:, i, :]  # (batch, token_dim)
            projected = proj(token)  # (batch, d_model)
            projected_tokens.append(projected)
        
        x = torch.stack(projected_tokens, dim=1)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Use CLS token (first token) or global average pooling
        # Here we use global average pooling
        x = encoded.mean(dim=1)  # (batch, d_model)
        
        # Classification head
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        x = self.fc2(x)
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

