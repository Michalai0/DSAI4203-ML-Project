"""
Neural network model definitions using TensorFlow/Keras
Contains three models: MLP, CNN, and Transformer
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def create_mlp(input_dim=768, hidden_dims=[512, 256, 128], num_classes=42, dropout=0.3):
    """
    Multi-Layer Perceptron model
    Uses BERT embeddings as input and classifies through fully connected layers
    """
    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    
    # Build hidden layers
    for hidden_dim in hidden_dims:
        x = layers.Dense(hidden_dim)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(dropout)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def create_cnn(input_dim=768, num_filters=128, filter_sizes=[3, 4, 5], num_classes=42, dropout=0.3):
    """
    Convolutional Neural Network model
    Reshapes BERT embeddings into sequence form and uses 1D convolution to extract features
    """
    inputs = keras.Input(shape=(input_dim,))
    
    # Reshape to (batch, 1, input_dim) for 1D convolution
    x = layers.Reshape((1, input_dim))(inputs)
    
    # Apply multiple convolution kernels
    conv_outputs = []
    for filter_size in filter_sizes:
        conv = layers.Conv1D(filters=num_filters, kernel_size=filter_size, activation='relu')(x)
        # Global max pooling
        pooled = layers.GlobalMaxPooling1D()(conv)
        conv_outputs.append(pooled)
    
    # Concatenate all convolution outputs
    x = layers.Concatenate()(conv_outputs)  # (batch, num_filters * len(filter_sizes))
    
    # Fully connected layers
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def positional_encoding(seq_len, d_model):
    """Create positional encoding"""
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe[np.newaxis, :, :]  # (1, seq_len, d_model)


class PositionalEncoding(layers.Layer):
    """Positional encoding layer"""
    def __init__(self, d_model, max_len=5000, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len
        
        # Create positional encoding matrix
        pe = positional_encoding(max_len, d_model)
        self.pe = tf.constant(pe, dtype=tf.float32)
    
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]


def create_transformer(input_dim=768, d_model=256, nhead=8, num_layers=3, num_classes=42, dropout=0.3, seq_len=3):
    """
    Transformer classification model
    Uses Transformer encoder to process BERT embeddings
    Splits 768-dim embeddings into multiple tokens for processing
    """
    inputs = keras.Input(shape=(input_dim,))
    
    # Split 768-dim into seq_len tokens
    token_dim = input_dim // seq_len
    x = layers.Reshape((seq_len, token_dim))(inputs)  # (batch, seq_len, token_dim)
    
    # Project each token to d_model dimension
    x = layers.Dense(d_model)(x)  # (batch, seq_len, d_model)
    
    # Add positional encoding
    pos_encoding = PositionalEncoding(d_model)
    x = pos_encoding(x)
    
    # Transformer encoder layers
    for _ in range(num_layers):
        # Multi-head attention
        # Ensure key_dim is valid (d_model must be divisible by nhead)
        key_dim = d_model // nhead
        if d_model % nhead != 0:
            # Adjust nhead if needed
            nhead_adjusted = nhead
            while d_model % nhead_adjusted != 0 and nhead_adjusted > 1:
                nhead_adjusted -= 1
            key_dim = d_model // nhead_adjusted
            nhead = nhead_adjusted
        
        attn_output = layers.MultiHeadAttention(
            num_heads=nhead,
            key_dim=key_dim,
            dropout=dropout
        )(x, x)
        attn_output = layers.Dropout(dropout)(attn_output)
        x = layers.LayerNormalization()(x + attn_output)
        
        # Feed forward
        ffn_output = layers.Dense(d_model * 4, activation='relu')(x)
        ffn_output = layers.Dense(d_model)(ffn_output)
        ffn_output = layers.Dropout(dropout)(ffn_output)
        x = layers.LayerNormalization()(x + ffn_output)
    
    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)  # (batch, d_model)
    
    # Classification head
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

