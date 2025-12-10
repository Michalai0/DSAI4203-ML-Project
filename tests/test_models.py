"""
Test script - Verify model definitions are correct
"""
import torch
import os
import sys

# Add project root directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.models import MLP, CNN, TransformerClassifier

# Set random seed
torch.manual_seed(42)

# Test parameters
batch_size = 4
input_dim = 768
num_classes = 42

# Create test input
test_input = torch.randn(batch_size, input_dim)

print("Testing model definitions...")
print(f"Input shape: {test_input.shape}\n")

# Test MLP
print("=" * 60)
print("Testing MLP model")
print("=" * 60)
mlp = MLP(input_dim=input_dim, num_classes=num_classes)
mlp.eval()
with torch.no_grad():
    mlp_output = mlp(test_input)
print(f"MLP output shape: {mlp_output.shape}")
print(f"MLP parameter count: {sum(p.numel() for p in mlp.parameters()):,}")
print("✓ MLP model OK\n")

# Test CNN
print("=" * 60)
print("Testing CNN model")
print("=" * 60)
cnn = CNN(input_dim=input_dim, num_classes=num_classes)
cnn.eval()
with torch.no_grad():
    cnn_output = cnn(test_input)
print(f"CNN output shape: {cnn_output.shape}")
print(f"CNN parameter count: {sum(p.numel() for p in cnn.parameters()):,}")
print("✓ CNN model OK\n")

# Test Transformer
print("=" * 60)
print("Testing Transformer model")
print("=" * 60)
transformer = TransformerClassifier(input_dim=input_dim, num_classes=num_classes)
transformer.eval()
with torch.no_grad():
    transformer_output = transformer(test_input)
print(f"Transformer output shape: {transformer_output.shape}")
print(f"Transformer parameter count: {sum(p.numel() for p in transformer.parameters()):,}")
print("✓ Transformer model OK\n")

print("=" * 60)
print("All model tests passed!")
print("=" * 60)

