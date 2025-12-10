"""
Model definition module
Supports both PyTorch and TensorFlow
"""
# PyTorch models
try:
    from .models import MLP, CNN, TransformerClassifier
    __all__ = ['MLP', 'CNN', 'TransformerClassifier']
except ImportError:
    __all__ = []

# TensorFlow models
try:
    from .models_tf import create_mlp, create_cnn, create_transformer
    __all__.extend(['create_mlp', 'create_cnn', 'create_transformer'])
except ImportError:
    pass

