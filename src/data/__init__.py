"""
Data loading module
Supports both PyTorch and TensorFlow
"""
# PyTorch data loader
try:
    from .data_loader import NewsDataset, load_data
    __all__ = ['NewsDataset', 'load_data']
except ImportError:
    __all__ = []

# TensorFlow data loader
try:
    from .data_loader_tf import load_data as load_data_tf
    __all__.append('load_data_tf')
except ImportError:
    pass

