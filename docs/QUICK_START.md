# Quick Start Guide

## How to Train Only MLP Model

### Step 1: Configure the Models

Edit `config.py` and set other models to `enabled: False`:

```python
# MLP Model Configuration
MLP_CONFIG = {
    'enabled': True,  # Keep this as True
    'hidden_dims': [512, 256, 128],
    'dropout': 0.3,
    'learning_rate': 0.001,
    'epochs': 20,
    # ... other parameters
}

# CNN Model Configuration
CNN_CONFIG = {
    'enabled': False,  # Set to False to skip CNN
    # ...
}

# Transformer Model Configuration
TRANSFORMER_CONFIG = {
    'enabled': False,  # Set to False to skip Transformer
    # ...
}
```

### Step 2: Run Training

```bash
python scripts/training/train.py
```

This will only train the MLP model.

## How to Train Only CNN Model

Set in `config.py`:
```python
MLP_CONFIG = {'enabled': False, ...}
CNN_CONFIG = {'enabled': True, ...}
TRANSFORMER_CONFIG = {'enabled': False, ...}
```

## How to Train Only Transformer Model

Set in `config.py`:
```python
MLP_CONFIG = {'enabled': False, ...}
CNN_CONFIG = {'enabled': False, ...}
TRANSFORMER_CONFIG = {'enabled': True, ...}
```

## Adjusting MLP Parameters

You can customize MLP parameters in `config.py`:

```python
MLP_CONFIG = {
    'enabled': True,
    'hidden_dims': [1024, 512, 256, 128],  # Change hidden layers
    'dropout': 0.5,  # Change dropout rate
    'learning_rate': 0.0005,  # Change learning rate
    'epochs': 30,  # Change number of epochs
    'weight_decay': 1e-4,  # Change weight decay
    'scheduler_step_size': 10,  # Change LR scheduler step
    'scheduler_gamma': 0.5  # Change LR decay factor
}
```

## Training Output

After training, you'll find:
- Model saved at: `saved_models/mlp_model.pth`
- Training history (loss and accuracy)
- Test set evaluation results

