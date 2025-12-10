# News Classification Models

This project implements three neural network models for news headline and content classification:
- **MLP** (Multi-Layer Perceptron)
- **CNN** (Convolutional Neural Network)
- **Transformer** (Transformer Encoder)

## File Description

- `data_loader.py`: Data loading module, responsible for loading BERT embeddings and category labels
- `models.py`: Model definitions, containing implementations of three models
- `train.py`: Training script for training and evaluating all models
- `requirements.txt`: Project dependencies

## Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Ensure the following files exist:
- `bert_embeddings_all.csv`: BERT embedding vectors (768-dim)
- `headlines_preprocessed.csv`: Preprocessed headlines and category labels

### 3. Train Models

Run the training script:

```bash
python train.py
```

This will:
- Automatically load data and split into training, validation, and test sets
- Train MLP, CNN, and Transformer models sequentially
- Save best models to `saved_models/` directory
- Evaluate model performance on test set
- Generate detailed classification reports

### 4. Model Configuration

You can modify model configuration in `train.py`:

```python
models_config = {
    'MLP': {
        'model': MLP(input_dim=768, num_classes=42),
        'lr': 0.001,
        'epochs': 20
    },
    'CNN': {
        'model': CNN(input_dim=768, num_classes=42),
        'lr': 0.001,
        'epochs': 20
    },
    'Transformer': {
        'model': TransformerClassifier(input_dim=768, num_classes=42),
        'lr': 0.0005,
        'epochs': 20
    }
}
```

## Model Architecture

### MLP (Multi-Layer Perceptron)
- Input: 768-dim BERT embeddings
- Structure: Fully connected layers + BatchNorm + ReLU + Dropout
- Hidden layer dimensions: [512, 256, 128]
- Output: Probability distribution over 42 categories

### CNN (Convolutional Neural Network)
- Input: 768-dim BERT embeddings (reshaped as sequence)
- Structure: 1D convolution layers + global max pooling + fully connected layers
- Convolution kernel sizes: [3, 4, 5]
- Output: Probability distribution over 42 categories

### Transformer
- Input: 768-dim BERT embeddings (split into 3 tokens of 256-dim each)
- Structure: Positional encoding + Transformer encoder + classification head
- Encoder layers: 3 layers
- Attention heads: 8
- Output: Probability distribution over 42 categories

## Output Results

After training completes, the following will be generated:
- `saved_models/mlp_model.pth`: MLP model weights
- `saved_models/cnn_model.pth`: CNN model weights
- `saved_models/transformer_model.pth`: Transformer model weights

Each model file contains:
- Model state dictionary
- Training history (loss and accuracy)
- Model configuration information

## Data Statistics

- Total samples: 209,527
- Number of categories: 42
- BERT embedding dimension: 768
- Dataset split:
  - Training set: 80%
  - Validation set: 10%
  - Test set: 10%
