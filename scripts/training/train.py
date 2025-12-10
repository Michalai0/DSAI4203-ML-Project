"""
Training script using PyTorch
Supports training three models: MLP, CNN, and Transformer
"""
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

# Add project root directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# Import configuration
from config import (
    DATA_CONFIG, MLP_CONFIG, CNN_CONFIG, 
    TRANSFORMER_CONFIG, TRAINING_CONFIG
)

from src.data.data_loader import load_data
from src.models.models import MLP, CNN, TransformerClassifier


def train_model(
    model_name,
    model,
    train_loader,
    val_loader,
    num_epochs=20,
    lr=0.001,
    weight_decay=1e-5,
    scheduler_step_size=7,
    scheduler_gamma=0.1,
    device='cuda',
    early_stopping_metric='val_acc',
    early_stopping_patience=5,
    early_stopping_min_delta=0.0,
):
    """Train model"""
    print(f"\n{'='*60}")
    print(f"Starting training for {model_name} model")
    print(f"{'='*60}\n")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    
    is_loss_metric = early_stopping_metric == 'val_loss'
    best_val_metric = float('inf') if is_loss_metric else 0.0
    best_model_state = None
    patience_counter = 0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for embeddings, labels in train_pbar:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * train_correct / train_total:.2f}%'
            })
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for embeddings, labels in val_pbar:
                embeddings = embeddings.to(device)
                labels = labels.to(device)
                
                outputs = model(embeddings)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * val_correct / val_total:.2f}%'
                })
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  LR: {current_lr:.6f}')
        
        # Early stopping and model checkpoint
        if is_loss_metric:
            improved = (best_val_metric - val_loss) > early_stopping_min_delta
            current_metric = val_loss
        else:
            improved = (val_acc - best_val_metric) > early_stopping_min_delta
            current_metric = val_acc

        if improved:
            best_val_metric = current_metric
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            metric_name = 'loss' if is_loss_metric else 'accuracy'
            metric_display = val_loss if is_loss_metric else val_acc
            print(f'  ✓ New best validation {metric_name}: {metric_display:.2f}{"%" if not is_loss_metric else ""}')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f'  Early stopping triggered (patience={early_stopping_patience})')
                break
        
        print()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        metric_name = 'val_loss' if is_loss_metric else 'val_acc'
        metric_suffix = '' if is_loss_metric else '%'
        print(f"Loaded best model weights ({metric_name}: {best_val_metric:.2f}{metric_suffix})")
    
    history = {
        'train_loss': train_losses,
        'train_accuracy': train_accs,
        'val_loss': val_losses,
        'val_accuracy': val_accs
    }
    
    return model, history


def test_model(model, test_loader, label_encoder, model_name, device='cuda'):
    """Test model and generate detailed report"""
    print(f"\n{'='*60}")
    print(f"Testing {model_name} model")
    print(f"{'='*60}\n")
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc='Testing')
        for embeddings, labels in test_pbar:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            outputs = model(embeddings)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    test_acc = 100 * accuracy_score(all_labels, all_predictions)
    
    print(f"Test accuracy: {test_acc:.2f}%\n")
    
    # Classification report
    report_text = classification_report(
        all_labels,
        all_predictions,
        target_names=label_encoder.classes_,
        digits=3
    )
    report_dict = classification_report(
        all_labels,
        all_predictions,
        target_names=label_encoder.classes_,
        digits=3,
        output_dict=True
    )
    print("Classification report:")
    print(report_text)
    
    return test_acc, all_predictions, all_labels, report_text, report_dict


def main():
    """Main function"""
    # Set device
    print("PyTorch version:", torch.__version__)
    
    if TRAINING_CONFIG['device'] == 'cpu':
        device = torch.device('cpu')
        print("Using CPU (forced by config)")
    elif TRAINING_CONFIG['device'] == 'cuda' or TRAINING_CONFIG['device'] == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"✓ Found {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print("✓ Using GPU for training")
            print("  Training will be significantly faster on GPU!")
        else:
            device = torch.device('cpu')
            print("⚠ No GPU detected. Using CPU.")
    else:
        device = torch.device(TRAINING_CONFIG['device'])
        print(f"Using device: {device}")
    
    # Load data
    data_root = os.path.join(project_root, 'data')
    batch_size = DATA_CONFIG.get('batch_size', 64)
    train_loader, val_loader, test_loader, label_encoder, num_classes = load_data(
        test_size=DATA_CONFIG['test_size'],
        val_size=DATA_CONFIG['val_size'],
        random_state=DATA_CONFIG['random_state'],
        data_root=data_root
    )
    
    # Override num_classes from data if different
    if num_classes != DATA_CONFIG['num_classes']:
        print(f"Warning: num_classes from data ({num_classes}) differs from config ({DATA_CONFIG['num_classes']})")
        print(f"Using num_classes from data: {num_classes}")
    
    # Model configuration from config file
    embedding_type = DATA_CONFIG.get('embedding_type', 'bert')
    input_dim = DATA_CONFIG.get('embedding_dims', {}).get(
        embedding_type,
        DATA_CONFIG.get('input_dim', 768)
    )
    models_config = {}
    
    # MLP Configuration
    if MLP_CONFIG['enabled']:
        models_config['MLP'] = {
            'model': MLP(
                input_dim=input_dim,
                hidden_dims=MLP_CONFIG['hidden_dims'],
                num_classes=num_classes,
                dropout=MLP_CONFIG['dropout']
            ),
            'lr': MLP_CONFIG['learning_rate'],
            'epochs': MLP_CONFIG['epochs'],
            'weight_decay': MLP_CONFIG['weight_decay'],
            'scheduler_step_size': MLP_CONFIG['scheduler_step_size'],
            'scheduler_gamma': MLP_CONFIG['scheduler_gamma']
        }
    
    # CNN Configuration
    if CNN_CONFIG['enabled']:
        models_config['CNN'] = {
            'model': CNN(
                input_dim=input_dim,
                num_filters=CNN_CONFIG['num_filters'],
                filter_sizes=CNN_CONFIG['filter_sizes'],
                num_classes=num_classes,
                dropout=CNN_CONFIG['dropout']
            ),
            'lr': CNN_CONFIG['learning_rate'],
            'epochs': CNN_CONFIG['epochs'],
            'weight_decay': CNN_CONFIG['weight_decay'],
            'scheduler_step_size': CNN_CONFIG['scheduler_step_size'],
            'scheduler_gamma': CNN_CONFIG['scheduler_gamma']
        }
    
    # Transformer Configuration
    if TRANSFORMER_CONFIG['enabled']:
        models_config['Transformer'] = {
            'model': TransformerClassifier(
                input_dim=input_dim,
                d_model=TRANSFORMER_CONFIG['d_model'],
                nhead=TRANSFORMER_CONFIG['nhead'],
                num_layers=TRANSFORMER_CONFIG['num_layers'],
                num_classes=num_classes,
                dropout=TRANSFORMER_CONFIG['dropout'],
                seq_len=TRANSFORMER_CONFIG['seq_len']
            ),
            'lr': TRANSFORMER_CONFIG['learning_rate'],
            'epochs': TRANSFORMER_CONFIG['epochs'],
            'weight_decay': TRANSFORMER_CONFIG['weight_decay'],
            'scheduler_step_size': TRANSFORMER_CONFIG['scheduler_step_size'],
            'scheduler_gamma': TRANSFORMER_CONFIG['scheduler_gamma']
        }
    
    if not models_config:
        print("Error: No models enabled in configuration!")
        return
    
    # Create model save directory
    saved_models_dir = os.path.join(project_root, TRAINING_CONFIG['saved_models_dir'])
    os.makedirs(saved_models_dir, exist_ok=True)
    
    # Train and test all models
    results = {}
    
    for model_name, config in models_config.items():
        model = config['model']
        
        # Training
        trained_model, history = train_model(
            model_name=model_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['epochs'],
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            scheduler_step_size=config['scheduler_step_size'],
            scheduler_gamma=config['scheduler_gamma'],
            device=device,
            early_stopping_metric=TRAINING_CONFIG.get('early_stopping_metric', 'val_acc'),
            early_stopping_patience=TRAINING_CONFIG.get('early_stopping_patience', 5),
            early_stopping_min_delta=TRAINING_CONFIG.get('early_stopping_min_delta', 0.0),
        )
        
        # Save model
        if TRAINING_CONFIG['save_models']:
            try:
                model_path = os.path.join(saved_models_dir, f'{model_name.lower()}_model.pth')
                torch.save(trained_model.state_dict(), model_path)
                print(f"\nModel saved to: {model_path}")
            except Exception as e:
                print(f"\nWarning: Failed to save model: {e}")
                print("Training completed but model was not saved.")
        
        # Testing
        test_acc, predictions, true_labels, report_text, report_dict = test_model(
            trained_model, 
            test_loader, 
            label_encoder, 
            model_name,
            device=device
        )
        
        results[model_name] = {
            'test_accuracy': test_acc,
            'history': history,
            'predictions': predictions,
            'true_labels': true_labels,
            'classification_report_text': report_text,
            'classification_report': report_dict,
        }
    
    # Print summary
    print(f"\n{'='*60}")
    print("Model Performance Summary")
    print(f"{'='*60}\n")
    print(f"{'Model':<15} {'Test Accuracy':<15}")
    print("-" * 30)
    for model_name, result in results.items():
        print(f"{model_name:<15} {result['test_accuracy']:.2f}%")
    
    print(f"\nAll models training completed!")

    # Save results for analysis
    analysis_dir = os.path.join(project_root, 'analysis')
    results_dir = os.path.join(analysis_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(results_dir, f'results_{timestamp}.json')

    serializable_results = {}
    for model_name, result in results.items():
        history_serializable = {
            k: [float(x) for x in v]
            for k, v in result['history'].items()
        }
        serializable_results[model_name] = {
            'test_accuracy': float(result['test_accuracy']),
            'history': history_serializable,
            'classification_report_text': result.get('classification_report_text', ''),
            'classification_report': result.get('classification_report', {}),
        }

    metadata = {
        'embedding_type': DATA_CONFIG.get('embedding_type', 'bert'),
        'embedding_dim': DATA_CONFIG.get('embedding_dims', {}).get(
            DATA_CONFIG.get('embedding_type', 'bert'),
            DATA_CONFIG.get('input_dim', 768)
        ),
        'num_classes': DATA_CONFIG.get('num_classes'),
        'timestamp': timestamp,
    }

    to_save = {
        'metadata': metadata,
        'results': serializable_results,
    }

    try:
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(to_save, f, indent=2)
        print(f"Results saved to: {results_path}")
    except Exception as e:
        print(f"Warning: failed to save results for analysis: {e}")


if __name__ == "__main__":
    main()
