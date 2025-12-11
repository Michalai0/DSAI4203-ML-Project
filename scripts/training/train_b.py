"""
Training script using PyTorch
Supports training three models: MLP, CNN, and Transformer
"""
import json
import os
import sys
from datetime import datetime

import itertools
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import f1_score #Ken
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau #Ken
from tqdm import tqdm

# Add project root directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# Import configuration
from config import (
    DATA_CONFIG, MLP_CONFIG, CNN_CONFIG, CNN_ENSEMBLE_CONFIG,
    TRANSFORMER_CONFIG, TRAINING_CONFIG
)

from src.data.data_loader import load_data
from src.models.models import MLP, CNN, CNNEnsemble, TransformerClassifier


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
    #early_stopping_metric='val_acc',
    early_stopping_metric='val_f1_macro', #Ken
    early_stopping_patience=5,
    early_stopping_min_delta=0.0,
    class_weights=None #Ken
):
    """Train model"""
    print(f"\n{'='*60}")
    print(f"Starting training for {model_name} model")
    print(f"{'='*60}\n")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=(class_weights.to(device) if class_weights is not None else None)) #Ken
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    #scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1) #Ken
    
    is_loss_metric = early_stopping_metric == 'val_loss'
    use_f1_metric = early_stopping_metric == 'val_f1_macro' #Ken
    best_val_metric = float('inf') if is_loss_metric else 0.0
    best_model_state = None
    patience_counter = 0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    val_f1_macro_list=[] #Ken
    
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
            _, predicted = torch.max(outputs, 1) #Ken
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
        y_true, y_pred = [], [] #Ken
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for embeddings, labels in val_pbar:
                embeddings = embeddings.to(device)
                labels = labels.to(device)
                
                outputs = model(embeddings)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1) #Ken
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                y_true.extend(labels.cpu().numpy()) #Ken
                y_pred.extend(predicted.cpu().numpy()) #Ken
                
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * val_correct / val_total:.2f}%'
                })
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_f1_macro = f1_score(y_true, y_pred, average='macro') #Ken
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1_macro_list.append(float(val_f1_macro)) #Ken
        
        # Learning rate scheduling
        scheduler.step(val_f1_macro) #Ken
        #current_lr = scheduler.get_last_lr()[0]
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  LR: {current_lr:.6f}')
        
        # Early stopping and model checkpoint, Ken
        if is_loss_metric:
            current_metric = val_loss
            improved = (best_val_metric - current_metric) > early_stopping_min_delta
        elif use_f1_metric:
            current_metric = val_f1_macro          #
            improved = (current_metric - best_val_metric) > early_stopping_min_delta
        else:
            current_metric = val_acc / 100.0       
            improved = (current_metric - best_val_metric) > (early_stopping_min_delta / 100.0)

        if improved:
            best_val_metric = current_metric
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            if is_loss_metric:
                metric_name, metric_display, suffix = 'loss', val_loss, ''
            elif use_f1_metric:
                metric_name, metric_display, suffix = 'f1_macro', val_f1_macro, ''
            else:
                metric_name, metric_display, suffix = 'accuracy', val_acc, '%'
            print(f'  ✓ New best validation {metric_name}: {metric_display:.4f}{suffix}')

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
        'val_accuracy': val_accs,
        'val_f1_macro': val_f1_macro_list
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
            _, predicted = torch.max(outputs, 1) #Ken
            
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

    def set_seed(seed=10):
        import random, numpy as np, torch
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(10)
    
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
    
    # CNN Ensemble Configuration
    if CNN_ENSEMBLE_CONFIG.get('enabled', False):
        models_config['CNNEnsemble'] = {
            'model': CNNEnsemble(
                input_dim=input_dim,
                variant_configs=CNN_ENSEMBLE_CONFIG.get('variants', []),
                num_classes=num_classes,
            ),
            'lr': CNN_ENSEMBLE_CONFIG['learning_rate'],
            'epochs': CNN_ENSEMBLE_CONFIG['epochs'],
            'weight_decay': CNN_ENSEMBLE_CONFIG['weight_decay'],
            'scheduler_step_size': CNN_ENSEMBLE_CONFIG['scheduler_step_size'],
            'scheduler_gamma': CNN_ENSEMBLE_CONFIG['scheduler_gamma'],
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

    # Build hyperparameter combinations (cartesian product)
    sweep_cfg = TRAINING_CONFIG.get('sweep', {})
    sweep_enabled = sweep_cfg.get('enabled', False) and sweep_cfg.get('params')
    param_grid = sweep_cfg.get('params', {}) if sweep_enabled else {}
    param_keys = list(param_grid.keys())
    param_values = [param_grid[k] for k in param_keys]
    sweep_combos = list(itertools.product(*param_values)) if sweep_enabled else [()]
    
    # Create model save directory
    saved_models_dir = os.path.join(project_root, TRAINING_CONFIG['saved_models_dir'])
    os.makedirs(saved_models_dir, exist_ok=True)
    
    # Train and test all models
    results = {}

    # Weighting Calculation, Ken

    counts = torch.zeros(num_classes, dtype=torch.long)
    for _, y in train_loader:
        if isinstance(y, torch.Tensor):
            counts += torch.bincount(y.cpu(), minlength=num_classes)
        else:
            y = torch.as_tensor(y)
            counts += torch.bincount(y, minlength=num_classes)

    total = counts.sum().item()
    weights = (total / torch.clamp(counts, min=1)).float()
    weights = weights / weights.mean()

    class_weights = weights  
    print("class_weights:", class_weights.tolist())


    
    for model_name, config in models_config.items():
        # Iterate sweep combos; if sweep is off, there is only one empty combo
        for combo in sweep_combos:
            override_params = dict(zip(param_keys, combo)) if sweep_enabled else {}
            run_tag = ""
            if sweep_enabled:
                tag_parts = [f"{k}={override_params[k]}" for k in param_keys]
                run_tag = "[" + ",".join(tag_parts) + "]"
                print(f"\n--- Sweep {run_tag} ---")

            lr = override_params.get('lr', config['lr'])
            weight_decay = override_params.get('weight_decay', config['weight_decay'])
            epochs = override_params.get('epochs', config['epochs'])
            sched_step = override_params.get('scheduler_step_size', config['scheduler_step_size'])
            sched_gamma = override_params.get('scheduler_gamma', config['scheduler_gamma'])

            # Rebuild model for each run to avoid weight carry-over
            model = copy.deepcopy(config['model'])
            
            # Training
            trained_model, history = train_model(
                model_name=model_name,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                scheduler_step_size=sched_step,
                scheduler_gamma=sched_gamma,
                device=device,
                #early_stopping_metric=TRAINING_CONFIG.get('early_stopping_metric', 'val_acc'), #Ken
                early_stopping_metric=TRAINING_CONFIG.get('early_stopping_metric', 'val_f1_macro'),
                early_stopping_patience=TRAINING_CONFIG.get('early_stopping_patience', 5),
                early_stopping_min_delta=TRAINING_CONFIG.get('early_stopping_min_delta', 0.0),
                class_weights=class_weights,
            )
            
            # Save model
            if TRAINING_CONFIG['save_models']:
                try:
                    suffix = f"_{run_tag}" if run_tag else ""
                    safe_suffix = suffix.replace("[", "").replace("]", "").replace(",", "_").replace("=", "-")
                    model_path = os.path.join(saved_models_dir, f'{model_name.lower()}_model{safe_suffix}.pth')
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

            macro_f1 = float(report_dict.get('macro avg', {}).get('f1-score', 0.0))
            weighted_f1 = float(report_dict.get('weighted avg', {}).get('f1-score', 0.0))
            
            results_key = f"{model_name}{run_tag}"
            results[results_key] = {
                'test_accuracy': test_acc,
                'macro_f1': macro_f1,
                'weighted_f1': weighted_f1,
                'run_params': {
                    'lr': lr,
                    'weight_decay': weight_decay,
                    'epochs': epochs,
                    'scheduler_step_size': sched_step,
                    'scheduler_gamma': sched_gamma,
                },
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
    # Sort results by test accuracy descending
    sorted_items = sorted(results.items(), key=lambda x: x[1].get('test_accuracy', 0), reverse=True)

    print(f"{'Model':<25} {'Test Acc':<12} {'Macro F1':<12} {'lr':<10} {'wd':<10} {'epochs':<8}")
    print("-" * 80)
    for model_key, result in sorted_items:
        macro_f1 = result.get('macro_f1', 0.0) * 100
        params = result.get('run_params', {})
        print(f"{model_key:<25} {result['test_accuracy']:.2f}%   {macro_f1:.2f}%   "
              f"{params.get('lr','-'):<10} {params.get('weight_decay','-'):<10} {params.get('epochs','-'):<8}")

    if sorted_items:
        best_key, best_res = sorted_items[0]
        best_params = best_res.get('run_params', {})
        print(f"\nBest by Test Acc: {best_key} "
              f"(acc={best_res.get('test_accuracy',0):.2f}%, "
              f"lr={best_params.get('lr')}, wd={best_params.get('weight_decay')}, epochs={best_params.get('epochs')})")
    
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
            'macro_f1': float(result.get('macro_f1', 0.0)),
            'weighted_f1': float(result.get('weighted_f1', 0.0)),
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
