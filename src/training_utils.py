import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
import wandb
import numpy as np
import time
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Callable
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class EarlyStopping:
    
    def __init__(self, patience: int = 7, min_delta: float = 0.001, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

class GradientClipper:
    
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def __call__(self, model: nn.Module) -> float:
        return torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                            self.max_norm, self.norm_type)

def get_optimizer(model: nn.Module, optimizer_name: str, lr: float, **kwargs) -> optim.Optimizer:
    optimizers = {
        'sgd': lambda: optim.SGD(model.parameters(), lr=lr, 
                                momentum=kwargs.get('momentum', 0.0),
                                weight_decay=kwargs.get('weight_decay', 0.0),
                                nesterov=kwargs.get('nesterov', False)),
        
        'adam': lambda: optim.Adam(model.parameters(), lr=lr,
                                  betas=kwargs.get('betas', (0.9, 0.999)),
                                  weight_decay=kwargs.get('weight_decay', 0.0)),
        
        'adamw': lambda: optim.AdamW(model.parameters(), lr=lr,
                                    betas=kwargs.get('betas', (0.9, 0.999)),
                                    weight_decay=kwargs.get('weight_decay', 0.01)),
        
        'rmsprop': lambda: optim.RMSprop(model.parameters(), lr=lr,
                                        alpha=kwargs.get('alpha', 0.99),
                                        weight_decay=kwargs.get('weight_decay', 0.0)),
        
        'adagrad': lambda: optim.Adagrad(model.parameters(), lr=lr,
                                        weight_decay=kwargs.get('weight_decay', 0.0))
    }
    
    if optimizer_name.lower() not in optimizers:
        raise ValueError(f"Optimizer {optimizer_name} not supported")
    
    return optimizers[optimizer_name.lower()]()

def get_scheduler(optimizer: optim.Optimizer, scheduler_name: str, **kwargs):
    schedulers = {
        'step': lambda: StepLR(optimizer, 
                              step_size=kwargs.get('step_size', 10),
                              gamma=kwargs.get('gamma', 0.1)),
        
        'exponential': lambda: ExponentialLR(optimizer,
                                           gamma=kwargs.get('gamma', 0.95)),
        
        'cosine': lambda: CosineAnnealingLR(optimizer,
                                          T_max=kwargs.get('T_max', 50)),
        
        'plateau': lambda: ReduceLROnPlateau(optimizer,
                                           mode='min',
                                           factor=kwargs.get('factor', 0.5),
                                           patience=kwargs.get('patience', 5))
    }
    
    if scheduler_name.lower() not in schedulers:
        return None
    
    return schedulers[scheduler_name.lower()]()

class ModelTrainer:
    
    def __init__(self, model: nn.Module, train_loader, val_loader,
                 criterion, optimizer, scheduler=None, device='cuda',
                 experiment_name: str = 'experiment', run_name: str = 'run',
                 use_wandb: bool = True):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_wandb = use_wandb
        
        if use_wandb:
            wandb.init(
                project="facial-expression-recognition",
                group=experiment_name,
                name=run_name,
                reinit=True
            )
            wandb.watch(self.model, log='all', log_freq=100)
        
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rates': []
        }
        
        self.best_val_acc = 0.0
        self.best_model_state = None
    
    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self) -> Tuple[float, float, np.ndarray, np.ndarray]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in tqdm(self.val_loader, desc="Validation"):
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, np.array(all_predictions), np.array(all_targets)
    
    def train(self, epochs: int, early_stopping: Optional[EarlyStopping] = None,
              gradient_clipper: Optional[GradientClipper] = None,
              save_best: bool = True) -> Dict:
        
        print(f"Starting training for {epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            train_loss, train_acc = self.train_epoch()
            
            val_loss, val_acc, val_preds, val_targets = self.validate_epoch()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            if save_best and val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            if self.use_wandb:
                log_dict = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'learning_rate': current_lr,
                    'epoch_time': time.time() - epoch_start_time
                }
                
                if (epoch + 1) % 10 == 0:
                    log_dict['confusion_matrix'] = wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=val_targets,
                        preds=val_preds,
                        class_names=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
                    )
                
                wandb.log(log_dict)
            
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")
            print(f"  Time: {time.time() - epoch_start_time:.2f}s")
            print("-" * 50)
            
            if early_stopping and early_stopping(val_loss, self.model):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s")
        
        if save_best and self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            print(f"Loaded best model with validation accuracy: {self.best_val_acc:.2f}%")
        
        return self.history
    
    def plot_training_history(self, save_path: Optional[str] = None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(self.history['train_acc'], label='Train Acc')
        axes[0, 1].plot(self.history['val_acc'], label='Val Acc')
        axes[0, 1].set_title('Accuracy Over Time')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(self.history['learning_rates'])
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        loss_diff = np.array(self.history['val_loss']) - np.array(self.history['train_loss'])
        axes[1, 1].plot(loss_diff)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Overfitting Indicator (Val Loss - Train Loss)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Difference')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def gradient_check(model: nn.Module, data_loader, criterion, device='cuda', epsilon=1e-7):
    model.eval()
    
    data, targets = next(iter(data_loader))
    data, targets = data[:2].to(device), targets[:2].to(device)  # Use only 2 samples
    
    params = [p for p in model.parameters() if p.requires_grad]
    
    print("Performing gradient check...")
    
    for i, param in enumerate(params):
        if param.numel() > 100:  # Only check first few parameters of large tensors
            flat_param = param.flatten()[:10]
            param_indices = [(0, j) for j in range(10)]
        else:
            flat_param = param.flatten()
            param_indices = [(0, j) for j in range(param.numel())]
        
        print(f"Checking parameter {i+1}/{len(params)} (shape: {param.shape})...")
        
        for idx, param_idx in enumerate(param_indices[:5]):  # Check first 5 elements
            param.data.flatten()[param_idx[1]] += epsilon
            loss_plus = criterion(model(data), targets)
            
            param.data.flatten()[param_idx[1]] -= 2 * epsilon
            loss_minus = criterion(model(data), targets)
            
            param.data.flatten()[param_idx[1]] += epsilon  # Restore original value
            
            numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
            
            model.zero_grad()
            loss = criterion(model(data), targets)
            loss.backward()
            
            analytical_grad = param.grad.flatten()[param_idx[1]]
            
            relative_error = abs(numerical_grad - analytical_grad) / (abs(numerical_grad) + abs(analytical_grad) + 1e-8)
            
            if idx == 0:  # Print only first element to avoid spam
                print(f"  Element {param_idx}: Numerical={numerical_grad:.8f}, "
                      f"Analytical={analytical_grad:.8f}, RelError={relative_error:.8f}")
                
                if relative_error > 1e-4:
                    print(f"  WARNING: Large gradient error!")
