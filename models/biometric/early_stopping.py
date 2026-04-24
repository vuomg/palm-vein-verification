"""
Early stopping callback implementation for deep learning training.
Monitors validation metrics and stops training when improvement plateaus.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import os
import logging


class EarlyStopping:
    """
    Early stopping callback to monitor training and stop when validation metric stops improving.
    Supports multiple metrics and restoration of best model weights.
    """
    
    def __init__(self, 
                 patience: int = 15,
                 min_delta: float = 0.001,
                 monitor: str = 'val_eer',
                 mode: str = 'min',
                 restore_best_weights: bool = True,
                 save_best_model: bool = True,
                 model_save_path: Optional[str] = None,
                 verbose: int = 1):
        """
        Args:
            patience: Number of epochs with no improvement after which training will be stopped
            min_delta: Minimum change in the monitored quantity to qualify as an improvement
            monitor: Metric to monitor ('val_eer', 'val_auc', 'val_loss', etc.)
            mode: 'min' for metrics where lower is better, 'max' for metrics where higher is better
            restore_best_weights: Whether to restore model weights from the best epoch
            save_best_model: Whether to save the best model to disk
            model_save_path: Path to save the best model (if save_best_model=True)
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.save_best_model = save_best_model
        self.model_save_path = model_save_path
        self.verbose = verbose
        
        # Internal state
        self.wait = 0
        self.stopped_epoch = 0
        self.best_epoch = 0
        self.best_weights = None
        self.best_metric_value = None
        
        # Determine comparison operator
        if mode == 'min':
            self.monitor_op = np.less
            self.best_metric_value = np.inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best_metric_value = -np.inf
        else:
            raise ValueError(f"Mode {mode} not supported. Use 'min' or 'max'.")
        
        if self.verbose >= 1:
            print(f"EarlyStopping: monitoring '{monitor}' with patience {patience}")
    
    def __call__(self, 
                 epoch: int, 
                 metrics: Dict[str, float], 
                 model: nn.Module) -> bool:
        """
        Check if training should stop and update best model if needed.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of validation metrics
            model: PyTorch model to monitor
        
        Returns:
            True if training should stop, False otherwise
        """
        current_value = metrics.get(self.monitor)
        
        if current_value is None:
            if self.verbose >= 1:
                print(f"Warning: Early stopping metric '{self.monitor}' not found in metrics")
            return False
        
        # Check if current value is better than best
        # For 'max' mode: current > (best + min_delta)
        # For 'min' mode: current < (best - min_delta)
        # Handle special case when best_metric_value is initial value (-inf or inf)
        if self.mode == 'max':
            if self.best_metric_value == -np.inf:
                is_better = True  # Any value is better than -inf
            else:
                is_better = current_value > (self.best_metric_value + self.min_delta)
        else:  # mode == 'min'
            if self.best_metric_value == np.inf:
                is_better = True  # Any value is better than inf
            else:
                is_better = current_value < (self.best_metric_value - self.min_delta)
        
        if is_better:
            self.best_metric_value = current_value
            self.best_epoch = epoch
            self.wait = 0
            
            # Save best weights
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            
            # Save best model to disk
            if self.save_best_model and self.model_save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'metrics': metrics,
                    'best_metric_value': self.best_metric_value
                }, self.model_save_path)
            
            if self.verbose >= 2:
                direction = "decreased" if self.mode == 'min' else "increased"
                print(f"Epoch {epoch}: {self.monitor} {direction} to {current_value:.6f}")
        
        else:
            self.wait += 1
            if self.verbose >= 2:
                print(f"Epoch {epoch}: {self.monitor} did not improve from {self.best_metric_value:.6f}")
        
        # Check if we should stop
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            if self.verbose >= 1:
                print(f"Early stopping at epoch {epoch}")
                print(f"Best {self.monitor}: {self.best_metric_value:.6f} at epoch {self.best_epoch}")
            
            # Restore best weights if requested
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                if self.verbose >= 1:
                    print(f"Restored model weights from epoch {self.best_epoch}")
            
            return True
        
        return False
    
    def get_best_score(self) -> float:
        """Get the best metric value achieved."""
        return self.best_metric_value
    
    def get_best_epoch(self) -> int:
        """Get the epoch with the best metric value."""
        return self.best_epoch


class LearningRateScheduler:
    """
    Advanced learning rate scheduler with multiple scheduling strategies.
    """
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 scheduler_type: str = 'cosine_warm_restarts',
                 **scheduler_kwargs):
        """
        Args:
            optimizer: PyTorch optimizer
            scheduler_type: Type of scheduler to use
            **scheduler_kwargs: Additional arguments for the scheduler
        """
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        
        # Create appropriate scheduler
        if scheduler_type == 'cosine_warm_restarts':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 
                T_0=scheduler_kwargs.get('T_0', 10),
                T_mult=scheduler_kwargs.get('T_mult', 2),
                eta_min=scheduler_kwargs.get('eta_min', 1e-6)
            )
            
        elif scheduler_type == 'reduce_on_plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=scheduler_kwargs.get('mode', 'min'),
                factor=scheduler_kwargs.get('factor', 0.5),
                patience=scheduler_kwargs.get('patience', 7),
                min_lr=scheduler_kwargs.get('min_lr', 1e-6)
            )
            
        elif scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_kwargs.get('step_size', 30),
                gamma=scheduler_kwargs.get('gamma', 0.1)
            )
            
        elif scheduler_type == 'exponential':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=scheduler_kwargs.get('gamma', 0.95)
            )
            
        elif scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_kwargs.get('T_max', 100),
                eta_min=scheduler_kwargs.get('eta_min', 1e-6)
            )
            
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
    def step(self, epoch: Optional[int] = None, metrics: Optional[Dict[str, float]] = None):
        """Step the learning rate scheduler."""
        if self.scheduler_type == 'reduce_on_plateau':
            if metrics is None:
                raise ValueError("ReduceLROnPlateau requires metrics")
            # Assume we're monitoring validation loss by default
            metric_value = metrics.get('val_loss', metrics.get('val_eer', 0))
            self.scheduler.step(metric_value)
        else:
            self.scheduler.step()
    
    def get_last_lr(self) -> float:
        """Get the last learning rate."""
        return self.scheduler.get_last_lr()[0]


class TrainingMonitor:
    """
    Monitor training progress and log metrics.
    """
    
    def __init__(self, 
                 log_dir: str = 'logs',
                 save_plots: bool = True,
                 plot_frequency: int = 10):
        """
        Args:
            log_dir: Directory to save logs and plots
            save_plots: Whether to save training plots
            plot_frequency: Frequency of plot updates (in epochs)
        """
        self.log_dir = log_dir
        self.save_plots = save_plots
        self.plot_frequency = plot_frequency
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Training history
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_eer': [],
            'val_auc': [],
            'learning_rate': []
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_epoch(self, 
                  epoch: int,
                  train_loss: float,
                  val_metrics: Dict[str, float],
                  learning_rate: float):
        """Log metrics for an epoch."""
        # Update history
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_metrics.get('val_loss', 0))
        self.history['val_eer'].append(val_metrics.get('val_eer', 0))
        self.history['val_auc'].append(val_metrics.get('val_auc', 0))
        self.history['learning_rate'].append(learning_rate)
        
        # Log to console and file with timestamp
        from datetime import datetime
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Only log val_eer and val_auc if they are non-zero (i.e., validation is enabled)
        if val_metrics.get('val_eer', 0) > 0 or val_metrics.get('val_auc', 0) > 0:
            # Full log with validation metrics
            self.logger.info(
                f"[{current_time}] Epoch {epoch:3d} - "
                f"Train Loss: {train_loss:.6f} - "
                f"Val EER: {val_metrics.get('val_eer', 0):.6f} - "
                f"Val AUC: {val_metrics.get('val_auc', 0):.6f} - "
                f"LR: {learning_rate:.2e}"
            )
        else:
            # Simplified log without validation metrics (no-validation mode)
            self.logger.info(
                f"[{current_time}] Epoch {epoch:3d} - "
                f"Train Loss: {train_loss:.6f} - "
                f"LR: {learning_rate:.2e}"
            )
        
        # Save plots periodically
        if self.save_plots and epoch % self.plot_frequency == 0:
            self._save_training_plots()
    
    def _save_training_plots(self):
        """Save training progress plots."""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss plot
            axes[0, 0].plot(self.history['epoch'], self.history['train_loss'], 'b-', label='Train Loss')
            axes[0, 0].plot(self.history['epoch'], self.history['val_loss'], 'r-', label='Val Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # EER plot
            axes[0, 1].plot(self.history['epoch'], self.history['val_eer'], 'g-', label='Val EER')
            axes[0, 1].set_title('Validation EER')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('EER')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # AUC plot
            axes[1, 0].plot(self.history['epoch'], self.history['val_auc'], 'm-', label='Val AUC')
            axes[1, 0].set_title('Validation AUC')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('AUC')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Learning rate plot
            axes[1, 1].plot(self.history['epoch'], self.history['learning_rate'], 'orange', label='Learning Rate')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, 'training_progress.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except ImportError:
            pass  # matplotlib not available
    
    def get_history(self) -> Dict[str, list]:
        """Get training history."""
        return self.history.copy()
    
    def save_history(self, filepath: str = None):
        """Save training history to file."""
        if filepath is None:
            filepath = os.path.join(self.log_dir, 'training_history.json')
        
        import json
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)


if __name__ == "__main__":
    # Test early stopping
    print("Testing Early Stopping callback...")
    
    # Mock model and metrics for testing
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
    
    model = MockModel()
    
    # Test early stopping
    early_stopping = EarlyStopping(
        patience=3,
        monitor='val_eer',
        mode='min',
        min_delta=0.001,
        verbose=2
    )
    
    # Simulate training with improving then plateauing metrics
    test_metrics = [
        {'val_eer': 0.1, 'val_auc': 0.85},
        {'val_eer': 0.08, 'val_auc': 0.87},  # Improvement
        {'val_eer': 0.06, 'val_auc': 0.89},  # Improvement
        {'val_eer': 0.065, 'val_auc': 0.88}, # No improvement
        {'val_eer': 0.067, 'val_auc': 0.87}, # No improvement
        {'val_eer': 0.069, 'val_auc': 0.86}, # No improvement (should stop)
    ]
    
    for epoch, metrics in enumerate(test_metrics):
        should_stop = early_stopping(epoch, metrics, model)
        if should_stop:
            print(f"Training stopped at epoch {epoch}")
            break
    
    print(f"Best EER: {early_stopping.get_best_score():.6f} at epoch {early_stopping.get_best_epoch()}")
    print("Early stopping test completed successfully!")
