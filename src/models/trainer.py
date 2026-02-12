"""
PCB Defect Classification - Trainer Module
===========================================

This module provides a comprehensive training framework for the PCB defect
classification model, including:

- Training and validation loops
- Learning rate scheduling
- Early stopping with patience
- Checkpoint saving and loading
- Training history logging
- TensorBoard integration
- Gradient clipping
- Mixed precision training (optional)

Features:
---------
- Clean, modular training API
- Configurable training parameters
- Progress visualization
- Model checkpointing
- History tracking and export

Author: PCB Defect Detection Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import json
import time
from collections import defaultdict


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Training parameters
    num_epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    
    # Learning rate scheduling
    scheduler_type: str = 'cosine'  # 'cosine', 'step', 'plateau'
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    warmup_epochs: int = 3
    min_lr: float = 1e-7
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.001
    
    # Regularization
    gradient_clip: float = 1.0
    label_smoothing: float = 0.1
    
    # Mixed precision training
    use_amp: bool = False  # Set True for GPU training
    
    # Checkpointing
    save_dir: str = 'models'
    save_best_only: bool = True
    save_frequency: int = 5
    
    # Logging
    log_frequency: int = 10
    use_tensorboard: bool = True
    tensorboard_dir: str = 'runs'
    
    # Device
    device: str = 'auto'  # 'auto', 'cuda', 'mps', 'cpu'
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'scheduler_type': self.scheduler_type,
            'early_stopping': self.early_stopping,
            'patience': self.patience,
            'gradient_clip': self.gradient_clip,
            'label_smoothing': self.label_smoothing,
            'use_amp': self.use_amp
        }


@dataclass
class TrainingHistory:
    """Training history tracker."""
    
    train_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    
    def append(
        self, 
        train_loss: float, 
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float,
        epoch_time: float
    ) -> None:
        """Append metrics for an epoch."""
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)
    
    def get_best_metrics(self) -> Dict:
        """Get best metrics achieved."""
        if not self.val_acc:
            return {}
        
        best_idx = int(np.argmax(self.val_acc))  # Convert to Python int
        return {
            'best_epoch': int(best_idx + 1),
            'best_val_acc': float(self.val_acc[best_idx]),
            'best_val_loss': float(self.val_loss[best_idx]),
            'best_train_acc': float(self.train_acc[best_idx]),
            'best_train_loss': float(self.train_loss[best_idx])
        }
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export."""
        return {
            'train_loss': self.train_loss,
            'train_acc': self.train_acc,
            'val_loss': self.val_loss,
            'val_acc': self.val_acc,
            'learning_rates': self.learning_rates,
            'epoch_times': self.epoch_times,
            'best_metrics': self.get_best_metrics()
        }
    
    def save(self, path: str) -> None:
        """Save history to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingHistory':
        """Load history from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        history = cls()
        history.train_loss = data.get('train_loss', [])
        history.train_acc = data.get('train_acc', [])
        history.val_loss = data.get('val_loss', [])
        history.val_acc = data.get('val_acc', [])
        history.learning_rates = data.get('learning_rates', [])
        history.epoch_times = data.get('epoch_times', [])
        
        return history


# ============================================================
# EARLY STOPPING
# ============================================================

class EarlyStopping:
    """
    Early stopping handler.
    
    Monitors validation metric and stops training when no improvement
    is observed for a specified number of epochs.
    """
    
    def __init__(
        self, 
        patience: int = 10, 
        min_delta: float = 0.001,
        mode: str = 'max'
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' to maximize metric, 'min' to minimize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.best_value = None
        self.counter = 0
        self.stopped = False
        self.best_epoch = 0
    
    def __call__(self, value: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            value: Current metric value
            epoch: Current epoch number
            
        Returns:
            True if training should stop
        """
        if self.best_value is None:
            self.best_value = value
            self.best_epoch = epoch
            return False
        
        # Check improvement
        if self.mode == 'max':
            improved = value > (self.best_value + self.min_delta)
        else:
            improved = value < (self.best_value - self.min_delta)
        
        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            
            if self.counter >= self.patience:
                self.stopped = True
                return True
        
        return False
    
    def reset(self) -> None:
        """Reset early stopping state."""
        self.best_value = None
        self.counter = 0
        self.stopped = False
        self.best_epoch = 0


# ============================================================
# LEARNING RATE SCHEDULERS
# ============================================================

class WarmupCosineScheduler:
    """
    Learning rate scheduler with warmup and cosine annealing.
    """
    
    def __init__(
        self, 
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        min_lr: float = 1e-7
    ):
        """
        Initialize scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of warmup epochs
            max_epochs: Total training epochs
            min_lr: Minimum learning rate
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
    
    def step(self, epoch: int) -> float:
        """
        Update learning rate for epoch.
        
        Args:
            epoch: Current epoch (0-indexed)
            
        Returns:
            Current learning rate
        """
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


# ============================================================
# TRAINER CLASS
# ============================================================

class PCBDefectTrainer:
    """
    Trainer class for PCB Defect Classification model.
    
    Provides a complete training pipeline with:
    - Training and validation loops
    - Learning rate scheduling
    - Early stopping
    - Checkpointing
    - Logging and visualization
    
    Example:
    --------
    >>> model = EfficientNetClassifier(config)
    >>> trainer = PCBDefectTrainer(model)
    >>> history = trainer.train(train_loader, val_loader)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainingConfig] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            config: Training configuration
        """
        self.config = config or TrainingConfig()
        
        # Setup device
        self.device = self._setup_device()
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Initialize training state
        self.history = TrainingHistory()
        self.best_val_acc = 0.0
        self.best_model_state = None
        self.current_epoch = 0
        
        # Setup directories
        self.save_dir = Path(self.config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = None
        if self.config.use_tensorboard:
            log_dir = Path(self.config.tensorboard_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
            self.writer = SummaryWriter(log_dir)
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.config.use_amp else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta,
            mode='max'
        ) if self.config.early_stopping else None
    
    def _setup_device(self) -> torch.device:
        """Setup computing device."""
        if self.config.device == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        else:
            device = torch.device(self.config.device)
        
        return device
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer."""
        # Use AdamW for better weight decay handling
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        return optimizer
    
    def _setup_scheduler(
        self, 
        optimizer: optim.Optimizer
    ) -> Optional[object]:
        """Setup learning rate scheduler."""
        if self.config.scheduler_type == 'cosine':
            return WarmupCosineScheduler(
                optimizer,
                warmup_epochs=self.config.warmup_epochs,
                max_epochs=self.config.num_epochs,
                min_lr=self.config.min_lr
            )
        elif self.config.scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience,
                min_lr=self.config.min_lr,
                verbose=True
            )
        elif self.config.scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.scheduler_patience,
                gamma=self.config.scheduler_factor
            )
        
        return None
    
    def _setup_criterion(
        self, 
        class_weights: Optional[torch.Tensor] = None
    ) -> nn.Module:
        """Setup loss function."""
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=self.config.label_smoothing
        )
        
        return criterion
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer
    ) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass (with optional mixed precision)
            if self.config.use_amp and self.scaler is not None:
                with autocast():
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip
                    )
                
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip
                    )
                
                optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = running_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(
        self,
        dataloader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """
        Validate model.
        
        Args:
            dataloader: Validation data loader
            criterion: Loss function
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = running_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(
        self,
        optimizer: optim.Optimizer,
        epoch: int,
        is_best: bool = False
    ) -> None:
        """
        Save training checkpoint.
        
        Args:
            optimizer: Optimizer state
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history.to_dict(),
            'config': self.config.to_dict()
        }
        
        # Save checkpoint
        if is_best:
            path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, path)
            
            # Also save just the model weights
            torch.save(self.model.state_dict(), self.save_dir / 'best_model_weights.pth')
        
        # Save periodic checkpoint
        if epoch % self.config.save_frequency == 0:
            path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> int:
        """
        Load training checkpoint.
        
        Args:
            path: Path to checkpoint
            
        Returns:
            Starting epoch
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        
        if 'history' in checkpoint:
            history_data = checkpoint['history']
            self.history = TrainingHistory()
            self.history.train_loss = history_data.get('train_loss', [])
            self.history.train_acc = history_data.get('train_acc', [])
            self.history.val_loss = history_data.get('val_loss', [])
            self.history.val_acc = history_data.get('val_acc', [])
        
        return checkpoint.get('epoch', 0) + 1
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: Optional[torch.Tensor] = None,
        resume_from: Optional[str] = None
    ) -> TrainingHistory:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            class_weights: Optional class weights for imbalance
            resume_from: Optional checkpoint path to resume
            
        Returns:
            Training history
        """
        # Setup
        optimizer = self._setup_optimizer()
        scheduler = self._setup_scheduler(optimizer)
        criterion = self._setup_criterion(class_weights)
        
        # Resume from checkpoint
        start_epoch = 0
        if resume_from and Path(resume_from).exists():
            start_epoch = self.load_checkpoint(resume_from)
            print(f"Resumed from epoch {start_epoch}")
        
        # Training info
        self._print_training_header(train_loader, val_loader)
        
        training_start_time = time.time()
        
        for epoch in range(start_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Update learning rate
            if isinstance(scheduler, WarmupCosineScheduler):
                current_lr = scheduler.step(epoch)
            else:
                current_lr = optimizer.param_groups[0]['lr']
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            epoch_time = time.time() - epoch_start_time
            
            # Update history
            self.history.append(
                train_loss, train_acc,
                val_loss, val_acc,
                current_lr, epoch_time
            )
            
            # Logging
            self._log_epoch(epoch, train_loss, train_acc, val_loss, val_acc, current_lr, epoch_time)
            
            # TensorBoard logging
            if self.writer:
                self.writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
                self.writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # LR scheduler step (for plateau scheduler)
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            elif isinstance(scheduler, optim.lr_scheduler.StepLR):
                scheduler.step()
            
            # Save best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                self.save_checkpoint(optimizer, epoch, is_best=True)
                print(f"  ✓ New best model! (Val Acc: {val_acc:.2f}%)")
            
            # Early stopping
            if self.early_stopping and self.early_stopping(val_acc, epoch):
                print(f"\n⏹ Early stopping triggered at epoch {epoch + 1}")
                print(f"  Best epoch: {self.early_stopping.best_epoch + 1}")
                print(f"  Best val acc: {self.best_val_acc:.2f}%")
                break
        
        # Training complete
        total_time = time.time() - training_start_time
        self._print_training_summary(total_time)
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Save final artifacts
        self.history.save(self.save_dir / 'training_history.json')
        
        # Close TensorBoard
        if self.writer:
            self.writer.close()
        
        return self.history
    
    def _print_training_header(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader
    ) -> None:
        """Print training header with configuration."""
        print("\n" + "=" * 70)
        print("PCB DEFECT CLASSIFICATION - TRAINING")
        print("=" * 70)
        print(f"\nDevice: {self.device}")
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        print(f"Batch size: {train_loader.batch_size}")
        print(f"\nConfiguration:")
        print(f"  Epochs: {self.config.num_epochs}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Weight decay: {self.config.weight_decay}")
        print(f"  Scheduler: {self.config.scheduler_type}")
        print(f"  Early stopping: {self.config.early_stopping} (patience={self.config.patience})")
        print(f"  Label smoothing: {self.config.label_smoothing}")
        print(f"  Mixed precision: {self.config.use_amp}")
        print("=" * 70)
    
    def _log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float,
        epoch_time: float
    ) -> None:
        """Log epoch results."""
        print(f"\nEpoch [{epoch+1}/{self.config.num_epochs}] ({epoch_time:.1f}s)")
        print(f"  Train | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"  Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        print(f"  LR: {lr:.2e}")
    
    def _print_training_summary(self, total_time: float) -> None:
        """Print training summary."""
        best_metrics = self.history.get_best_metrics()
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"\nTotal time: {total_time/60:.1f} minutes")
        print(f"Total epochs: {len(self.history.train_loss)}")
        print(f"\nBest Results:")
        print(f"  Epoch: {best_metrics.get('best_epoch', 'N/A')}")
        print(f"  Val Accuracy: {best_metrics.get('best_val_acc', 0):.2f}%")
        print(f"  Val Loss: {best_metrics.get('best_val_loss', 0):.4f}")
        print(f"  Train Accuracy: {best_metrics.get('best_train_acc', 0):.2f}%")
        print(f"\nModel saved to: {self.save_dir}")
        print("=" * 70)


# ============================================================
# VISUALIZATION UTILITIES
# ============================================================

def plot_training_history(
    history: TrainingHistory,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot training history curves.
    
    Args:
        history: Training history object
        save_path: Optional path to save figure
        show: Whether to display plot
    """
    import matplotlib.pyplot as plt
    
    epochs = range(1, len(history.train_loss) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss plot
    axes[0].plot(epochs, history.train_loss, 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history.val_loss, 'r-', label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history.train_acc, 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, history.val_acc, 'r-', label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Mark best epoch
    if history.val_acc:
        best_epoch = np.argmax(history.val_acc)
        best_acc = history.val_acc[best_epoch]
        axes[1].axhline(y=best_acc, color='g', linestyle='--', alpha=0.5)
        axes[1].scatter([best_epoch + 1], [best_acc], color='g', s=100, zorder=5, marker='*')
        axes[1].annotate(f'Best: {best_acc:.1f}%', xy=(best_epoch + 1, best_acc),
                        xytext=(5, 5), textcoords='offset points', fontsize=10, color='green')
    
    # Learning rate plot
    axes[2].plot(epochs, history.learning_rates, 'g-', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Learning Rate', fontsize=12)
    axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale('log')
    
    plt.suptitle('Training History', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Training curves saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    print("Testing PCB Defect Trainer Module...")
    print()
    
    # Create a simple test model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 128 * 128, 256),
        nn.ReLU(),
        nn.Linear(256, 6)
    )
    
    # Create config
    config = TrainingConfig(
        num_epochs=2,
        learning_rate=0.001,
        save_dir='models/test'
    )
    
    # Create trainer
    trainer = PCBDefectTrainer(model, config)
    
    print("✓ Trainer created successfully")
    print(f"Device: {trainer.device}")
    print(f"Save directory: {trainer.save_dir}")
    
    # Test early stopping
    early_stop = EarlyStopping(patience=3)
    for i, val in enumerate([0.8, 0.85, 0.84, 0.83, 0.82, 0.81]):
        stopped = early_stop(val, i)
        print(f"Epoch {i+1}: val_acc={val:.2f}, stopped={stopped}")
        if stopped:
            break
    
    print("\n✓ All tests passed!")
