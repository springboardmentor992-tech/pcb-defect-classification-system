"""
PCB Defect Classification - Models Package
===========================================

This package contains the CNN model architecture, dataset utilities,
training framework, and evaluation tools for PCB defect classification.

Modules:
--------
- cnn_model: EfficientNet-based classifier architecture
- pcb_dataset: Dataset and DataLoader utilities
- trainer: Training framework with early stopping and logging
- evaluator: Model evaluation and reporting

Main Classes:
------------
- EfficientNetClassifier: Main CNN model
- PCBDefectDataset: PyTorch Dataset for PCB images
- PCBDefectTrainer: Training framework
- ModelEvaluator: Evaluation and reporting
- ModelConfig: Model configuration
- DataConfig: Data configuration
- TrainingConfig: Training configuration
- EvaluationConfig: Evaluation configuration

Example:
--------
>>> from src.models import EfficientNetClassifier, PCBDefectDataset, PCBDefectTrainer
>>> 
>>> # Create model
>>> model = EfficientNetClassifier()
>>> 
>>> # Create dataset
>>> dataset = PCBDefectDataset('data/dataset', split='train')
>>> 
>>> # Train
>>> trainer = PCBDefectTrainer(model)
>>> history = trainer.train(train_loader, val_loader)
>>> 
>>> # Evaluate
>>> evaluator = ModelEvaluator(model)
>>> results = evaluator.evaluate(test_loader)

Author: PCB Defect Detection Team
Version: 1.0.0
"""

# Model
from .cnn_model import (
    EfficientNetClassifier,
    ModelConfig,
    ClassificationHead,
    SimpleCNN,
    ResNetClassifier,
    create_model
)

# Dataset
from .pcb_dataset import (
    PCBDefectDataset,
    DataConfig,
    DataLoaderFactory,
    TransformBuilder,
    visualize_batch,
    visualize_augmentations
)

# Trainer
from .trainer import (
    PCBDefectTrainer,
    TrainingConfig,
    TrainingHistory,
    EarlyStopping,
    WarmupCosineScheduler,
    plot_training_history
)

# Evaluator
from .evaluator import (
    ModelEvaluator,
    EvaluationConfig,
    EvaluationResults
)


__all__ = [
    # Model
    'EfficientNetClassifier',
    'ModelConfig',
    'ClassificationHead',
    'SimpleCNN',
    'ResNetClassifier',
    'create_model',
    
    # Dataset
    'PCBDefectDataset',
    'DataConfig',
    'DataLoaderFactory',
    'TransformBuilder',
    'visualize_batch',
    'visualize_augmentations',
    
    # Trainer
    'PCBDefectTrainer',
    'TrainingConfig',
    'TrainingHistory',
    'EarlyStopping',
    'WarmupCosineScheduler',
    'plot_training_history',
    
    # Evaluator
    'ModelEvaluator',
    'EvaluationConfig',
    'EvaluationResults'
]

__version__ = '1.0.0'

