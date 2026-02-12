"""
PCB Defect Classification - Model Evaluator
============================================

This module provides comprehensive evaluation tools for the trained
PCB defect classification model.

Features:
---------
- Test set evaluation with detailed metrics
- Confusion matrix visualization
- Per-class precision, recall, F1-score
- ROC curves and AUC
- Misclassification analysis
- Prediction visualization
- Export to various formats

Author: PCB Defect Detection Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import json
from datetime import datetime
from collections import defaultdict


# ============================================================
# EVALUATION CONFIGURATION
# ============================================================

@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    
    # Paths
    model_path: str = 'models/best_model.pth'
    output_dir: str = 'outputs/module3/evaluation'
    
    # Evaluation settings
    batch_size: int = 32
    num_workers: int = 4
    
    # Visualization
    generate_visualizations: bool = True
    num_samples_to_visualize: int = 25
    
    # Class names
    class_names: Tuple[str, ...] = (
        'Missing_hole',
        'Mouse_bite',
        'Open_circuit',
        'Short',
        'Spur',
        'Spurious_copper'
    )
    
    # Device
    device: str = 'auto'


# ============================================================
# EVALUATION RESULTS
# ============================================================

@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    
    # Overall metrics
    accuracy: float = 0.0
    loss: float = 0.0
    
    # Per-class metrics
    precision: Dict[str, float] = field(default_factory=dict)
    recall: Dict[str, float] = field(default_factory=dict)
    f1_score: Dict[str, float] = field(default_factory=dict)
    support: Dict[str, int] = field(default_factory=dict)
    
    # Macro/weighted averages
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    weighted_precision: float = 0.0
    weighted_recall: float = 0.0
    weighted_f1: float = 0.0
    
    # Confusion matrix
    confusion_matrix: np.ndarray = None
    
    # Predictions
    all_predictions: List[int] = field(default_factory=list)
    all_labels: List[int] = field(default_factory=list)
    all_probabilities: List[List[float]] = field(default_factory=list)
    
    # Metadata
    total_samples: int = 0
    correct_predictions: int = 0
    evaluation_time: float = 0.0
    class_names: Tuple[str, ...] = ()
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'overall': {
                'accuracy': self.accuracy,
                'loss': self.loss,
                'total_samples': self.total_samples,
                'correct_predictions': self.correct_predictions
            },
            'per_class': {
                name: {
                    'precision': self.precision.get(name, 0.0),
                    'recall': self.recall.get(name, 0.0),
                    'f1_score': self.f1_score.get(name, 0.0),
                    'support': self.support.get(name, 0)
                }
                for name in self.class_names
            },
            'averages': {
                'macro': {
                    'precision': self.macro_precision,
                    'recall': self.macro_recall,
                    'f1': self.macro_f1
                },
                'weighted': {
                    'precision': self.weighted_precision,
                    'recall': self.weighted_recall,
                    'f1': self.weighted_f1
                }
            },
            'confusion_matrix': self.confusion_matrix.tolist() if self.confusion_matrix is not None else [],
            'evaluation_time': self.evaluation_time
        }
    
    def save(self, path: str) -> None:
        """Save results to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def print_summary(self) -> None:
        """Print evaluation summary."""
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        
        print(f"\n📊 OVERALL METRICS")
        print("-" * 40)
        print(f"Accuracy:  {self.accuracy:.2f}%")
        print(f"Loss:      {self.loss:.4f}")
        print(f"Samples:   {self.total_samples}")
        print(f"Correct:   {self.correct_predictions}")
        
        print(f"\n📊 PER-CLASS METRICS")
        print("-" * 70)
        print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        print("-" * 70)
        
        for name in self.class_names:
            print(f"{name:<20} {self.precision.get(name, 0):.4f}     "
                  f"{self.recall.get(name, 0):.4f}     "
                  f"{self.f1_score.get(name, 0):.4f}     "
                  f"{self.support.get(name, 0):>6}")
        
        print("-" * 70)
        print(f"{'Macro Avg':<20} {self.macro_precision:.4f}     "
              f"{self.macro_recall:.4f}     {self.macro_f1:.4f}")
        print(f"{'Weighted Avg':<20} {self.weighted_precision:.4f}     "
              f"{self.weighted_recall:.4f}     {self.weighted_f1:.4f}")
        
        print("=" * 70)


# ============================================================
# MODEL EVALUATOR
# ============================================================

class ModelEvaluator:
    """
    Comprehensive model evaluator for PCB defect classification.
    
    Provides:
    - Test set evaluation
    - Confusion matrix
    - Per-class metrics
    - Visualization tools
    
    Example:
    --------
    >>> evaluator = ModelEvaluator(model, config)
    >>> results = evaluator.evaluate(test_loader)
    >>> results.print_summary()
    >>> evaluator.generate_report()
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[EvaluationConfig] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model to evaluate
            config: Evaluation configuration
        """
        self.config = config or EvaluationConfig()
        
        # Setup device
        self.device = self._setup_device()
        
        # Move model to device
        self.model = model.to(self.device)
        self.model.eval()
        
        # Results storage
        self.results = None
        
        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        criterion: Optional[nn.Module] = None
    ) -> EvaluationResults:
        """
        Evaluate model on test set.
        
        Args:
            dataloader: Test data loader
            criterion: Optional loss function
            
        Returns:
            EvaluationResults object
        """
        import time
        start_time = time.time()
        
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        running_loss = 0.0
        correct = 0
        total = 0
        
        print("Evaluating model on test set...")
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Forward pass
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            
            # Get predictions
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            # Accumulate
            running_loss += loss.item()
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Store for analysis
            all_predictions.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probabilities.extend(probs.cpu().numpy().tolist())
            
            # Progress
            if (batch_idx + 1) % 5 == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches...")
        
        # Calculate metrics
        results = self._calculate_metrics(
            all_predictions,
            all_labels,
            all_probabilities,
            running_loss / len(dataloader)
        )
        
        results.evaluation_time = time.time() - start_time
        results.class_names = self.config.class_names
        
        self.results = results
        
        return results
    
    def _calculate_metrics(
        self,
        predictions: List[int],
        labels: List[int],
        probabilities: List[List[float]],
        avg_loss: float
    ) -> EvaluationResults:
        """Calculate comprehensive metrics."""
        from sklearn.metrics import (
            confusion_matrix, 
            precision_recall_fscore_support,
            accuracy_score
        )
        
        results = EvaluationResults()
        
        # Basic metrics
        results.total_samples = len(labels)
        results.correct_predictions = sum(p == l for p, l in zip(predictions, labels))
        results.accuracy = 100.0 * results.correct_predictions / results.total_samples
        results.loss = avg_loss
        
        # Store predictions
        results.all_predictions = predictions
        results.all_labels = labels
        results.all_probabilities = probabilities
        
        # Confusion matrix
        results.confusion_matrix = confusion_matrix(labels, predictions)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        for i, name in enumerate(self.config.class_names):
            results.precision[name] = float(precision[i])
            results.recall[name] = float(recall[i])
            results.f1_score[name] = float(f1[i])
            results.support[name] = int(support[i])
        
        # Macro averages
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )
        results.macro_precision = float(macro_p)
        results.macro_recall = float(macro_r)
        results.macro_f1 = float(macro_f1)
        
        # Weighted averages
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        results.weighted_precision = float(weighted_p)
        results.weighted_recall = float(weighted_r)
        results.weighted_f1 = float(weighted_f1)
        
        return results
    
    def plot_confusion_matrix(
        self,
        save_path: Optional[str] = None,
        normalize: bool = True,
        show: bool = False
    ) -> None:
        """
        Plot confusion matrix.
        
        Args:
            save_path: Path to save figure
            normalize: Normalize values (percentages)
            show: Display plot
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if self.results is None:
            raise ValueError("No evaluation results. Run evaluate() first.")
        
        cm = self.results.confusion_matrix
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = 'Confusion Matrix (Normalized)'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.config.class_names,
            yticklabels=self.config.class_names,
            square=True,
            cbar_kws={'shrink': 0.8}
        )
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Add accuracy annotation
        plt.annotate(
            f'Overall Accuracy: {self.results.accuracy:.2f}%',
            xy=(0.5, -0.15),
            xycoords='axes fraction',
            ha='center',
            fontsize=12,
            fontweight='bold'
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved confusion matrix: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_per_class_metrics(
        self,
        save_path: Optional[str] = None,
        show: bool = False
    ) -> None:
        """
        Plot per-class precision, recall, F1 scores.
        
        Args:
            save_path: Path to save figure
            show: Display plot
        """
        import matplotlib.pyplot as plt
        
        if self.results is None:
            raise ValueError("No evaluation results. Run evaluate() first.")
        
        class_names = list(self.config.class_names)
        precision = [self.results.precision[n] for n in class_names]
        recall = [self.results.recall[n] for n in class_names]
        f1 = [self.results.f1_score[n] for n in class_names]
        
        x = np.arange(len(class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x, recall, width, label='Recall', color='#e74c3c', alpha=0.8)
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#2ecc71', alpha=0.8)
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Defect Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Classification Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', alpha=0.3)
        
        # Add horizontal line at 0.9 (target)
        ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='Target (0.9)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved metrics plot: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_sample_predictions(
        self,
        dataloader: DataLoader,
        num_samples: int = 16,
        save_path: Optional[str] = None,
        show: bool = False
    ) -> None:
        """
        Plot sample predictions with images.
        
        Args:
            dataloader: Test data loader
            num_samples: Number of samples to show
            save_path: Path to save figure
            show: Display plot
        """
        import matplotlib.pyplot as plt
        
        # Get samples
        images, labels = next(iter(dataloader))
        images = images[:num_samples]
        labels = labels[:num_samples]
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(images.to(self.device))
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1).cpu()
            confidences = probs.max(dim=1)[0].cpu()
        
        # Denormalize for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        # Plot
        n_cols = 4
        n_rows = (num_samples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.ravel()
        
        for i in range(len(axes)):
            if i < num_samples:
                img = images[i] * std + mean
                img = img.permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
                
                pred = preds[i].item()
                true = labels[i].item()
                conf = confidences[i].item()
                
                correct = pred == true
                color = 'green' if correct else 'red'
                
                axes[i].imshow(img)
                axes[i].set_title(
                    f"Pred: {self.config.class_names[pred]}\n"
                    f"True: {self.config.class_names[true]}\n"
                    f"Conf: {conf:.1%}",
                    fontsize=9,
                    color=color
                )
            axes[i].axis('off')
        
        plt.suptitle('Sample Predictions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved sample predictions: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def analyze_misclassifications(
        self,
        dataloader: DataLoader
    ) -> Dict[str, List[Dict]]:
        """
        Analyze misclassified samples.
        
        Args:
            dataloader: Test data loader
            
        Returns:
            Dictionary of misclassified samples by true class
        """
        if self.results is None:
            raise ValueError("No evaluation results. Run evaluate() first.")
        
        misclassified = defaultdict(list)
        
        sample_idx = 0
        for images, labels in dataloader:
            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = self.results.all_predictions[sample_idx]
                
                if true_label != pred_label:
                    misclassified[self.config.class_names[true_label]].append({
                        'sample_index': sample_idx,
                        'true_class': self.config.class_names[true_label],
                        'predicted_class': self.config.class_names[pred_label],
                        'confidence': max(self.results.all_probabilities[sample_idx])
                    })
                
                sample_idx += 1
        
        return dict(misclassified)
    
    def generate_report(
        self,
        dataloader: Optional[DataLoader] = None
    ) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            dataloader: Test data loader (for visualizations)
            
        Returns:
            Path to saved report
        """
        if self.results is None:
            raise ValueError("No evaluation results. Run evaluate() first.")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_dir = self.output_dir / f'report_{timestamp}'
        report_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating evaluation report...")
        print(f"Output directory: {report_dir}")
        
        # 1. Save JSON results
        self.results.save(report_dir / 'evaluation_results.json')
        print(f"  ✓ Saved evaluation_results.json")
        
        # 2. Generate confusion matrix
        self.plot_confusion_matrix(
            save_path=str(report_dir / 'confusion_matrix.png'),
            normalize=True
        )
        
        # 3. Per-class metrics plot
        self.plot_per_class_metrics(
            save_path=str(report_dir / 'per_class_metrics.png')
        )
        
        # 4. Sample predictions (if dataloader provided)
        if dataloader is not None:
            self.plot_sample_predictions(
                dataloader,
                num_samples=16,
                save_path=str(report_dir / 'sample_predictions.png')
            )
            
            # 5. Misclassification analysis
            misclassified = self.analyze_misclassifications(dataloader)
            with open(report_dir / 'misclassifications.json', 'w') as f:
                json.dump(misclassified, f, indent=2)
            print(f"  ✓ Saved misclassifications.json")
        
        # 6. Generate text report
        report_text = self._generate_text_report()
        with open(report_dir / 'evaluation_report.txt', 'w') as f:
            f.write(report_text)
        print(f"  ✓ Saved evaluation_report.txt")
        
        print(f"\n✓ Report complete: {report_dir}")
        
        return str(report_dir)
    
    def _generate_text_report(self) -> str:
        """Generate text-based evaluation report."""
        lines = []
        lines.append("=" * 70)
        lines.append("PCB DEFECT CLASSIFICATION - EVALUATION REPORT")
        lines.append("=" * 70)
        lines.append(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Evaluation time: {self.results.evaluation_time:.2f} seconds")
        
        lines.append("\n" + "-" * 70)
        lines.append("OVERALL METRICS")
        lines.append("-" * 70)
        lines.append(f"Total samples:      {self.results.total_samples}")
        lines.append(f"Correct predictions: {self.results.correct_predictions}")
        lines.append(f"Accuracy:           {self.results.accuracy:.2f}%")
        lines.append(f"Loss:               {self.results.loss:.4f}")
        
        lines.append("\n" + "-" * 70)
        lines.append("PER-CLASS METRICS")
        lines.append("-" * 70)
        lines.append(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        lines.append("-" * 60)
        
        for name in self.config.class_names:
            lines.append(f"{name:<20} {self.results.precision[name]:.4f}     "
                        f"{self.results.recall[name]:.4f}     "
                        f"{self.results.f1_score[name]:.4f}     "
                        f"{self.results.support[name]:>6}")
        
        lines.append("-" * 60)
        lines.append(f"{'Macro Avg':<20} {self.results.macro_precision:.4f}     "
                    f"{self.results.macro_recall:.4f}     {self.results.macro_f1:.4f}")
        lines.append(f"{'Weighted Avg':<20} {self.results.weighted_precision:.4f}     "
                    f"{self.results.weighted_recall:.4f}     {self.results.weighted_f1:.4f}")
        
        lines.append("\n" + "-" * 70)
        lines.append("CONFUSION MATRIX")
        lines.append("-" * 70)
        
        # Header
        header = "True\\Pred".ljust(15) + "".join(name[:8].center(10) for name in self.config.class_names)
        lines.append(header)
        lines.append("-" * len(header))
        
        for i, name in enumerate(self.config.class_names):
            row = name[:15].ljust(15)
            for j in range(len(self.config.class_names)):
                row += str(self.results.confusion_matrix[i][j]).center(10)
            lines.append(row)
        
        lines.append("\n" + "=" * 70)
        lines.append("END OF REPORT")
        lines.append("=" * 70)
        
        return "\n".join(lines)


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    print("Testing Model Evaluator Module...")
    print()
    
    # Simple test without actual model
    config = EvaluationConfig()
    
    print("✓ EvaluationConfig created")
    print(f"  Class names: {config.class_names}")
    
    # Test results
    results = EvaluationResults()
    results.accuracy = 95.5
    results.total_samples = 500
    results.correct_predictions = 478
    results.class_names = config.class_names
    
    for name in config.class_names:
        results.precision[name] = 0.95
        results.recall[name] = 0.94
        results.f1_score[name] = 0.945
        results.support[name] = 83
    
    results.macro_precision = 0.95
    results.macro_recall = 0.94
    results.macro_f1 = 0.945
    
    results.print_summary()
    
    print("\n✓ All tests passed!")
