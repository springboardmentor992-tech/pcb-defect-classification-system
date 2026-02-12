"""
PCB Defect Classification - CNN Model Architecture
===================================================

This module provides the core CNN model architecture for PCB defect classification.
It uses EfficientNet as the backbone with custom classification head designed
specifically for the 6-class PCB defect detection task.

Features:
---------
- EfficientNet backbone (B0-B7 variants supported)
- Transfer learning with ImageNet pre-trained weights
- Custom classification head with dropout regularization
- Flexible architecture for different compute budgets
- Built-in feature extraction mode

Model Variants (EfficientNet):
-----------------------------
| Model | Params | Top-1 Acc | Speed  | Use Case                |
|-------|--------|-----------|--------|-------------------------|
| B0    | 5.3M   | 77.1%     | Fast   | Quick experiments       |
| B1    | 7.8M   | 79.1%     | Fast   | Good balance            |
| B2    | 9.2M   | 80.1%     | Medium | Better accuracy         |
| B3    | 12M    | 81.6%     | Medium | PCB dataset (default)   |
| B4    | 19M    | 82.9%     | Slow   | High accuracy needed    |

Author: PCB Defect Detection Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, Tuple, Dict, List, Union
from dataclasses import dataclass
from pathlib import Path
import json


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class ModelConfig:
    """Configuration for CNN model."""
    
    # Architecture
    model_name: str = 'efficientnet_b3'
    num_classes: int = 6
    pretrained: bool = True
    
    # Regularization
    dropout_rate: float = 0.3
    drop_path_rate: float = 0.2
    
    # Input
    input_size: Tuple[int, int] = (128, 128)
    in_channels: int = 3
    
    # Classification head
    hidden_dim: Optional[int] = 512
    use_bn_in_head: bool = True
    
    # Class names for PCB defects
    class_names: Tuple[str, ...] = (
        'Missing_hole',
        'Mouse_bite',
        'Open_circuit',
        'Short',
        'Spur',
        'Spurious_copper'
    )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'pretrained': self.pretrained,
            'dropout_rate': self.dropout_rate,
            'drop_path_rate': self.drop_path_rate,
            'input_size': list(self.input_size),
            'in_channels': self.in_channels,
            'hidden_dim': self.hidden_dim,
            'use_bn_in_head': self.use_bn_in_head,
            'class_names': list(self.class_names)
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ModelConfig':
        """Create config from dictionary."""
        config_dict['input_size'] = tuple(config_dict['input_size'])
        config_dict['class_names'] = tuple(config_dict['class_names'])
        return cls(**config_dict)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ModelConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


# ============================================================
# CLASSIFICATION HEAD
# ============================================================

class ClassificationHead(nn.Module):
    """
    Custom classification head for PCB defect detection.
    
    Architecture:
        Input Features -> [Dropout -> Linear -> BN -> ReLU] -> Dropout -> Linear -> Output
        
    The optional hidden layer adds capacity for learning complex patterns
    specific to PCB defects.
    """
    
    def __init__(
        self, 
        in_features: int,
        num_classes: int,
        hidden_dim: Optional[int] = 512,
        dropout_rate: float = 0.3,
        use_bn: bool = True
    ):
        """
        Initialize classification head.
        
        Args:
            in_features: Number of input features from backbone
            num_classes: Number of output classes
            hidden_dim: Hidden layer dimension (None to skip)
            dropout_rate: Dropout probability
            use_bn: Use batch normalization
        """
        super().__init__()
        
        self.in_features = in_features
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        if hidden_dim is not None:
            # Two-layer head with intermediate representation
            layers = [
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, hidden_dim),
            ]
            
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.extend([
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate / 2),  # Less dropout in second layer
                nn.Linear(hidden_dim, num_classes)
            ])
            
            self.head = nn.Sequential(*layers)
        else:
            # Simple linear head
            self.head = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.head(x)


# ============================================================
# MAIN MODEL: EFFICIENTNET CLASSIFIER
# ============================================================

class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-based classifier for PCB defect detection.
    
    This model combines:
    1. EfficientNet backbone (pre-trained on ImageNet)
    2. Custom classification head for PCB defects
    3. Optional feature extraction mode
    
    The model is designed to achieve ≥95% accuracy on the PCB defect
    classification task with 6 classes.
    
    Example:
    --------
    >>> config = ModelConfig(model_name='efficientnet_b3', num_classes=6)
    >>> model = EfficientNetClassifier(config)
    >>> x = torch.randn(4, 3, 128, 128)
    >>> output = model(x)
    >>> print(output.shape)  # torch.Size([4, 6])
    """
    
    # Mapping of model names to their feature dimensions
    FEATURE_DIMS = {
        'efficientnet_b0': 1280,
        'efficientnet_b1': 1280,
        'efficientnet_b2': 1408,
        'efficientnet_b3': 1536,
        'efficientnet_b4': 1792,
        'efficientnet_b5': 2048,
        'efficientnet_b6': 2304,
        'efficientnet_b7': 2560,
        'efficientnetv2_s': 1280,
        'efficientnetv2_m': 1280,
        'efficientnetv2_l': 1280,
    }
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the EfficientNet classifier.
        
        Args:
            config: Model configuration. Uses default if None.
        """
        super().__init__()
        
        self.config = config or ModelConfig()
        
        # Create backbone
        self.backbone = self._create_backbone()
        
        # Get feature dimension
        self.feature_dim = self._get_feature_dim()
        
        # Create classification head
        self.classifier = ClassificationHead(
            in_features=self.feature_dim,
            num_classes=self.config.num_classes,
            hidden_dim=self.config.hidden_dim,
            dropout_rate=self.config.dropout_rate,
            use_bn=self.config.use_bn_in_head
        )
        
        # Feature extraction mode flag
        self._feature_mode = False
    
    def _create_backbone(self) -> nn.Module:
        """Create EfficientNet backbone using timm."""
        # Create model with global average pooling, no classifier
        backbone = timm.create_model(
            self.config.model_name,
            pretrained=self.config.pretrained,
            num_classes=0,  # Remove classifier
            global_pool='avg',  # Global average pooling
            drop_rate=self.config.dropout_rate,
            drop_path_rate=self.config.drop_path_rate
        )
        
        return backbone
    
    def _get_feature_dim(self) -> int:
        """Get the feature dimension from backbone."""
        # Try predefined dimensions first
        if self.config.model_name in self.FEATURE_DIMS:
            return self.FEATURE_DIMS[self.config.model_name]
        
        # Otherwise, infer by forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, self.config.in_channels, 
                              *self.config.input_size)
            features = self.backbone(dummy)
            return features.shape[1]
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Feature tensor of shape (B, feature_dim)
        """
        return self.backbone(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            - If feature_mode: Feature tensor of shape (B, feature_dim)
            - Otherwise: Logits of shape (B, num_classes)
        """
        # Extract features
        features = self.backbone(x)
        
        # Return features if in feature extraction mode
        if self._feature_mode:
            return features
        
        # Classification
        logits = self.classifier(features)
        
        return logits
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions with probabilities.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Tuple of (predicted_classes, probabilities)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        
        return preds, probs
    
    def predict_with_confidence(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get predictions with class names and confidence scores.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Dictionary with predictions, confidences, and class names
        """
        preds, probs = self.predict(x)
        confidences = probs.max(dim=1)[0]
        
        # Get class names
        class_names = [self.config.class_names[p.item()] for p in preds]
        
        return {
            'predictions': preds,
            'confidences': confidences,
            'probabilities': probs,
            'class_names': class_names
        }
    
    def set_feature_mode(self, mode: bool = True) -> None:
        """Enable/disable feature extraction mode."""
        self._feature_mode = mode
    
    def freeze_backbone(self, freeze: bool = True) -> None:
        """
        Freeze/unfreeze backbone parameters.
        
        Useful for transfer learning - first train classifier head,
        then fine-tune entire network.
        
        Args:
            freeze: If True, freeze backbone weights
        """
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
    
    def unfreeze_last_n_blocks(self, n: int = 2) -> None:
        """
        Unfreeze last n blocks of backbone for gradual fine-tuning.
        
        Args:
            n: Number of blocks to unfreeze from the end
        """
        # First freeze all
        self.freeze_backbone(True)
        
        # Get all named modules
        all_modules = list(self.backbone.named_modules())
        
        # Find block indices (EfficientNet has 'blocks' attribute)
        if hasattr(self.backbone, 'blocks'):
            blocks = self.backbone.blocks
            num_blocks = len(blocks)
            
            # Unfreeze last n blocks
            for i in range(max(0, num_blocks - n), num_blocks):
                for param in blocks[i].parameters():
                    param.requires_grad = True
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Get parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        
        return {
            'total': total,
            'trainable': trainable,
            'frozen': frozen
        }
    
    def save_model(self, path: Union[str, Path], save_config: bool = True) -> None:
        """
        Save model weights and optionally config.
        
        Args:
            path: Path to save model (.pth file)
            save_config: Also save config JSON
        """
        path = Path(path)
        
        # Save weights
        torch.save(self.state_dict(), path)
        
        # Save config
        if save_config:
            config_path = path.with_suffix('.json')
            self.config.save(config_path)
        
        print(f"✓ Model saved to: {path}")
    
    @classmethod
    def load_model(
        cls, 
        path: Union[str, Path], 
        config: Optional[ModelConfig] = None,
        device: str = 'cpu'
    ) -> 'EfficientNetClassifier':
        """
        Load model from checkpoint.
        
        Args:
            path: Path to model weights (.pth file)
            config: Model config. If None, tries to load from JSON.
            device: Device to load model to
            
        Returns:
            Loaded model
        """
        path = Path(path)
        
        # Try to load config
        if config is None:
            config_path = path.with_suffix('.json')
            if config_path.exists():
                config = ModelConfig.load(config_path)
            else:
                config = ModelConfig()
        
        # Create model
        model = cls(config)
        
        # Load weights
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        
        return model
    
    def summary(self) -> str:
        """Get model summary string."""
        params = self.get_num_parameters()
        
        lines = [
            "=" * 60,
            "EfficientNet PCB Defect Classifier",
            "=" * 60,
            f"Backbone:        {self.config.model_name}",
            f"Input Size:      {self.config.input_size}",
            f"Num Classes:     {self.config.num_classes}",
            f"Feature Dim:     {self.feature_dim}",
            f"Hidden Dim:      {self.config.hidden_dim}",
            f"Dropout Rate:    {self.config.dropout_rate}",
            f"Pretrained:      {self.config.pretrained}",
            "-" * 60,
            f"Total Params:    {params['total']:,}",
            f"Trainable:       {params['trainable']:,}",
            f"Frozen:          {params['frozen']:,}",
            "-" * 60,
            "Classes:",
        ]
        
        for i, name in enumerate(self.config.class_names):
            lines.append(f"  {i}: {name}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


# ============================================================
# MODEL FACTORY
# ============================================================

def create_model(
    model_name: str = 'efficientnet_b3',
    num_classes: int = 6,
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    **kwargs
) -> EfficientNetClassifier:
    """
    Factory function to create a PCB defect classifier.
    
    Args:
        model_name: Backbone model name
        num_classes: Number of classes
        pretrained: Use pretrained weights
        dropout_rate: Dropout rate
        **kwargs: Additional config parameters
        
    Returns:
        Configured EfficientNetClassifier
    """
    config = ModelConfig(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        **kwargs
    )
    
    return EfficientNetClassifier(config)


# ============================================================
# ALTERNATIVE MODELS (for comparison)
# ============================================================

class SimpleCNN(nn.Module):
    """
    Simple CNN for baseline comparison.
    
    A lightweight model for quick experiments and baseline performance.
    """
    
    def __init__(self, num_classes: int = 6, dropout: float = 0.3):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: 128x128 -> 64x64
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4: 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 5: 8x8 -> 4x4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResNetClassifier(nn.Module):
    """
    ResNet-based classifier for comparison.
    
    Uses ResNet-50 backbone as an alternative to EfficientNet.
    """
    
    def __init__(self, num_classes: int = 6, pretrained: bool = True, 
                 dropout: float = 0.3):
        super().__init__()
        
        # Use timm for ResNet
        self.backbone = timm.create_model(
            'resnet50',
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    print("Testing PCB Defect CNN Model...")
    print()
    
    # Create model
    config = ModelConfig(model_name='efficientnet_b3', pretrained=False)
    model = EfficientNetClassifier(config)
    
    # Print summary
    print(model.summary())
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 128, 128)
    
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test prediction
    preds, probs = model.predict(x)
    print(f"\nPredictions: {preds}")
    print(f"Probabilities shape: {probs.shape}")
    
    # Test feature extraction
    model.set_feature_mode(True)
    features = model(x)
    print(f"\nFeature shape: {features.shape}")
    
    print("\n✓ All tests passed!")
