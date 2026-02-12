"""
PCB Defect Classification - Dataset Module
===========================================

This module provides PyTorch Dataset and DataLoader classes for loading
and preprocessing PCB defect images for CNN training.

Features:
---------
- Custom Dataset class for PCB defects
- Configurable data augmentation
- Train/Val/Test split support
- Class balancing utilities
- Visualization tools
- ImageFolder compatibility

Data Organization Expected:
--------------------------
dataset/
├── train/
│   ├── Missing_hole/
│   ├── Mouse_bite/
│   ├── Open_circuit/
│   ├── Short/
│   ├── Spur/
│   └── Spurious_copper/
├── val/
│   └── ... (same structure)
└── test/
    └── ... (same structure)

Author: PCB Defect Detection Team
Version: 1.0.0
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Callable, Union
from dataclasses import dataclass, field
import json
import random
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class DataConfig:
    """Configuration for dataset loading and preprocessing."""
    
    # Data paths
    data_dir: str = 'outputs/full_dataset/rois/dataset'
    
    # Image settings
    image_size: Tuple[int, int] = (128, 128)
    
    # Normalization (ImageNet statistics for transfer learning)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Augmentation settings
    augment_train: bool = True
    rotation_degrees: int = 30
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.5
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.1
    random_affine_translate: Tuple[float, float] = (0.1, 0.1)
    
    # DataLoader settings
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    
    # Defect classes
    class_names: Tuple[str, ...] = (
        'Missing_hole',
        'Mouse_bite',
        'Open_circuit',
        'Short',
        'Spur',
        'Spurious_copper'
    )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'data_dir': self.data_dir,
            'image_size': list(self.image_size),
            'normalize_mean': list(self.normalize_mean),
            'normalize_std': list(self.normalize_std),
            'augment_train': self.augment_train,
            'rotation_degrees': self.rotation_degrees,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'class_names': list(self.class_names)
        }


# ============================================================
# DATA AUGMENTATION TRANSFORMS
# ============================================================

class TransformBuilder:
    """
    Builder class for creating data augmentation transforms.
    
    Provides different augmentation strategies for training, validation,
    and test data.
    """
    
    def __init__(self, config: Optional[DataConfig] = None):
        """
        Initialize transform builder.
        
        Args:
            config: Data configuration
        """
        self.config = config or DataConfig()
    
    def get_train_transform(self, augment: bool = True) -> transforms.Compose:
        """
        Get training transforms with optional augmentation.
        
        Args:
            augment: Whether to apply data augmentation
            
        Returns:
            Transform composition
        """
        transform_list = [
            transforms.Resize(self.config.image_size),
        ]
        
        if augment:
            transform_list.extend([
                # Geometric transformations
                transforms.RandomRotation(
                    degrees=self.config.rotation_degrees,
                    interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.RandomHorizontalFlip(p=self.config.horizontal_flip_prob),
                transforms.RandomVerticalFlip(p=self.config.vertical_flip_prob),
                transforms.RandomAffine(
                    degrees=0,
                    translate=self.config.random_affine_translate,
                    scale=(0.9, 1.1)
                ),
                
                # Color transformations
                transforms.ColorJitter(
                    brightness=self.config.color_jitter_brightness,
                    contrast=self.config.color_jitter_contrast,
                    saturation=self.config.color_jitter_saturation
                ),
            ])
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.normalize_mean,
                std=self.config.normalize_std
            )
        ])
        
        return transforms.Compose(transform_list)
    
    def get_eval_transform(self) -> transforms.Compose:
        """
        Get evaluation transforms (no augmentation).
        
        Returns:
            Transform composition
        """
        return transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.normalize_mean,
                std=self.config.normalize_std
            )
        ])
    
    def get_visualization_transform(self) -> transforms.Compose:
        """
        Get transforms for visualization (no normalization).
        
        Returns:
            Transform composition
        """
        return transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.ToTensor()
        ])


# ============================================================
# PCB DEFECT DATASET
# ============================================================

class PCBDefectDataset(Dataset):
    """
    PyTorch Dataset for PCB Defect Classification.
    
    Loads images from directory structure organized by class:
    split/class_name/*.png
    
    Example:
    --------
    >>> dataset = PCBDefectDataset('data/dataset', split='train')
    >>> image, label = dataset[0]
    >>> print(f"Image shape: {image.shape}, Label: {label}")
    """
    
    # Class names in standard order
    CLASS_NAMES = (
        'Missing_hole',
        'Mouse_bite',
        'Open_circuit',
        'Short',
        'Spur',
        'Spurious_copper'
    )
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        config: Optional[DataConfig] = None,
        augment: bool = True,
        verbose: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Root directory containing train/val/test splits
            split: One of 'train', 'val', or 'test'
            transform: Optional custom transform
            config: Data configuration
            augment: Apply augmentation (only for train)
            verbose: Print loading information
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.config = config or DataConfig()
        self.verbose = verbose
        
        # Validate split
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")
        
        # Set split directory
        self.split_dir = self.data_dir / split
        
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")
        
        # Create class mappings
        self.class_to_idx = {name: idx for idx, name in enumerate(self.CLASS_NAMES)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        
        # Set up transforms
        if transform is not None:
            self.transform = transform
        else:
            transform_builder = TransformBuilder(self.config)
            if split == 'train' and augment:
                self.transform = transform_builder.get_train_transform(augment=True)
            else:
                self.transform = transform_builder.get_eval_transform()
        
        # Load samples
        self.samples = self._load_samples()
        
        if verbose:
            self._print_info()
    
    def _load_samples(self) -> List[Dict]:
        """
        Load all image paths and labels.
        
        Returns:
            List of sample dictionaries with path, label, class_name
        """
        samples = []
        
        for class_name in self.CLASS_NAMES:
            class_dir = self.split_dir / class_name
            
            if not class_dir.exists():
                if self.verbose:
                    print(f"  ⚠ Class directory not found: {class_name}")
                continue
            
            # Find all images (support multiple extensions)
            extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
            class_images = []
            for ext in extensions:
                class_images.extend(class_dir.glob(ext))
            
            for img_path in class_images:
                samples.append({
                    'path': str(img_path),
                    'label': self.class_to_idx[class_name],
                    'class_name': class_name
                })
        
        return samples
    
    def _print_info(self) -> None:
        """Print dataset information."""
        print(f"\n{'='*60}")
        print(f"PCB Defect Dataset: {self.split.upper()}")
        print(f"{'='*60}")
        print(f"Root: {self.split_dir}")
        print(f"Total samples: {len(self.samples)}")
        print(f"\nClass distribution:")
        
        distribution = self.get_class_distribution()
        for class_name, count in distribution.items():
            pct = count / len(self.samples) * 100 if self.samples else 0
            bar = '█' * int(pct / 2)
            print(f"  {class_name:<20}: {count:>4} ({pct:>5.1f}%) {bar}")
        
        print(f"{'='*60}\n")
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image_tensor, label)
        """
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['path']).convert('RGB')
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        label = sample['label']
        
        return image, label
    
    def get_sample_info(self, idx: int) -> Dict:
        """
        Get full information about a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Sample dictionary
        """
        return self.samples[idx].copy()
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get distribution of classes.
        
        Returns:
            Dictionary mapping class names to counts
        """
        distribution = {name: 0 for name in self.CLASS_NAMES}
        for sample in self.samples:
            distribution[sample['class_name']] += 1
        return distribution
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling class imbalance.
        
        Returns weights inversely proportional to class frequency.
        
        Returns:
            Tensor of class weights
        """
        distribution = self.get_class_distribution()
        counts = torch.tensor([distribution[name] for name in self.CLASS_NAMES], dtype=torch.float)
        
        # Inverse frequency weighting
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(self.CLASS_NAMES)  # Normalize
        
        return weights
    
    def get_sample_weights(self) -> List[float]:
        """
        Get weight for each sample (for WeightedRandomSampler).
        
        Returns:
            List of sample weights
        """
        class_weights = self.get_class_weights()
        sample_weights = [class_weights[sample['label']].item() for sample in self.samples]
        return sample_weights


# ============================================================
# DATA LOADERS
# ============================================================

class DataLoaderFactory:
    """
    Factory class for creating PyTorch DataLoaders.
    
    Handles:
    - Creating train/val/test dataloaders
    - Class balancing via weighted sampling
    - Proper shuffle settings
    """
    
    def __init__(self, config: Optional[DataConfig] = None):
        """
        Initialize factory.
        
        Args:
            config: Data configuration
        """
        self.config = config or DataConfig()
    
    def create_dataloaders(
        self,
        data_dir: Optional[str] = None,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        use_balanced_sampling: bool = False,
        verbose: bool = True
    ) -> Dict[str, DataLoader]:
        """
        Create all dataloaders.
        
        Args:
            data_dir: Optional override for data directory
            batch_size: Optional override for batch size
            num_workers: Optional override for workers
            use_balanced_sampling: Use weighted sampling for class balance
            verbose: Print information
            
        Returns:
            Dictionary with 'train', 'val', 'test' dataloaders
        """
        data_dir = data_dir or self.config.data_dir
        batch_size = batch_size or self.config.batch_size
        num_workers = num_workers or self.config.num_workers
        
        dataloaders = {}
        
        for split in ['train', 'val', 'test']:
            dataset = PCBDefectDataset(
                data_dir=data_dir,
                split=split,
                config=self.config,
                augment=(split == 'train'),
                verbose=verbose
            )
            
            # Determine sampler for training
            sampler = None
            shuffle = (split == 'train')
            
            if split == 'train' and use_balanced_sampling:
                sample_weights = dataset.get_sample_weights()
                sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(dataset),
                    replacement=True
                )
                shuffle = False  # Can't use shuffle with sampler
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=self.config.pin_memory,
                sampler=sampler,
                drop_last=(split == 'train')  # Drop incomplete batch for training
            )
            
            dataloaders[split] = dataloader
        
        if verbose:
            print("\n" + "="*60)
            print("DATALOADER SUMMARY")
            print("="*60)
            for split, loader in dataloaders.items():
                print(f"{split.upper()}: {len(loader.dataset)} samples, "
                      f"{len(loader)} batches (batch_size={batch_size})")
            print("="*60 + "\n")
        
        return dataloaders


# ============================================================
# VISUALIZATION UTILITIES
# ============================================================

def visualize_batch(
    dataloader: DataLoader,
    num_samples: int = 16,
    figsize: Tuple[int, int] = (12, 12),
    save_path: Optional[str] = None
) -> None:
    """
    Visualize a batch of samples from dataloader.
    
    Args:
        dataloader: PyTorch DataLoader
        num_samples: Number of samples to show
        figsize: Figure size
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    
    # Get a batch
    images, labels = next(iter(dataloader))
    
    # Limit samples
    num_samples = min(num_samples, len(images))
    
    # Calculate grid size
    n_cols = 4
    n_rows = (num_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.ravel()
    
    # Mean and std for denormalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    class_names = PCBDefectDataset.CLASS_NAMES
    
    for i in range(len(axes)):
        if i < num_samples:
            # Denormalize image
            img = images[i] * std + mean
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            
            axes[i].imshow(img)
            axes[i].set_title(class_names[labels[i]], fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle('Sample Batch from DataLoader', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to: {save_path}")
    
    plt.show()


def visualize_augmentations(
    image_path: str,
    num_augmented: int = 9,
    config: Optional[DataConfig] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize data augmentation on a single image.
    
    Args:
        image_path: Path to image
        num_augmented: Number of augmented versions to show
        config: Data config
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    
    config = config or DataConfig()
    transform_builder = TransformBuilder(config)
    
    # Load original image
    original = Image.open(image_path).convert('RGB')
    
    # Get transforms
    aug_transform = transform_builder.get_train_transform(augment=True)
    no_aug_transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor()
    ])
    
    # Create figure
    n_cols = 5
    n_rows = (num_augmented + 1) // n_cols + 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    axes = axes.ravel()
    
    # Mean and std for denormalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Show original
    axes[0].imshow(no_aug_transform(original).permute(1, 2, 0))
    axes[0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Show augmented versions
    for i in range(1, num_augmented + 1):
        aug_img = aug_transform(original)
        aug_img = aug_img * std + mean  # Denormalize
        aug_img = aug_img.permute(1, 2, 0).numpy()
        aug_img = np.clip(aug_img, 0, 1)
        
        if i < len(axes):
            axes[i].imshow(aug_img)
            axes[i].set_title(f'Augmented {i}', fontsize=10)
            axes[i].axis('off')
    
    # Hide unused axes
    for i in range(num_augmented + 1, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Data Augmentation Examples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to: {save_path}")
    
    plt.show()


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    print("Testing PCB Defect Dataset Module...")
    print()
    
    # Test with sample data directory
    data_dir = "outputs/full_dataset/rois/dataset"
    
    # Check if data exists
    if not Path(data_dir).exists():
        print(f"Data directory not found: {data_dir}")
        print("Please run the dataset creation script first.")
        exit(1)
    
    # Create config
    config = DataConfig(data_dir=data_dir)
    
    # Create dataset
    print("Creating training dataset...")
    train_dataset = PCBDefectDataset(data_dir, split='train', config=config)
    
    print("\nCreating validation dataset...")
    val_dataset = PCBDefectDataset(data_dir, split='val', config=config)
    
    # Test loading a sample
    if len(train_dataset) > 0:
        image, label = train_dataset[0]
        print(f"\nSample loaded:")
        print(f"  Image shape: {image.shape}")
        print(f"  Label: {label} ({train_dataset.idx_to_class[label]})")
    
    # Test class weights
    print("\nClass weights for balanced training:")
    weights = train_dataset.get_class_weights()
    for i, (name, weight) in enumerate(zip(train_dataset.CLASS_NAMES, weights)):
        print(f"  {name}: {weight:.4f}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    factory = DataLoaderFactory(config)
    dataloaders = factory.create_dataloaders(data_dir, batch_size=16)
    
    # Test iteration
    print("\nTesting batch loading...")
    images, labels = next(iter(dataloaders['train']))
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels}")
    
    print("\n✓ All tests passed!")
