"""
PCB Dataset Loader
==================

A comprehensive loader for the PCB Defect Detection dataset.
Handles loading, inspecting, visualizing, and preparing the dataset
for image processing and machine learning tasks.

Features:
---------
- Load images from 6 defect categories
- Parse Pascal VOC XML annotations
- Create template-test image pairs
- Generate dataset statistics
- Visualize samples with annotations

Author: PCB Defect Detection Team
Version: 1.0.0

Usage:
------
>>> from pcb_dataset_loader import PCBDatasetLoader
>>> loader = PCBDatasetLoader('/path/to/PCB_DATASET')
>>> loader.print_statistics()
>>> pairs = loader.create_template_test_pairs('output_dir', num_pairs_per_class=5)
"""

import os
import sys
import cv2
import json
import random
import shutil
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

# Add parent to path for relative imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from utils.image_utils import (
        load_image, 
        save_image, 
        draw_bounding_boxes,
        visualize_comparison
    )
except ImportError:
    # Fallback for standalone usage
    pass


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class BoundingBox:
    """Represents a bounding box annotation."""
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    label: str
    
    @property
    def width(self) -> int:
        return self.xmax - self.xmin
    
    @property
    def height(self) -> int:
        return self.ymax - self.ymin
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.xmin + self.width // 2, self.ymin + self.height // 2)
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.xmin, self.ymin, self.xmax, self.ymax)
    
    def to_dict(self) -> dict:
        return {
            'xmin': self.xmin,
            'ymin': self.ymin,
            'xmax': self.xmax,
            'ymax': self.ymax,
            'label': self.label,
            'width': self.width,
            'height': self.height,
            'area': self.area
        }


@dataclass
class Annotation:
    """Represents complete annotation for an image."""
    filename: str
    width: int
    height: int
    depth: int
    boxes: List[BoundingBox] = field(default_factory=list)
    
    @property
    def num_objects(self) -> int:
        return len(self.boxes)
    
    def to_dict(self) -> dict:
        return {
            'filename': self.filename,
            'width': self.width,
            'height': self.height,
            'depth': self.depth,
            'num_objects': self.num_objects,
            'boxes': [box.to_dict() for box in self.boxes]
        }


@dataclass
class ImageSample:
    """Represents a complete image sample with metadata."""
    image_path: Path
    annotation_path: Optional[Path]
    defect_class: str
    pcb_id: str  # The reference PCB (01, 04, 05, etc.)
    sample_idx: int  # The sample index within the class
    annotation: Optional[Annotation] = None
    
    def load_image(self, color_mode: str = 'color') -> np.ndarray:
        """Load the image."""
        return cv2.imread(str(self.image_path), 
                          cv2.IMREAD_GRAYSCALE if color_mode == 'gray' else cv2.IMREAD_COLOR)
    
    def to_dict(self) -> dict:
        return {
            'image_path': str(self.image_path),
            'annotation_path': str(self.annotation_path) if self.annotation_path else None,
            'defect_class': self.defect_class,
            'pcb_id': self.pcb_id,
            'sample_idx': self.sample_idx,
            'annotation': self.annotation.to_dict() if self.annotation else None
        }


# ============================================================
# MAIN DATASET LOADER CLASS
# ============================================================

class PCBDatasetLoader:
    """
    Comprehensive loader for the PCB Defect Detection dataset.
    
    This class provides functionality to:
    - Load and organize the PCB dataset
    - Parse XML annotations
    - Create template-test pairs for subtraction
    - Generate statistics and visualizations
    
    Attributes
    ----------
    dataset_path : Path
        Root path to the PCB_DATASET directory
    defect_classes : list
        List of defect class names
    samples : dict
        Dictionary mapping defect classes to lists of ImageSamples
    templates : dict
        Dictionary mapping PCB IDs to template image paths
        
    Examples
    --------
    >>> loader = PCBDatasetLoader('/path/to/PCB_DATASET')
    >>> loader.print_statistics()
    >>> sample = loader.get_sample('Missing_hole', 0)
    >>> pairs = loader.create_template_test_pairs('data/raw', num_pairs_per_class=5)
    """
    
    # Standard defect classes in the PCB dataset
    DEFECT_CLASSES = [
        'Missing_hole',
        'Mouse_bite',
        'Open_circuit',
        'Short',
        'Spur',
        'Spurious_copper'
    ]
    
    # PCB reference IDs (from PCB_USED folder)
    PCB_IDS = ['01', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    
    def __init__(self, dataset_path: Union[str, Path], verbose: bool = True):
        """
        Initialize the PCB Dataset Loader.
        
        Parameters
        ----------
        dataset_path : str or Path
            Path to the PCB_DATASET root directory
        verbose : bool, optional
            Print loading progress, default True
        """
        self.dataset_path = Path(dataset_path)
        self.verbose = verbose
        
        # Validate dataset path
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at: {self.dataset_path}")
        
        # Define subdirectories
        self.images_dir = self.dataset_path / 'images'
        self.annotations_dir = self.dataset_path / 'Annotations'
        self.templates_dir = self.dataset_path / 'PCB_USED'
        
        # Validate structure
        self._validate_dataset_structure()
        
        # Storage for loaded data
        self.samples: Dict[str, List[ImageSample]] = {}
        self.templates: Dict[str, Path] = {}
        
        # Load the dataset
        if self.verbose:
            print("\n" + "="*66)
            print("    PCB DATASET LOADER")
            print("="*66)
            print(f"📂 Dataset path: {self.dataset_path}")
        
        self._load_templates()
        self._load_samples()
        
        if self.verbose:
            print("="*66)
            print("    ✓ DATASET LOADED SUCCESSFULLY")
            print("="*66 + "\n")
    
    def _validate_dataset_structure(self) -> None:
        """Validate the expected dataset structure exists."""
        required_dirs = [self.images_dir, self.annotations_dir, self.templates_dir]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                raise FileNotFoundError(
                    f"Required directory not found: {dir_path}\n"
                    f"Expected structure:\n"
                    f"  PCB_DATASET/\n"
                    f"  ├── images/\n"
                    f"  ├── Annotations/\n"
                    f"  └── PCB_USED/"
                )
    
    def _load_templates(self) -> None:
        """Load reference PCB template images."""
        if self.verbose:
            print(f"\n📷 Loading template images from: PCB_USED/")
        
        template_files = list(self.templates_dir.glob('*.JPG')) + \
                        list(self.templates_dir.glob('*.jpg'))
        
        for template_path in sorted(template_files):
            # Extract PCB ID from filename (e.g., "01.JPG" -> "01")
            pcb_id = template_path.stem
            self.templates[pcb_id] = template_path
        
        if self.verbose:
            print(f"   ✓ Found {len(self.templates)} template images: {list(self.templates.keys())}")
    
    def _load_samples(self) -> None:
        """Load all defect samples organized by class."""
        if self.verbose:
            print(f"\n🔍 Loading defect samples...")
        
        for defect_class in self.DEFECT_CLASSES:
            self.samples[defect_class] = []
            
            # Get image directory for this class
            class_image_dir = self.images_dir / defect_class
            class_annotation_dir = self.annotations_dir / defect_class
            
            if not class_image_dir.exists():
                if self.verbose:
                    print(f"   ⚠ Directory not found: {class_image_dir}")
                continue
            
            # Get all images
            image_files = sorted(
                list(class_image_dir.glob('*.jpg')) + 
                list(class_image_dir.glob('*.JPG'))
            )
            
            for image_path in image_files:
                # Parse filename to extract PCB ID and sample index
                # Format: XX_defect_type_NN.jpg (e.g., 01_missing_hole_05.jpg)
                parts = image_path.stem.split('_')
                pcb_id = parts[0]
                sample_idx = int(parts[-1]) if parts[-1].isdigit() else 0
                
                # Find corresponding annotation
                annotation_path = class_annotation_dir / (image_path.stem + '.xml')
                if not annotation_path.exists():
                    annotation_path = None
                
                # Create sample
                sample = ImageSample(
                    image_path=image_path,
                    annotation_path=annotation_path,
                    defect_class=defect_class,
                    pcb_id=pcb_id,
                    sample_idx=sample_idx
                )
                
                self.samples[defect_class].append(sample)
            
            if self.verbose:
                print(f"   ✓ {defect_class}: {len(self.samples[defect_class])} samples")
    
    def _parse_annotation(self, annotation_path: Path) -> Annotation:
        """
        Parse a Pascal VOC XML annotation file.
        
        Parameters
        ----------
        annotation_path : Path
            Path to the XML annotation file
            
        Returns
        -------
        Annotation
            Parsed annotation object
        """
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        # Extract image info
        filename = root.find('filename').text
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        depth = int(size.find('depth').text)
        
        # Extract bounding boxes
        boxes = []
        for obj in root.findall('object'):
            label = obj.find('name').text
            bndbox = obj.find('bndbox')
            
            box = BoundingBox(
                xmin=int(bndbox.find('xmin').text),
                ymin=int(bndbox.find('ymin').text),
                xmax=int(bndbox.find('xmax').text),
                ymax=int(bndbox.find('ymax').text),
                label=label
            )
            boxes.append(box)
        
        return Annotation(
            filename=filename,
            width=width,
            height=height,
            depth=depth,
            boxes=boxes
        )
    
    # ========================================================
    # PUBLIC API - DATA ACCESS
    # ========================================================
    
    def get_sample(
        self, 
        defect_class: str, 
        sample_idx: int,
        load_annotation: bool = True
    ) -> ImageSample:
        """
        Get a specific sample by class and index.
        
        Parameters
        ----------
        defect_class : str
            Defect class name
        sample_idx : int
            Sample index within the class
        load_annotation : bool, optional
            Whether to load the annotation, default True
            
        Returns
        -------
        ImageSample
            The requested sample
            
        Examples
        --------
        >>> sample = loader.get_sample('Missing_hole', 0)
        >>> image = sample.load_image()
        """
        if defect_class not in self.samples:
            raise ValueError(f"Unknown defect class: {defect_class}")
        
        if sample_idx >= len(self.samples[defect_class]):
            raise IndexError(
                f"Sample index {sample_idx} out of range for {defect_class} "
                f"(0-{len(self.samples[defect_class])-1})"
            )
        
        sample = self.samples[defect_class][sample_idx]
        
        if load_annotation and sample.annotation_path and sample.annotation is None:
            sample.annotation = self._parse_annotation(sample.annotation_path)
        
        return sample
    
    def get_template(self, pcb_id: str) -> Path:
        """
        Get the template image path for a PCB ID.
        
        Parameters
        ----------
        pcb_id : str
            PCB reference ID (e.g., '01', '04')
            
        Returns
        -------
        Path
            Path to the template image
        """
        if pcb_id not in self.templates:
            raise ValueError(f"Unknown PCB ID: {pcb_id}. Available: {list(self.templates.keys())}")
        
        return self.templates[pcb_id]
    
    def get_all_samples(self, defect_class: Optional[str] = None) -> List[ImageSample]:
        """
        Get all samples, optionally filtered by defect class.
        
        Parameters
        ----------
        defect_class : str, optional
            If provided, only return samples from this class
            
        Returns
        -------
        list
            List of ImageSample objects
        """
        if defect_class:
            return self.samples.get(defect_class, [])
        
        # Return all samples
        all_samples = []
        for samples in self.samples.values():
            all_samples.extend(samples)
        return all_samples
    
    def get_random_sample(
        self, 
        defect_class: Optional[str] = None,
        load_annotation: bool = True
    ) -> ImageSample:
        """
        Get a random sample.
        
        Parameters
        ----------
        defect_class : str, optional
            If provided, pick from this class only
        load_annotation : bool, optional
            Whether to load annotation
            
        Returns
        -------
        ImageSample
            A random sample
        """
        if defect_class:
            samples = self.samples[defect_class]
        else:
            samples = self.get_all_samples()
        
        sample = random.choice(samples)
        
        if load_annotation and sample.annotation_path and sample.annotation is None:
            sample.annotation = self._parse_annotation(sample.annotation_path)
        
        return sample
    
    # ========================================================
    # PUBLIC API - STATISTICS
    # ========================================================
    
    def get_statistics(self) -> dict:
        """
        Get comprehensive dataset statistics.
        
        Returns
        -------
        dict
            Dictionary containing dataset statistics
        """
        stats = {
            'total_images': 0,
            'total_annotations': 0,
            'total_templates': len(self.templates),
            'defect_classes': {},
            'pcb_distribution': {pcb_id: 0 for pcb_id in self.PCB_IDS},
            'bounding_box_stats': {
                'total': 0,
                'min_area': float('inf'),
                'max_area': 0,
                'avg_area': 0
            }
        }
        
        all_box_areas = []
        
        for defect_class, samples in self.samples.items():
            class_stats = {
                'image_count': len(samples),
                'annotation_count': sum(1 for s in samples if s.annotation_path),
                'pcb_ids': set()
            }
            
            for sample in samples:
                class_stats['pcb_ids'].add(sample.pcb_id)
                stats['pcb_distribution'][sample.pcb_id] = \
                    stats['pcb_distribution'].get(sample.pcb_id, 0) + 1
                
                # Load annotation for box stats (sample only)
                if sample.annotation_path and len(all_box_areas) < 200:
                    try:
                        ann = self._parse_annotation(sample.annotation_path)
                        for box in ann.boxes:
                            all_box_areas.append(box.area)
                            stats['bounding_box_stats']['total'] += 1
                    except:
                        pass
            
            class_stats['pcb_ids'] = list(class_stats['pcb_ids'])
            stats['defect_classes'][defect_class] = class_stats
            stats['total_images'] += class_stats['image_count']
            stats['total_annotations'] += class_stats['annotation_count']
        
        # Calculate box stats
        if all_box_areas:
            stats['bounding_box_stats']['min_area'] = min(all_box_areas)
            stats['bounding_box_stats']['max_area'] = max(all_box_areas)
            stats['bounding_box_stats']['avg_area'] = sum(all_box_areas) / len(all_box_areas)
        
        return stats
    
    def print_statistics(self) -> None:
        """Print formatted dataset statistics to console."""
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("                     PCB DATASET STATISTICS")
        print("="*70)
        
        # Header
        print(f"\n{'Defect Type':<25} {'Images':>10} {'Annotations':>15}")
        print("-"*70)
        
        # Per-class stats
        for defect_class in self.DEFECT_CLASSES:
            class_stats = stats['defect_classes'].get(defect_class, {})
            images = class_stats.get('image_count', 0)
            annotations = class_stats.get('annotation_count', 0)
            print(f"{defect_class:<25} {images:>10} {annotations:>15}")
        
        # Totals
        print("-"*70)
        print(f"{'TOTAL':<25} {stats['total_images']:>10} {stats['total_annotations']:>15}")
        
        # Template info
        print(f"\n📎 Reference PCB Templates: {stats['total_templates']}")
        print(f"   PCB IDs: {', '.join(self.templates.keys())}")
        
        # PCB distribution
        print(f"\n📊 Samples per PCB template:")
        for pcb_id, count in sorted(stats['pcb_distribution'].items()):
            if count > 0:
                print(f"   PCB {pcb_id}: {count} samples")
        
        print("="*70 + "\n")
    
    # ========================================================
    # PUBLIC API - VISUALIZATION
    # ========================================================
    
    def visualize_sample(
        self,
        defect_class: str,
        sample_idx: int,
        show_annotations: bool = True,
        save_path: Optional[Union[str, Path]] = None
    ) -> np.ndarray:
        """
        Visualize a sample with optional annotations.
        
        Parameters
        ----------
        defect_class : str
            Defect class name
        sample_idx : int
            Sample index
        show_annotations : bool, optional
            Whether to overlay bounding boxes
        save_path : str or Path, optional
            Path to save the visualization
            
        Returns
        -------
        np.ndarray
            Visualization image
        """
        sample = self.get_sample(defect_class, sample_idx)
        image = sample.load_image()
        
        if show_annotations and sample.annotation:
            boxes = [box.to_tuple() for box in sample.annotation.boxes]
            labels = [box.label for box in sample.annotation.boxes]
            
            # Define color based on defect type
            colors = {
                'missing_hole': (0, 0, 255),      # Red
                'mouse_bite': (255, 0, 0),        # Blue
                'open_circuit': (0, 165, 255),    # Orange
                'short': (0, 255, 255),           # Yellow
                'spur': (255, 0, 255),            # Magenta
                'spurious_copper': (0, 255, 0)    # Green
            }
            
            color = colors.get(sample.annotation.boxes[0].label, (0, 255, 0))
            
            for box in boxes:
                xmin, ymin, xmax, ymax = box
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 3)
                
                # Add label
                label_text = defect_class.replace('_', ' ')
                cv2.putText(
                    image, label_text, (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
                )
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), image)
            print(f"✓ Saved visualization to: {save_path}")
        
        return image
    
    def visualize_sample_with_annotation(
        self,
        defect_class: str,
        sample_idx: int,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (15, 10)
    ) -> None:
        """
        Create a comprehensive visualization with annotations sidebar.
        
        Parameters
        ----------
        defect_class : str
            Defect class name
        sample_idx : int
            Sample index
        save_path : str or Path, optional
            Path to save the figure
        figsize : tuple, optional
            Figure size
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        sample = self.get_sample(defect_class, sample_idx)
        image = sample.load_image()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize, 
                                  gridspec_kw={'width_ratios': [3, 1]})
        
        # Main image with bounding boxes
        axes[0].imshow(image_rgb)
        
        # Colors for boxes
        colors = ['red', 'blue', 'orange', 'yellow', 'magenta', 'green']
        
        if sample.annotation:
            for i, box in enumerate(sample.annotation.boxes):
                color = colors[i % len(colors)]
                rect = Rectangle(
                    (box.xmin, box.ymin), box.width, box.height,
                    linewidth=3, edgecolor=color, facecolor='none'
                )
                axes[0].add_patch(rect)
                axes[0].text(
                    box.xmin, box.ymin - 5,
                    f"Defect {i+1}",
                    color=color, fontsize=10, fontweight='bold'
                )
        
        axes[0].set_title(f"{defect_class.replace('_', ' ')} - Sample {sample_idx}", 
                          fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Info sidebar
        axes[1].axis('off')
        
        info_text = [
            "SAMPLE INFORMATION",
            "─" * 25,
            f"Defect Class: {defect_class}",
            f"PCB ID: {sample.pcb_id}",
            f"Sample Index: {sample.sample_idx}",
            "",
        ]
        
        if sample.annotation:
            info_text.extend([
                "ANNOTATION DETAILS",
                "─" * 25,
                f"Image Size: {sample.annotation.width}x{sample.annotation.height}",
                f"Objects: {sample.annotation.num_objects}",
                ""
            ])
            
            for i, box in enumerate(sample.annotation.boxes):
                info_text.extend([
                    f"Defect {i+1}:",
                    f"  Position: ({box.xmin}, {box.ymin})",
                    f"  Size: {box.width}x{box.height}",
                    f"  Area: {box.area} px²",
                    ""
                ])
        
        # Display info
        axes[1].text(
            0.05, 0.95, '\n'.join(info_text),
            transform=axes[1].transAxes,
            verticalalignment='top',
            fontfamily='monospace',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8)
        )
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved visualization to: {save_path}")
        
        plt.show()
    
    def visualize_all_classes(
        self,
        save_path: Optional[Union[str, Path]] = None,
        samples_per_class: int = 1
    ) -> None:
        """
        Create a grid showing samples from all defect classes.
        
        Parameters
        ----------
        save_path : str or Path, optional
            Path to save the figure
        samples_per_class : int, optional
            Number of samples per class to show
        """
        import matplotlib.pyplot as plt
        
        num_classes = len(self.DEFECT_CLASSES)
        fig, axes = plt.subplots(
            samples_per_class, num_classes, 
            figsize=(4 * num_classes, 4 * samples_per_class)
        )
        
        if samples_per_class == 1:
            axes = axes.reshape(1, -1)
        
        for col, defect_class in enumerate(self.DEFECT_CLASSES):
            samples = self.samples.get(defect_class, [])
            
            for row in range(samples_per_class):
                ax = axes[row, col]
                
                if row < len(samples):
                    sample = samples[row]
                    image = sample.load_image()
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    ax.imshow(image_rgb)
                
                if row == 0:
                    ax.set_title(
                        defect_class.replace('_', '\n'), 
                        fontsize=11, fontweight='bold'
                    )
                
                ax.axis('off')
        
        plt.suptitle(
            'PCB Defect Classes Overview', 
            fontsize=16, fontweight='bold', y=1.02
        )
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved overview to: {save_path}")
        
        plt.show()
    
    # ========================================================
    # PUBLIC API - TEMPLATE-TEST PAIR CREATION
    # ========================================================
    
    def create_template_test_pairs(
        self,
        output_dir: Union[str, Path],
        num_pairs_per_class: int = 5,
        random_seed: Optional[int] = 42,
        resize_to: Optional[Tuple[int, int]] = None,
        copy_files: bool = True
    ) -> List[dict]:
        """
        Create template-test image pairs for image subtraction.
        
        This method:
        1. Matches defect images to their corresponding template PCB
        2. Optionally copies/resizes images to the output directory
        3. Creates metadata file for tracking pairs
        
        Parameters
        ----------
        output_dir : str or Path
            Output directory for paired images
        num_pairs_per_class : int, optional
            Number of pairs to create per defect class
            Use -1 for all samples
        random_seed : int, optional
            Random seed for reproducibility
        resize_to : tuple, optional
            Target size (width, height) to resize images
        copy_files : bool, optional
            Whether to copy files to output directory
            
        Returns
        -------
        list
            List of dictionaries containing pair metadata
            
        Examples
        --------
        >>> pairs = loader.create_template_test_pairs(
        ...     output_dir='data/raw',
        ...     num_pairs_per_class=5
        ... )
        >>> print(f"Created {len(pairs)} pairs")
        """
        output_dir = Path(output_dir)
        templates_out = output_dir / 'templates'
        test_images_out = output_dir / 'test_images'
        
        # Create directories
        templates_out.mkdir(parents=True, exist_ok=True)
        test_images_out.mkdir(parents=True, exist_ok=True)
        
        if random_seed is not None:
            random.seed(random_seed)
        
        pairs = []
        pair_id = 1
        copied_templates = set()
        
        print("\n" + "="*66)
        print("    CREATING TEMPLATE-TEST PAIRS")
        print("="*66)
        
        for defect_class in self.DEFECT_CLASSES:
            samples = self.samples.get(defect_class, [])
            
            if not samples:
                print(f"⚠ No samples for {defect_class}")
                continue
            
            # Select samples
            if num_pairs_per_class == -1:
                selected = samples
            else:
                selected = random.sample(
                    samples, 
                    min(num_pairs_per_class, len(samples))
                )
            
            print(f"\n📁 {defect_class}: Creating {len(selected)} pairs")
            
            for sample in selected:
                pcb_id = sample.pcb_id
                
                # Check if template exists
                if pcb_id not in self.templates:
                    print(f"   ⚠ No template for PCB {pcb_id}, skipping")
                    continue
                
                template_src = self.templates[pcb_id]
                
                # Define output paths
                template_out_name = f"template_{pcb_id}.jpg"
                template_out_path = templates_out / template_out_name
                
                test_out_name = f"test_{pair_id:03d}_{defect_class}.jpg"
                test_out_path = test_images_out / test_out_name
                
                # Copy/process files if requested
                if copy_files:
                    # Copy template if not already copied
                    if pcb_id not in copied_templates:
                        if resize_to:
                            img = cv2.imread(str(template_src))
                            img = cv2.resize(img, resize_to)
                            cv2.imwrite(str(template_out_path), img)
                        else:
                            shutil.copy(template_src, template_out_path)
                        copied_templates.add(pcb_id)
                    
                    # Copy test image
                    if resize_to:
                        img = cv2.imread(str(sample.image_path))
                        img = cv2.resize(img, resize_to)
                        cv2.imwrite(str(test_out_path), img)
                    else:
                        shutil.copy(sample.image_path, test_out_path)
                
                # Load annotation if available
                annotation_data = None
                if sample.annotation_path:
                    try:
                        ann = self._parse_annotation(sample.annotation_path)
                        annotation_data = ann.to_dict()
                    except:
                        pass
                
                # Create pair record
                pair = {
                    'id': pair_id,
                    'defect_class': defect_class,
                    'pcb_id': pcb_id,
                    'template': str(template_out_path) if copy_files else str(template_src),
                    'template_original': str(template_src),
                    'test': str(test_out_path) if copy_files else str(sample.image_path),
                    'test_original': str(sample.image_path),
                    'annotation': annotation_data
                }
                
                pairs.append(pair)
                pair_id += 1
        
        # Save metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'dataset_path': str(self.dataset_path),
            'total_pairs': len(pairs),
            'pairs_per_class': {
                dc: sum(1 for p in pairs if p['defect_class'] == dc)
                for dc in self.DEFECT_CLASSES
            },
            'resize_to': resize_to,
            'pairs': pairs
        }
        
        metadata_path = output_dir / 'pairs_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n{'─'*66}")
        print(f"✅ PAIR CREATION COMPLETE")
        print(f"{'─'*66}")
        print(f"   Total pairs created: {len(pairs)}")
        print(f"   Templates copied: {len(copied_templates)}")
        print(f"   Output directory: {output_dir}")
        print(f"   Metadata saved to: {metadata_path}")
        print("="*66 + "\n")
        
        return pairs
    
    def verify_pairs(
        self, 
        metadata_path: Union[str, Path],
        show_samples: int = 3
    ) -> bool:
        """
        Verify created pairs are valid and displayable.
        
        Parameters
        ----------
        metadata_path : str or Path
            Path to pairs_metadata.json
        show_samples : int, optional
            Number of sample pairs to display
            
        Returns
        -------
        bool
            True if all pairs are valid
        """
        import matplotlib.pyplot as plt
        
        metadata_path = Path(metadata_path)
        
        if not metadata_path.exists():
            print(f"❌ Metadata file not found: {metadata_path}")
            return False
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        pairs = metadata['pairs']
        valid = 0
        invalid = 0
        
        print("\n" + "="*50)
        print("    VERIFYING TEMPLATE-TEST PAIRS")
        print("="*50)
        
        for pair in pairs:
            template_exists = Path(pair['template']).exists()
            test_exists = Path(pair['test']).exists()
            
            if template_exists and test_exists:
                valid += 1
            else:
                invalid += 1
                print(f"❌ Pair {pair['id']}: Missing files")
                if not template_exists:
                    print(f"   Template: {pair['template']}")
                if not test_exists:
                    print(f"   Test: {pair['test']}")
        
        print(f"\n✅ Valid pairs: {valid}/{len(pairs)}")
        if invalid > 0:
            print(f"❌ Invalid pairs: {invalid}")
        
        # Show sample pairs
        if show_samples > 0 and valid > 0:
            print(f"\n📸 Displaying {min(show_samples, valid)} sample pairs...")
            
            fig, axes = plt.subplots(
                min(show_samples, valid), 2, 
                figsize=(12, 4 * min(show_samples, valid))
            )
            
            if show_samples == 1:
                axes = axes.reshape(1, -1)
            
            sample_pairs = [p for p in pairs[:show_samples] 
                           if Path(p['template']).exists() and Path(p['test']).exists()]
            
            for i, pair in enumerate(sample_pairs):
                template = cv2.imread(pair['template'])
                template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
                test = cv2.imread(pair['test'])
                test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
                
                axes[i, 0].imshow(template)
                axes[i, 0].set_title(f"Template (PCB {pair['pcb_id']})", fontweight='bold')
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(test)
                axes[i, 1].set_title(f"Test - {pair['defect_class']}", fontweight='bold')
                axes[i, 1].axis('off')
            
            plt.suptitle('Sample Template-Test Pairs', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
        
        print("="*50 + "\n")
        
        return invalid == 0


# ============================================================
# MAIN / CLI INTERFACE
# ============================================================

def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='PCB Dataset Loader - Load and prepare PCB defect detection dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Print dataset statistics
  python pcb_dataset_loader.py /path/to/PCB_DATASET --stats
  
  # Create template-test pairs
  python pcb_dataset_loader.py /path/to/PCB_DATASET --create-pairs data/raw --num-pairs 5
  
  # Visualize all classes
  python pcb_dataset_loader.py /path/to/PCB_DATASET --visualize-all --output overview.png
        """
    )
    
    parser.add_argument(
        'dataset_path',
        type=str,
        help='Path to PCB_DATASET directory'
    )
    
    parser.add_argument(
        '--stats', '-s',
        action='store_true',
        help='Print dataset statistics'
    )
    
    parser.add_argument(
        '--create-pairs', '-c',
        type=str,
        metavar='OUTPUT_DIR',
        help='Create template-test pairs in specified directory'
    )
    
    parser.add_argument(
        '--num-pairs', '-n',
        type=int,
        default=5,
        help='Number of pairs per class (default: 5, use -1 for all)'
    )
    
    parser.add_argument(
        '--visualize-all', '-v',
        action='store_true',
        help='Visualize sample from each class'
    )
    
    parser.add_argument(
        '--visualize-sample',
        type=str,
        nargs=2,
        metavar=('CLASS', 'INDEX'),
        help='Visualize specific sample'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output path for visualizations'
    )
    
    args = parser.parse_args()
    
    # Load dataset
    try:
        loader = PCBDatasetLoader(args.dataset_path)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1
    
    # Execute requested actions
    if args.stats:
        loader.print_statistics()
    
    if args.create_pairs:
        loader.create_template_test_pairs(
            output_dir=args.create_pairs,
            num_pairs_per_class=args.num_pairs
        )
    
    if args.visualize_all:
        loader.visualize_all_classes(save_path=args.output)
    
    if args.visualize_sample:
        defect_class, sample_idx = args.visualize_sample
        loader.visualize_sample_with_annotation(
            defect_class=defect_class,
            sample_idx=int(sample_idx),
            save_path=args.output
        )
    
    # Default action: print stats
    if not any([args.stats, args.create_pairs, args.visualize_all, args.visualize_sample]):
        loader.print_statistics()
    
    return 0


if __name__ == '__main__':
    exit(main())
