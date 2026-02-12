"""
ROI Extractor Module
====================

Advanced Region of Interest (ROI) extraction and dataset organization
for PCB defect detection CNN training.

Features:
---------
- Crop ROIs from images using bounding boxes
- Resize to standardized dimensions (128x128)
- Organize by defect class
- Create train/val/test splits with stratification
- Data augmentation preparation
- Dataset statistics and visualization
- Export to CNN-ready format

Classes:
--------
- ROIExtractor: Main class for ROI extraction and dataset creation
- DatasetSplit: Enum for dataset splits
- ROIMetadata: Dataclass for ROI information

Author: PCB Defect Detection Team
Version: 1.0.0

Usage:
------
>>> from detection.roi_extractor import ROIExtractor
>>> extractor = ROIExtractor(output_dir='data/rois', roi_size=(128, 128))
>>> result = extractor.extract_from_bboxes(image, boxes, defect_class)
>>> extractor.create_dataset_splits(train_ratio=0.7, val_ratio=0.15)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union, Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import shutil
import random
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# ENUMS AND CONSTANTS
# ============================================================

class DatasetSplit(Enum):
    """Dataset split types."""
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    ALL = "all"


# Defect class definitions
DEFECT_CLASSES = [
    'Missing_hole',
    'Mouse_bite', 
    'Open_circuit',
    'Short',
    'Spur',
    'Spurious_copper'
]

# Default ROI size for CNN input
DEFAULT_ROI_SIZE = (128, 128)


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class ROIMetadata:
    """
    Metadata for a single extracted ROI.
    
    Attributes
    ----------
    roi_id : str
        Unique identifier for the ROI
    source_image : str
        Name of source image
    defect_class : str
        Defect class label
    bbox : Dict
        Original bounding box coordinates
    original_size : Tuple[int, int]
        Original ROI size before resize
    final_size : Tuple[int, int]
        Final size after resize
    file_path : str
        Path to saved ROI image
    split : str
        Dataset split (train/val/test)
    """
    
    roi_id: str
    source_image: str
    defect_class: str
    bbox: Dict
    original_size: Tuple[int, int]
    final_size: Tuple[int, int]
    file_path: str
    split: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'roi_id': self.roi_id,
            'source_image': self.source_image,
            'defect_class': self.defect_class,
            'bbox': self.bbox,
            'original_size': {'width': self.original_size[0], 'height': self.original_size[1]},
            'final_size': {'width': self.final_size[0], 'height': self.final_size[1]},
            'file_path': self.file_path,
            'split': self.split
        }


@dataclass
class ExtractionStats:
    """
    Statistics from ROI extraction.
    
    Attributes
    ----------
    total_rois : int
        Total ROIs extracted
    rois_by_class : Dict[str, int]
        Count per defect class
    rois_by_split : Dict[str, int]
        Count per dataset split
    avg_original_size : Tuple[float, float]
        Average original ROI dimensions
    processing_time : float
        Total processing time
    """
    
    total_rois: int = 0
    rois_by_class: Dict[str, int] = field(default_factory=dict)
    rois_by_split: Dict[str, int] = field(default_factory=dict)
    avg_original_size: Tuple[float, float] = (0.0, 0.0)
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'total_rois': self.total_rois,
            'rois_by_class': self.rois_by_class,
            'rois_by_split': self.rois_by_split,
            'avg_original_size': {
                'width': round(self.avg_original_size[0], 1),
                'height': round(self.avg_original_size[1], 1)
            },
            'processing_time': round(self.processing_time, 3)
        }


# ============================================================
# MAIN ROI EXTRACTOR CLASS
# ============================================================

class ROIExtractor:
    """
    Advanced ROI extraction and dataset organization for CNN training.
    
    This class provides comprehensive tools for extracting defect regions,
    resizing to standard dimensions, organizing by class, and creating
    train/val/test splits for deep learning.
    
    Attributes
    ----------
    output_dir : Path
        Base output directory for ROIs
    roi_size : Tuple[int, int]
        Target size for all ROIs (width, height)
    defect_classes : List[str]
        List of defect class names
    
    Examples
    --------
    >>> extractor = ROIExtractor('data/rois', roi_size=(128, 128))
    >>> result = extractor.extract_all(images, boxes_list, defect_classes)
    >>> extractor.create_dataset_splits()
    >>> stats = extractor.get_statistics()
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        roi_size: Tuple[int, int] = DEFAULT_ROI_SIZE,
        defect_classes: List[str] = None,
        padding_color: Tuple[int, int, int] = (0, 0, 0),
        verbose: bool = True
    ):
        """
        Initialize the ROIExtractor.
        
        Parameters
        ----------
        output_dir : str or Path
            Base directory for saving ROIs
        roi_size : tuple
            Target (width, height) for all ROIs
        defect_classes : list
            List of defect class names
        padding_color : tuple
            Color for padding when aspect ratio differs
        verbose : bool
            Print progress messages
        """
        self.output_dir = Path(output_dir)
        self.roi_size = roi_size
        self.defect_classes = defect_classes or DEFECT_CLASSES
        self.padding_color = padding_color
        self.verbose = verbose
        
        # Internal storage
        self._all_rois: List[ROIMetadata] = []
        self._stats = ExtractionStats()
        
        # Create directory structure
        self._create_directories()
        
        if self.verbose:
            print(f"✓ ROIExtractor initialized")
            print(f"   Output: {self.output_dir}")
            print(f"   ROI size: {roi_size[0]}×{roi_size[1]}")
            print(f"   Classes: {len(self.defect_classes)}")
    
    def _create_directories(self) -> None:
        """Create output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create class directories
        self.rois_dir = self.output_dir / 'all_rois'
        self.rois_dir.mkdir(exist_ok=True)
        
        for defect_class in self.defect_classes:
            (self.rois_dir / defect_class).mkdir(exist_ok=True)
        
        # Create split directories (will be populated later)
        self.dataset_dir = self.output_dir / 'dataset'
    
    # ========================================================
    # CORE EXTRACTION METHODS
    # ========================================================
    
    def crop_roi(
        self,
        image: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        resize: bool = True,
        maintain_aspect: bool = True
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Crop and resize a single ROI from an image.
        
        Parameters
        ----------
        image : np.ndarray
            Source image
        x1, y1, x2, y2 : int
            Bounding box coordinates
        resize : bool
            Whether to resize to target size
        maintain_aspect : bool
            Maintain aspect ratio with padding
            
        Returns
        -------
        tuple
            (roi_image, original_size)
        """
        # Ensure valid coordinates
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        
        # Crop
        roi = image[y1:y2, x1:x2].copy()
        original_size = (roi.shape[1], roi.shape[0])  # (width, height)
        
        if not resize:
            return roi, original_size
        
        # Resize
        if maintain_aspect:
            roi = self._resize_with_padding(roi)
        else:
            roi = cv2.resize(roi, self.roi_size, interpolation=cv2.INTER_AREA)
        
        return roi, original_size
    
    def _resize_with_padding(self, roi: np.ndarray) -> np.ndarray:
        """Resize ROI with padding to maintain aspect ratio."""
        h, w = roi.shape[:2]
        target_w, target_h = self.roi_size
        
        # Calculate scale
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create padded image
        if len(roi.shape) == 3:
            padded = np.full((target_h, target_w, roi.shape[2]), 
                           self.padding_color, dtype=roi.dtype)
        else:
            padded = np.full((target_h, target_w), 
                           self.padding_color[0], dtype=roi.dtype)
        
        # Center the resized image
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded
    
    def extract_single_roi(
        self,
        image: np.ndarray,
        bbox: Dict,
        defect_class: str,
        source_image: str,
        roi_index: int = 1
    ) -> Optional[ROIMetadata]:
        """
        Extract and save a single ROI.
        
        Parameters
        ----------
        image : np.ndarray
            Source image
        bbox : dict
            Bounding box with x1, y1, x2, y2 or x, y, width, height
        defect_class : str
            Defect class label
        source_image : str
            Source image name
        roi_index : int
            Index for naming
            
        Returns
        -------
        ROIMetadata or None
            Metadata for extracted ROI
        """
        # Parse bbox coordinates
        if 'x1' in bbox:
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        elif 'x' in bbox:
            x1 = bbox['x']
            y1 = bbox['y']
            x2 = x1 + bbox.get('width', bbox.get('w', 50))
            y2 = y1 + bbox.get('height', bbox.get('h', 50))
        else:
            return None
        
        # Crop ROI
        roi, original_size = self.crop_roi(image, x1, y1, x2, y2)
        
        if roi.size == 0:
            return None
        
        # Generate filename
        source_stem = Path(source_image).stem
        roi_id = f"{source_stem}_roi_{roi_index:03d}"
        filename = f"{roi_id}.png"
        
        # Ensure class directory exists (for dynamic class names)
        class_dir = self.rois_dir / defect_class
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # Save path
        save_path = class_dir / filename
        
        # Save ROI (convert to BGR if needed for OpenCV)
        if len(roi.shape) == 3 and roi.shape[2] == 3:
            # Check if RGB (needs conversion to BGR for cv2.imwrite)
            cv2.imwrite(str(save_path), roi)
        else:
            cv2.imwrite(str(save_path), roi)
        
        # Create metadata
        metadata = ROIMetadata(
            roi_id=roi_id,
            source_image=source_image,
            defect_class=defect_class,
            bbox={'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
            original_size=original_size,
            final_size=self.roi_size,
            file_path=str(save_path)
        )
        
        self._all_rois.append(metadata)
        
        return metadata
    
    def extract_from_image(
        self,
        image: np.ndarray,
        bboxes: List[Dict],
        defect_class: str,
        source_image: str
    ) -> List[ROIMetadata]:
        """
        Extract all ROIs from a single image.
        
        Parameters
        ----------
        image : np.ndarray
            Source image
        bboxes : list
            List of bounding box dictionaries
        defect_class : str
            Defect class for all ROIs
        source_image : str
            Source image name
            
        Returns
        -------
        list
            List of ROIMetadata for extracted ROIs
        """
        extracted = []
        
        for i, bbox in enumerate(bboxes):
            metadata = self.extract_single_roi(
                image, bbox, defect_class, source_image, roi_index=i+1
            )
            if metadata:
                extracted.append(metadata)
        
        return extracted
    
    def extract_from_detection_result(
        self,
        image: np.ndarray,
        extraction_result: 'ExtractionResult',
        defect_class: str,
        source_image: str
    ) -> List[ROIMetadata]:
        """
        Extract ROIs from BoundingBoxExtractor result.
        
        Parameters
        ----------
        image : np.ndarray
            Source image
        extraction_result : ExtractionResult
            Result from BoundingBoxExtractor
        defect_class : str
            Defect class label
        source_image : str
            Source image name
            
        Returns
        -------
        list
            List of extracted ROI metadata
        """
        bboxes = [
            {
                'x1': box.x1, 'y1': box.y1,
                'x2': box.x2, 'y2': box.y2
            }
            for box in extraction_result.boxes
        ]
        
        return self.extract_from_image(image, bboxes, defect_class, source_image)
    
    # ========================================================
    # BATCH EXTRACTION
    # ========================================================
    
    def extract_batch(
        self,
        image_paths: List[Path],
        mask_paths: List[Path],
        defect_classes: List[str],
        detector: 'ContourDetector' = None,
        extractor: 'BoundingBoxExtractor' = None
    ) -> ExtractionStats:
        """
        Extract ROIs from multiple images.
        
        Parameters
        ----------
        image_paths : list
            Paths to source images
        mask_paths : list
            Paths to binary masks
        defect_classes : list
            Defect class for each image
        detector : ContourDetector
            Contour detector instance
        extractor : BoundingBoxExtractor
            Bounding box extractor instance
            
        Returns
        -------
        ExtractionStats
            Statistics from extraction
        """
        start_time = time.time()
        
        # Import if not provided
        if detector is None:
            from .contour_detector import ContourDetector
            detector = ContourDetector(min_area=50, max_area=10000, verbose=False)
        
        if extractor is None:
            from .bbox_extractor import BoundingBoxExtractor
            extractor = BoundingBoxExtractor(expand_pixels=10, verbose=False)
        
        total_extracted = 0
        class_counts = defaultdict(int)
        all_sizes = []
        
        for i, (img_path, mask_path, defect_class) in enumerate(
            zip(image_paths, mask_paths, defect_classes)
        ):
            if self.verbose:
                print(f"   [{i+1}/{len(image_paths)}] {img_path.name}...", end=" ")
            
            # Load image and mask
            image = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            if image is None or mask is None:
                if self.verbose:
                    print("SKIP (load failed)")
                continue
            
            # Detect contours
            detection = detector.detect(mask)
            
            if detection.num_contours == 0:
                if self.verbose:
                    print("0 ROIs")
                continue
            
            # Extract bounding boxes
            extractor.image_shape = detection.image_shape
            bbox_result = extractor.extract_from_detection(detection)
            
            # Extract ROIs
            rois = self.extract_from_detection_result(
                image, bbox_result, defect_class, img_path.name
            )
            
            total_extracted += len(rois)
            class_counts[defect_class] += len(rois)
            
            # Track sizes
            for roi in rois:
                all_sizes.append(roi.original_size)
            
            if self.verbose:
                print(f"{len(rois)} ROIs")
        
        elapsed = time.time() - start_time
        
        # Calculate average size
        if all_sizes:
            avg_w = np.mean([s[0] for s in all_sizes])
            avg_h = np.mean([s[1] for s in all_sizes])
        else:
            avg_w, avg_h = 0, 0
        
        # Update stats
        self._stats = ExtractionStats(
            total_rois=total_extracted,
            rois_by_class=dict(class_counts),
            avg_original_size=(avg_w, avg_h),
            processing_time=elapsed
        )
        
        if self.verbose:
            print(f"\n✓ Extracted {total_extracted} ROIs in {elapsed:.2f}s")
        
        return self._stats
    
    # ========================================================
    # DATASET SPLITTING
    # ========================================================
    
    def create_dataset_splits(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        stratify: bool = True
    ) -> Dict[str, int]:
        """
        Create train/val/test splits from extracted ROIs.
        
        Parameters
        ----------
        train_ratio : float
            Proportion for training (default: 0.70)
        val_ratio : float
            Proportion for validation (default: 0.15)
        test_ratio : float
            Proportion for testing (default: 0.15)
        random_seed : int
            Random seed for reproducibility
        stratify : bool
            Maintain class proportions in splits
            
        Returns
        -------
        dict
            Count per split
        """
        if self.verbose:
            print("\n📊 Creating dataset splits...")
            print(f"   Train: {train_ratio*100:.0f}%")
            print(f"   Val: {val_ratio*100:.0f}%")
            print(f"   Test: {test_ratio*100:.0f}%")
        
        random.seed(random_seed)
        
        # Create split directories
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        for split in ['train', 'val', 'test']:
            split_dir = self.dataset_dir / split
            split_dir.mkdir(exist_ok=True)
            for defect_class in self.defect_classes:
                (split_dir / defect_class).mkdir(exist_ok=True)
        
        # Group ROIs by class
        rois_by_class = defaultdict(list)
        for roi in self._all_rois:
            rois_by_class[roi.defect_class].append(roi)
        
        # Split each class
        split_counts = {'train': 0, 'val': 0, 'test': 0}
        
        for defect_class, class_rois in rois_by_class.items():
            if self.verbose:
                print(f"\n   {defect_class}: {len(class_rois)} ROIs")
            
            # Shuffle
            random.shuffle(class_rois)
            
            # Calculate split indices
            n = len(class_rois)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            # Assign splits
            for i, roi in enumerate(class_rois):
                if i < train_end:
                    split = 'train'
                elif i < val_end:
                    split = 'val'
                else:
                    split = 'test'
                
                roi.split = split
                split_counts[split] += 1
                
                # Copy to split directory
                src_path = Path(roi.file_path)
                dst_dir = self.dataset_dir / split / defect_class
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst_path = dst_dir / src_path.name
                
                if src_path.exists():
                    shutil.copy2(src_path, dst_path)
            
            # Print class split distribution
            class_train = sum(1 for r in class_rois if r.split == 'train')
            class_val = sum(1 for r in class_rois if r.split == 'val')
            class_test = sum(1 for r in class_rois if r.split == 'test')
            
            if self.verbose:
                print(f"      Train: {class_train}, Val: {class_val}, Test: {class_test}")
        
        # Update stats
        self._stats.rois_by_split = split_counts
        
        if self.verbose:
            print(f"\n✓ Dataset splits created:")
            print(f"   Train: {split_counts['train']}")
            print(f"   Val: {split_counts['val']}")
            print(f"   Test: {split_counts['test']}")
        
        return split_counts
    
    # ========================================================
    # STATISTICS AND REPORTING
    # ========================================================
    
    def get_statistics(self) -> ExtractionStats:
        """Get current extraction statistics."""
        return self._stats
    
    def generate_report(self, output_path: Union[str, Path] = None) -> str:
        """
        Generate comprehensive extraction report.
        
        Parameters
        ----------
        output_path : str or Path
            Path to save report (optional)
            
        Returns
        -------
        str
            Report text
        """
        lines = []
        lines.append("=" * 70)
        lines.append("PCB DEFECT DETECTION - ROI EXTRACTION REPORT")
        lines.append("=" * 70)
        lines.append("")
        
        # Overview
        lines.append("OVERVIEW")
        lines.append("-" * 70)
        lines.append(f"Total ROIs extracted: {self._stats.total_rois}")
        lines.append(f"ROI size: {self.roi_size[0]}×{self.roi_size[1]} pixels")
        lines.append(f"Avg original size: {self._stats.avg_original_size[0]:.1f}×"
                    f"{self._stats.avg_original_size[1]:.1f}")
        lines.append(f"Processing time: {self._stats.processing_time:.2f}s")
        lines.append("")
        
        # By class
        lines.append("ROIS BY DEFECT CLASS")
        lines.append("-" * 70)
        for defect_class, count in sorted(self._stats.rois_by_class.items()):
            pct = (count / self._stats.total_rois * 100) if self._stats.total_rois > 0 else 0
            lines.append(f"  {defect_class:<20}: {count:>4} ({pct:>5.1f}%)")
        lines.append("")
        
        # By split
        if self._stats.rois_by_split:
            lines.append("DATASET SPLITS")
            lines.append("-" * 70)
            for split, count in sorted(self._stats.rois_by_split.items()):
                pct = (count / self._stats.total_rois * 100) if self._stats.total_rois > 0 else 0
                lines.append(f"  {split:<10}: {count:>4} ({pct:>5.1f}%)")
        lines.append("")
        
        # Paths
        lines.append("OUTPUT PATHS")
        lines.append("-" * 70)
        lines.append(f"  All ROIs: {self.rois_dir}")
        lines.append(f"  Dataset: {self.dataset_dir}")
        lines.append("")
        
        lines.append("=" * 70)
        
        report = "\n".join(lines)
        
        if output_path:
            output_path = Path(output_path)
            output_path.write_text(report)
            if self.verbose:
                print(f"✓ Report saved to: {output_path}")
        
        return report
    
    def save_metadata(self, output_path: Union[str, Path] = None) -> None:
        """
        Save all ROI metadata to JSON.
        
        Parameters
        ----------
        output_path : str or Path
            Output path (default: output_dir/roi_metadata.json)
        """
        if output_path is None:
            output_path = self.output_dir / 'roi_metadata.json'
        else:
            output_path = Path(output_path)
        
        metadata = {
            'extraction_settings': {
                'roi_size': {'width': self.roi_size[0], 'height': self.roi_size[1]},
                'defect_classes': self.defect_classes,
                'output_dir': str(self.output_dir)
            },
            'statistics': self._stats.to_dict(),
            'rois': [roi.to_dict() for roi in self._all_rois]
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if self.verbose:
            print(f"✓ Metadata saved to: {output_path}")
    
    # ========================================================
    # VISUALIZATION
    # ========================================================
    
    def visualize_samples(
        self,
        samples_per_class: int = 3,
        figsize: Tuple[int, int] = None
    ) -> 'matplotlib.figure.Figure':
        """
        Visualize sample ROIs from each class.
        
        Parameters
        ----------
        samples_per_class : int
            Number of samples to show per class
        figsize : tuple
            Figure size
            
        Returns
        -------
        matplotlib.figure.Figure
            Sample visualization figure
        """
        import matplotlib.pyplot as plt
        
        # Group by class
        rois_by_class = defaultdict(list)
        for roi in self._all_rois:
            rois_by_class[roi.defect_class].append(roi)
        
        # Determine grid size
        n_classes = len([c for c in self.defect_classes if c in rois_by_class])
        
        if figsize is None:
            figsize = (4 * samples_per_class, 4 * n_classes)
        
        fig, axes = plt.subplots(n_classes, samples_per_class, figsize=figsize)
        
        if n_classes == 1:
            axes = axes.reshape(1, -1)
        
        row = 0
        for defect_class in self.defect_classes:
            if defect_class not in rois_by_class:
                continue
            
            class_rois = rois_by_class[defect_class]
            
            for col in range(samples_per_class):
                ax = axes[row, col]
                
                if col < len(class_rois):
                    roi = class_rois[col]
                    roi_img = cv2.imread(roi.file_path)
                    
                    if roi_img is not None:
                        roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
                        ax.imshow(roi_img)
                        
                        if col == 0:
                            ax.set_ylabel(defect_class, fontsize=10, fontweight='bold')
                    
                    ax.set_title(f"ROI {col+1}", fontsize=9)
                
                ax.axis('off')
            
            row += 1
        
        plt.suptitle(f'Sample ROIs ({self.roi_size[0]}×{self.roi_size[1]})', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def visualize_class_distribution(
        self,
        figsize: Tuple[int, int] = (12, 5)
    ) -> 'matplotlib.figure.Figure':
        """
        Visualize class distribution with bar chart.
        
        Parameters
        ----------
        figsize : tuple
            Figure size
            
        Returns
        -------
        matplotlib.figure.Figure
            Distribution figure
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # By class
        classes = list(self._stats.rois_by_class.keys())
        counts = list(self._stats.rois_by_class.values())
        colors = plt.cm.Set2(np.linspace(0, 1, len(classes)))
        
        axes[0].bar(classes, counts, color=colors)
        axes[0].set_title('ROIs by Defect Class', fontweight='bold')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=45)
        
        # By split
        if self._stats.rois_by_split:
            splits = list(self._stats.rois_by_split.keys())
            split_counts = list(self._stats.rois_by_split.values())
            split_colors = ['#2ecc71', '#3498db', '#e74c3c']
            
            axes[1].pie(split_counts, labels=splits, autopct='%1.1f%%', 
                       colors=split_colors, startangle=90)
            axes[1].set_title('Dataset Splits', fontweight='bold')
        else:
            axes[1].text(0.5, 0.5, 'No splits created yet', 
                        ha='center', va='center')
            axes[1].set_title('Dataset Splits', fontweight='bold')
        
        plt.tight_layout()
        
        return fig
    
    # ========================================================
    # UTILITY METHODS
    # ========================================================
    
    def get_rois_for_split(self, split: str) -> List[ROIMetadata]:
        """Get all ROIs for a specific split."""
        return [roi for roi in self._all_rois if roi.split == split]
    
    def get_rois_for_class(self, defect_class: str) -> List[ROIMetadata]:
        """Get all ROIs for a specific class."""
        return [roi for roi in self._all_rois if roi.defect_class == defect_class]
    
    def get_split_paths(self) -> Dict[str, Path]:
        """Get paths to split directories."""
        return {
            'train': self.dataset_dir / 'train',
            'val': self.dataset_dir / 'val',
            'test': self.dataset_dir / 'test'
        }
    
    def clear(self) -> None:
        """Clear all extracted ROIs and reset state."""
        self._all_rois = []
        self._stats = ExtractionStats()
        
        # Remove files
        if self.rois_dir.exists():
            shutil.rmtree(self.rois_dir)
        if self.dataset_dir.exists():
            shutil.rmtree(self.dataset_dir)
        
        # Recreate directories
        self._create_directories()
        
        if self.verbose:
            print("✓ Cleared all ROIs and reset state")


# ============================================================
# MAIN / CLI
# ============================================================

def main():
    """Command line interface for ROI extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='PCB Defect ROI Extraction'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/rois',
        help='Output directory'
    )
    
    parser.add_argument(
        '--size',
        type=int,
        default=128,
        help='ROI size (default: 128)'
    )
    
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.70,
        help='Training set ratio'
    )
    
    args = parser.parse_args()
    
    print("ROI Extractor CLI")
    print("Use the quickstart script for full functionality:")
    print("  python quickstart_module2_part3.py")
    
    return 0


if __name__ == '__main__':
    exit(main())
