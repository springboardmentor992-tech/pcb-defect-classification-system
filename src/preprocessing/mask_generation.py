"""
Mask Generation Module
======================

Creates binary masks from difference maps to identify defect regions.
Implements thresholding techniques, morphological operations, and
contour-based region extraction.

Features:
---------
- Otsu's automatic thresholding
- Adaptive thresholding
- Multi-level thresholding
- Morphological cleanup (opening, closing, erosion, dilation)
- Contour detection and filtering
- Connected component analysis
- Defect region extraction

Author: PCB Defect Detection Team
Version: 1.0.0

Usage:
------
>>> from mask_generation import MaskGenerator
>>> generator = MaskGenerator(threshold_method='otsu')
>>> result = generator.generate_mask(difference_map)
>>> contours = result.contours
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union, Dict, List
from dataclasses import dataclass, field
from enum import Enum
import json
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_utils import (
    load_image,
    save_image,
    convert_to_grayscale,
    apply_morphological_operations,
    clean_binary_mask
)


# ============================================================
# ENUMS AND DATA CLASSES
# ============================================================

class ThresholdMethod(Enum):
    """Supported thresholding methods."""
    OTSU = "otsu"                    # Otsu's automatic thresholding
    ADAPTIVE_MEAN = "adaptive_mean"  # Adaptive with mean calculation
    ADAPTIVE_GAUSSIAN = "adaptive_gaussian"  # Adaptive with Gaussian weights
    BINARY = "binary"                # Simple binary threshold
    TRIANGLE = "triangle"            # Triangle algorithm
    MULTI_OTSU = "multi_otsu"       # Multi-level Otsu


@dataclass
class BoundingBox:
    """Bounding box for a detected region."""
    x: int
    y: int
    width: int
    height: int
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def aspect_ratio(self) -> float:
        return self.width / max(self.height, 1)
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Return (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)
    
    def to_xyxy(self) -> Tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2)."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    def to_dict(self) -> dict:
        return {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'area': self.area,
            'center': self.center,
            'aspect_ratio': round(self.aspect_ratio, 3)
        }


@dataclass
class DetectedRegion:
    """A detected potential defect region."""
    contour: np.ndarray
    bounding_box: BoundingBox
    area: float
    perimeter: float
    circularity: float
    solidity: float
    convex_hull: np.ndarray
    
    @classmethod
    def from_contour(cls, contour: np.ndarray) -> 'DetectedRegion':
        """Create DetectedRegion from a contour."""
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        # Circularity: 1.0 = perfect circle
        circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-8)
        
        # Solidity: ratio of contour area to convex hull area
        solidity = area / (hull_area + 1e-8)
        
        return cls(
            contour=contour,
            bounding_box=BoundingBox(x, y, w, h),
            area=area,
            perimeter=perimeter,
            circularity=circularity,
            solidity=solidity,
            convex_hull=hull
        )
    
    def to_dict(self) -> dict:
        return {
            'bounding_box': self.bounding_box.to_dict(),
            'area': round(self.area, 2),
            'perimeter': round(self.perimeter, 2),
            'circularity': round(self.circularity, 4),
            'solidity': round(self.solidity, 4)
        }


@dataclass
class MaskResult:
    """Results from mask generation."""
    binary_mask: np.ndarray
    cleaned_mask: np.ndarray
    threshold_value: float
    threshold_method: str
    contours: List[DetectedRegion]
    processing_time: float
    stats: Dict = field(default_factory=dict)
    
    @property
    def num_regions(self) -> int:
        return len(self.contours)
    
    @property
    def total_defect_area(self) -> float:
        return sum(r.area for r in self.contours)
    
    def to_dict(self) -> dict:
        return {
            'threshold_value': round(self.threshold_value, 2),
            'threshold_method': self.threshold_method,
            'num_regions': self.num_regions,
            'total_defect_area': round(self.total_defect_area, 2),
            'processing_time': round(self.processing_time, 4),
            'regions': [r.to_dict() for r in self.contours],
            'stats': self.stats
        }


# ============================================================
# MAIN MASK GENERATOR CLASS
# ============================================================

class MaskGenerator:
    """
    Generate binary masks from difference maps for defect detection.
    
    This class provides multiple thresholding methods and morphological
    operations to create clean binary masks highlighting defect regions.
    
    Attributes
    ----------
    threshold_method : ThresholdMethod
        Method for binarization
    threshold_value : int
        Manual threshold value (for binary method)
    morph_kernel_size : int
        Kernel size for morphological operations
    min_area : int
        Minimum contour area to keep
    max_area : int
        Maximum contour area to keep
    
    Examples
    --------
    >>> generator = MaskGenerator(threshold_method='otsu')
    >>> result = generator.generate_mask(difference_map)
    >>> print(f"Found {result.num_regions} potential defects")
    """
    
    # Default parameters
    DEFAULT_THRESHOLD = 30
    DEFAULT_MORPH_KERNEL = 3
    DEFAULT_MIN_AREA = 50
    DEFAULT_MAX_AREA = 50000
    DEFAULT_ADAPTIVE_BLOCK = 11
    DEFAULT_ADAPTIVE_C = 2
    
    def __init__(
        self,
        threshold_method: Union[str, ThresholdMethod] = 'otsu',
        threshold_value: int = DEFAULT_THRESHOLD,
        morph_kernel_size: int = DEFAULT_MORPH_KERNEL,
        min_area: int = DEFAULT_MIN_AREA,
        max_area: int = DEFAULT_MAX_AREA,
        adaptive_block_size: int = DEFAULT_ADAPTIVE_BLOCK,
        adaptive_c: int = DEFAULT_ADAPTIVE_C,
        verbose: bool = True
    ):
        """
        Initialize the MaskGenerator.
        
        Parameters
        ----------
        threshold_method : str or ThresholdMethod
            Thresholding method: 'otsu', 'adaptive_mean', 'adaptive_gaussian', 
            'binary', 'triangle', 'multi_otsu'
        threshold_value : int
            Threshold value for binary method (0-255)
        morph_kernel_size : int
            Kernel size for morphological operations (odd number)
        min_area : int
            Minimum contour area to keep
        max_area : int
            Maximum contour area to keep
        adaptive_block_size : int
            Block size for adaptive thresholding (odd number)
        adaptive_c : int
            Constant subtracted from mean in adaptive thresholding
        verbose : bool
            Print progress messages
        """
        if isinstance(threshold_method, str):
            threshold_method = ThresholdMethod(threshold_method.lower())
        self.threshold_method = threshold_method
        
        self.threshold_value = threshold_value
        self.morph_kernel_size = morph_kernel_size | 1  # Ensure odd
        self.min_area = min_area
        self.max_area = max_area
        self.adaptive_block_size = adaptive_block_size | 1  # Ensure odd
        self.adaptive_c = adaptive_c
        self.verbose = verbose
        
        if self.verbose:
            print(f"✓ MaskGenerator initialized with {self.threshold_method.value}")
    
    # ========================================================
    # THRESHOLDING METHODS
    # ========================================================
    
    def apply_threshold(
        self,
        image: np.ndarray,
        method: Optional[ThresholdMethod] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Apply thresholding to create binary mask.
        
        Parameters
        ----------
        image : np.ndarray
            Grayscale input image
        method : ThresholdMethod, optional
            Override default method
            
        Returns
        -------
        tuple
            (binary_mask, threshold_value)
        """
        method = method or self.threshold_method
        
        # Ensure grayscale
        if len(image.shape) == 3:
            image = convert_to_grayscale(image)
        
        if method == ThresholdMethod.OTSU:
            thresh_val, binary = cv2.threshold(
                image, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        
        elif method == ThresholdMethod.TRIANGLE:
            thresh_val, binary = cv2.threshold(
                image, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
            )
        
        elif method == ThresholdMethod.BINARY:
            thresh_val = self.threshold_value
            _, binary = cv2.threshold(
                image, thresh_val, 255,
                cv2.THRESH_BINARY
            )
        
        elif method == ThresholdMethod.ADAPTIVE_MEAN:
            binary = cv2.adaptiveThreshold(
                image, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                self.adaptive_block_size,
                self.adaptive_c
            )
            thresh_val = -1  # Adaptive doesn't have single threshold
        
        elif method == ThresholdMethod.ADAPTIVE_GAUSSIAN:
            binary = cv2.adaptiveThreshold(
                image, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                self.adaptive_block_size,
                self.adaptive_c
            )
            thresh_val = -1
        
        elif method == ThresholdMethod.MULTI_OTSU:
            # Multi-level Otsu (2 thresholds)
            try:
                thresholds = self._multi_otsu(image, levels=3)
                thresh_val = thresholds[0]  # Use first threshold
                _, binary = cv2.threshold(
                    image, thresh_val, 255,
                    cv2.THRESH_BINARY
                )
            except:
                # Fallback to regular Otsu
                thresh_val, binary = cv2.threshold(
                    image, 0, 255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
        
        else:
            raise ValueError(f"Unknown threshold method: {method}")
        
        return binary, thresh_val
    
    def _multi_otsu(self, image: np.ndarray, levels: int = 3) -> List[float]:
        """
        Compute multi-level Otsu thresholds.
        
        Parameters
        ----------
        image : np.ndarray
            Grayscale image
        levels : int
            Number of levels (thresholds = levels - 1)
            
        Returns
        -------
        list
            List of threshold values
        """
        hist, _ = np.histogram(image.ravel(), 256, [0, 256])
        hist = hist.astype(np.float64)
        hist /= hist.sum()
        
        # Simple 2-level Otsu
        total = np.arange(256)
        threshold1 = 0
        max_variance = 0
        
        for t in range(1, 255):
            # Class 0: [0, t], Class 1: [t+1, 255]
            w0 = hist[:t+1].sum()
            w1 = hist[t+1:].sum()
            
            if w0 == 0 or w1 == 0:
                continue
            
            mu0 = (hist[:t+1] * np.arange(t+1)).sum() / w0
            mu1 = (hist[t+1:] * np.arange(t+1, 256)).sum() / w1
            
            variance = w0 * w1 * (mu0 - mu1) ** 2
            
            if variance > max_variance:
                max_variance = variance
                threshold1 = t
        
        return [float(threshold1)]
    
    # ========================================================
    # MORPHOLOGICAL OPERATIONS
    # ========================================================
    
    def apply_morphology(
        self,
        mask: np.ndarray,
        operations: List[Tuple[str, int]] = None
    ) -> np.ndarray:
        """
        Apply sequence of morphological operations.
        
        Parameters
        ----------
        mask : np.ndarray
            Binary mask
        operations : list of tuples, optional
            List of (operation_name, kernel_size) tuples
            Default: [('open', 3), ('close', 3)]
            
        Returns
        -------
        np.ndarray
            Cleaned mask
        """
        if operations is None:
            operations = [
                ('open', self.morph_kernel_size),
                ('close', self.morph_kernel_size)
            ]
        
        result = mask.copy()
        
        for op_name, kernel_size in operations:
            result = apply_morphological_operations(
                result, 
                operation=op_name, 
                kernel_size=kernel_size,
                kernel_shape='ellipse'
            )
        
        return result
    
    def remove_small_objects(
        self,
        mask: np.ndarray,
        min_size: int = None
    ) -> np.ndarray:
        """
        Remove small connected components.
        
        Parameters
        ----------
        mask : np.ndarray
            Binary mask
        min_size : int, optional
            Minimum component size in pixels
            
        Returns
        -------
        np.ndarray
            Cleaned mask
        """
        min_size = min_size or self.min_area
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        
        # Create output mask
        cleaned = np.zeros_like(mask)
        
        # Keep components larger than min_size (skip background label 0)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_size:
                cleaned[labels == i] = 255
        
        return cleaned
    
    def fill_holes(self, mask: np.ndarray) -> np.ndarray:
        """
        Fill holes in binary mask.
        
        Parameters
        ----------
        mask : np.ndarray
            Binary mask
            
        Returns
        -------
        np.ndarray
            Mask with holes filled
        """
        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Fill each contour
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 255, -1)
        
        return filled
    
    # ========================================================
    # CONTOUR DETECTION
    # ========================================================
    
    def find_contours(
        self,
        mask: np.ndarray,
        min_area: int = None,
        max_area: int = None
    ) -> List[DetectedRegion]:
        """
        Find contours and create DetectedRegion objects.
        
        Parameters
        ----------
        mask : np.ndarray
            Binary mask
        min_area : int, optional
            Minimum contour area
        max_area : int, optional
            Maximum contour area
            
        Returns
        -------
        list of DetectedRegion
            Detected regions with properties
        """
        min_area = min_area or self.min_area
        max_area = max_area or self.max_area
        
        # Find contours
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter and create DetectedRegion objects
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if min_area <= area <= max_area:
                region = DetectedRegion.from_contour(contour)
                regions.append(region)
        
        # Sort by area (largest first)
        regions.sort(key=lambda r: r.area, reverse=True)
        
        return regions
    
    def filter_regions(
        self,
        regions: List[DetectedRegion],
        min_circularity: float = 0.0,
        max_circularity: float = 1.0,
        min_solidity: float = 0.0,
        min_aspect_ratio: float = 0.0,
        max_aspect_ratio: float = float('inf')
    ) -> List[DetectedRegion]:
        """
        Filter regions based on shape properties.
        
        Parameters
        ----------
        regions : list of DetectedRegion
            Input regions
        min_circularity : float
            Minimum circularity (0-1)
        max_circularity : float
            Maximum circularity (0-1)
        min_solidity : float
            Minimum solidity (0-1)
        min_aspect_ratio : float
            Minimum width/height ratio
        max_aspect_ratio : float
            Maximum width/height ratio
            
        Returns
        -------
        list of DetectedRegion
            Filtered regions
        """
        filtered = []
        
        for region in regions:
            if not (min_circularity <= region.circularity <= max_circularity):
                continue
            if region.solidity < min_solidity:
                continue
            aspect = region.bounding_box.aspect_ratio
            if not (min_aspect_ratio <= aspect <= max_aspect_ratio):
                continue
            filtered.append(region)
        
        return filtered
    
    # ========================================================
    # MAIN PIPELINE
    # ========================================================
    
    def generate_mask(
        self,
        difference_map: np.ndarray,
        apply_morphology: bool = True,
        remove_small: bool = True,
        fill_holes: bool = False
    ) -> MaskResult:
        """
        Generate binary mask from difference map.
        
        This is the main entry point for mask generation.
        
        Parameters
        ----------
        difference_map : np.ndarray
            Grayscale difference map from image subtraction
        apply_morphology : bool
            Apply morphological cleanup
        remove_small : bool
            Remove small connected components
        fill_holes : bool
            Fill holes in regions
            
        Returns
        -------
        MaskResult
            Complete mask generation results
        """
        start_time = time.time()
        
        # Ensure grayscale
        if len(difference_map.shape) == 3:
            difference_map = convert_to_grayscale(difference_map)
        
        if self.verbose:
            print(f"\n🔄 Generating mask...")
            print(f"   Method: {self.threshold_method.value}")
        
        # Step 1: Apply thresholding
        binary_mask, thresh_val = self.apply_threshold(difference_map)
        
        if self.verbose:
            thresh_str = f"{thresh_val:.1f}" if thresh_val >= 0 else "adaptive"
            print(f"   Threshold: {thresh_str}")
            print(f"   Initial white pixels: {(binary_mask > 0).sum()}")
        
        # Step 2: Morphological cleanup
        if apply_morphology:
            cleaned_mask = self.apply_morphology(binary_mask)
        else:
            cleaned_mask = binary_mask.copy()
        
        # Step 3: Remove small objects
        if remove_small:
            cleaned_mask = self.remove_small_objects(cleaned_mask)
        
        # Step 4: Fill holes
        if fill_holes:
            cleaned_mask = self.fill_holes(cleaned_mask)
        
        if self.verbose:
            print(f"   Cleaned white pixels: {(cleaned_mask > 0).sum()}")
        
        # Step 5: Find contours
        contours = self.find_contours(cleaned_mask)
        
        if self.verbose:
            print(f"   Regions found: {len(contours)}")
        
        # Calculate statistics
        stats = {
            'input_size': difference_map.shape,
            'initial_white_pixels': int((binary_mask > 0).sum()),
            'cleaned_white_pixels': int((cleaned_mask > 0).sum()),
            'mask_coverage': round((cleaned_mask > 0).sum() / cleaned_mask.size * 100, 4)
        }
        
        elapsed = time.time() - start_time
        
        result = MaskResult(
            binary_mask=binary_mask,
            cleaned_mask=cleaned_mask,
            threshold_value=thresh_val,
            threshold_method=self.threshold_method.value,
            contours=contours,
            processing_time=elapsed,
            stats=stats
        )
        
        if self.verbose:
            print(f"   ✓ Mask generated in {elapsed:.3f}s")
        
        return result
    
    def process_difference_map(
        self,
        difference_map: Union[np.ndarray, str, Path],
        original_image: Optional[np.ndarray] = None,
        output_dir: Optional[Union[str, Path]] = None,
        save_visualizations: bool = True,
        visualize: bool = False,
        output_prefix: str = ""
    ) -> MaskResult:
        """
        Process a difference map and optionally save outputs.
        
        Parameters
        ----------
        difference_map : ndarray, str, or Path
            Difference map image or path
        original_image : ndarray, optional
            Original image for overlay visualization
        output_dir : str or Path, optional
            Directory to save outputs
        save_visualizations : bool
            Save visualization images
        visualize : bool
            Display visualizations
        output_prefix : str
            Prefix for output filenames
            
        Returns
        -------
        MaskResult
            Complete results
        """
        # Load if path
        if isinstance(difference_map, (str, Path)):
            difference_map = load_image(difference_map, color_mode='gray')
        
        # Generate mask
        result = self.generate_mask(
            difference_map,
            apply_morphology=True,
            remove_small=True
        )
        
        # Save outputs
        if output_dir:
            self._save_outputs(
                result, difference_map, original_image,
                output_dir, output_prefix, save_visualizations
            )
        
        # Visualize
        if visualize:
            self._visualize_result(result, difference_map, original_image)
        
        return result
    
    def _save_outputs(
        self,
        result: MaskResult,
        difference_map: np.ndarray,
        original_image: Optional[np.ndarray],
        output_dir: Union[str, Path],
        prefix: str,
        save_viz: bool
    ) -> None:
        """Save processing outputs."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        name = f"{prefix}_" if prefix else ""
        
        # Save binary mask
        save_image(result.binary_mask, output_dir / f"{name}binary_mask.png")
        
        # Save cleaned mask
        save_image(result.cleaned_mask, output_dir / f"{name}cleaned_mask.png")
        
        if save_viz:
            # Create contour visualization
            viz = self.draw_contours_on_image(
                cv2.cvtColor(difference_map, cv2.COLOR_GRAY2BGR),
                result.contours
            )
            save_image(viz, output_dir / f"{name}contours.png")
            
            # Overlay on original if available
            if original_image is not None:
                overlay = self.create_mask_overlay(original_image, result.cleaned_mask)
                save_image(overlay, output_dir / f"{name}overlay.png")
        
        # Save metadata
        meta_path = output_dir / f"{name}mask_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        if self.verbose:
            print(f"   ✓ Outputs saved to: {output_dir}")
    
    def _visualize_result(
        self,
        result: MaskResult,
        difference_map: np.ndarray,
        original_image: Optional[np.ndarray]
    ) -> None:
        """Display visualization of results."""
        import matplotlib.pyplot as plt
        
        # Determine layout
        if original_image is not None:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        else:
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            axes = axes.reshape(1, -1)
        
        # Row 1: Difference, Binary, Cleaned, Contours
        axes.flat[0].imshow(difference_map, cmap='gray')
        axes.flat[0].set_title('Difference Map', fontweight='bold')
        axes.flat[0].axis('off')
        
        axes.flat[1].imshow(result.binary_mask, cmap='gray')
        axes.flat[1].set_title(f'Binary (thresh={result.threshold_value:.0f})', fontweight='bold')
        axes.flat[1].axis('off')
        
        axes.flat[2].imshow(result.cleaned_mask, cmap='gray')
        axes.flat[2].set_title(f'Cleaned ({result.num_regions} regions)', fontweight='bold')
        axes.flat[2].axis('off')
        
        # Contour visualization
        contour_viz = self.draw_contours_on_image(
            cv2.cvtColor(difference_map, cv2.COLOR_GRAY2BGR),
            result.contours
        )
        axes.flat[3].imshow(cv2.cvtColor(contour_viz, cv2.COLOR_BGR2RGB))
        axes.flat[3].set_title('Detected Regions', fontweight='bold')
        axes.flat[3].axis('off')
        
        # Row 2 if original available
        if original_image is not None and len(axes.flat) > 4:
            axes.flat[4].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            axes.flat[4].set_title('Original Image', fontweight='bold')
            axes.flat[4].axis('off')
            
            overlay = self.create_mask_overlay(original_image, result.cleaned_mask)
            axes.flat[5].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            axes.flat[5].set_title('Defect Overlay', fontweight='bold')
            axes.flat[5].axis('off')
        
        plt.suptitle(f'Mask Generation Results - {result.threshold_method}', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    # ========================================================
    # VISUALIZATION HELPERS
    # ========================================================
    
    def draw_contours_on_image(
        self,
        image: np.ndarray,
        regions: List[DetectedRegion],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        draw_boxes: bool = True,
        draw_centers: bool = True,
        show_labels: bool = True
    ) -> np.ndarray:
        """
        Draw detected regions on an image.
        
        Parameters
        ----------
        image : np.ndarray
            Input image (will be copied)
        regions : list of DetectedRegion
            Regions to draw
        color : tuple
            BGR color for contours
        thickness : int
            Line thickness
        draw_boxes : bool
            Draw bounding boxes
        draw_centers : bool
            Draw center points
        show_labels : bool
            Show region labels
            
        Returns
        -------
        np.ndarray
            Image with annotations
        """
        result = image.copy()
        
        for i, region in enumerate(regions):
            # Draw contour
            cv2.drawContours(result, [region.contour], 0, color, thickness)
            
            # Draw bounding box
            if draw_boxes:
                bb = region.bounding_box
                cv2.rectangle(
                    result,
                    (bb.x, bb.y),
                    (bb.x + bb.width, bb.y + bb.height),
                    (0, 0, 255), 1
                )
            
            # Draw center
            if draw_centers:
                cx, cy = region.bounding_box.center
                cv2.circle(result, (cx, cy), 3, (255, 0, 0), -1)
            
            # Show label
            if show_labels:
                bb = region.bounding_box
                label = f"#{i+1}: {int(region.area)}px"
                cv2.putText(
                    result, label,
                    (bb.x, bb.y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
                )
        
        return result
    
    def create_mask_overlay(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        color: Tuple[int, int, int] = (0, 0, 255),
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Create overlay of mask on original image.
        
        Parameters
        ----------
        image : np.ndarray
            Original image
        mask : np.ndarray
            Binary mask
        color : tuple
            BGR color for overlay
        alpha : float
            Transparency (0-1)
            
        Returns
        -------
        np.ndarray
            Image with mask overlay
        """
        # Ensure same size
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        
        # Create colored overlay
        overlay = image.copy()
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = color
        
        # Blend
        mask_bool = mask > 0
        overlay[mask_bool] = cv2.addWeighted(
            image[mask_bool], 1 - alpha,
            colored_mask[mask_bool], alpha,
            0
        )
        
        return overlay
    
    def extract_roi(
        self,
        image: np.ndarray,
        region: DetectedRegion,
        padding: int = 10
    ) -> np.ndarray:
        """
        Extract ROI around a detected region.
        
        Parameters
        ----------
        image : np.ndarray
            Input image
        region : DetectedRegion
            Region to extract
        padding : int
            Padding around bounding box
            
        Returns
        -------
        np.ndarray
            Cropped ROI
        """
        bb = region.bounding_box
        h, w = image.shape[:2]
        
        x1 = max(0, bb.x - padding)
        y1 = max(0, bb.y - padding)
        x2 = min(w, bb.x + bb.width + padding)
        y2 = min(h, bb.y + bb.height + padding)
        
        return image[y1:y2, x1:x2].copy()
    
    # ========================================================
    # BATCH PROCESSING
    # ========================================================
    
    def process_batch(
        self,
        difference_maps: List[Union[np.ndarray, str, Path]],
        output_dir: Union[str, Path],
        original_images: Optional[List[np.ndarray]] = None,
        save_visualizations: bool = True
    ) -> List[Dict]:
        """
        Process multiple difference maps.
        
        Parameters
        ----------
        difference_maps : list
            List of difference maps (arrays or paths)
        output_dir : str or Path
            Output directory
        original_images : list, optional
            Corresponding original images
        save_visualizations : bool
            Save visualization images
            
        Returns
        -------
        list of dict
            Results for each map
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        total = len(difference_maps)
        
        print(f"\n{'='*60}")
        print(f"    BATCH MASK GENERATION: {total} images")
        print(f"{'='*60}")
        
        for i, diff_map in enumerate(difference_maps):
            print(f"\n[{i+1}/{total}] Processing...")
            
            # Load if path
            if isinstance(diff_map, (str, Path)):
                diff_path = Path(diff_map)
                diff_map = load_image(diff_path, color_mode='gray')
                name = diff_path.stem
            else:
                name = f"image_{i+1:03d}"
            
            # Get corresponding original if available
            original = original_images[i] if original_images else None
            
            try:
                result = self.process_difference_map(
                    diff_map,
                    original_image=original,
                    output_dir=output_dir,
                    save_visualizations=save_visualizations,
                    output_prefix=name
                )
                
                results.append({
                    'name': name,
                    'status': 'success',
                    'num_regions': result.num_regions,
                    'total_area': result.total_defect_area,
                    'threshold': result.threshold_value,
                    'processing_time': result.processing_time
                })
                
            except Exception as e:
                results.append({
                    'name': name,
                    'status': 'failed',
                    'error': str(e)
                })
                print(f"   ❌ Error: {e}")
        
        # Save summary
        summary = {
            'total_processed': total,
            'successful': sum(1 for r in results if r['status'] == 'success'),
            'failed': sum(1 for r in results if r['status'] == 'failed'),
            'threshold_method': self.threshold_method.value,
            'results': results
        }
        
        with open(output_dir / 'batch_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"    BATCH COMPLETE: {summary['successful']}/{total} successful")
        print(f"{'='*60}")
        
        return results
    
    # ========================================================
    # THRESHOLD COMPARISON
    # ========================================================
    
    @staticmethod
    def compare_thresholds(
        difference_map: np.ndarray,
        methods: List[str] = None,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict:
        """
        Compare different thresholding methods.
        
        Parameters
        ----------
        difference_map : np.ndarray
            Input difference map
        methods : list, optional
            Methods to compare
        output_dir : str or Path, optional
            Output directory
            
        Returns
        -------
        dict
            Comparison results
        """
        if methods is None:
            methods = ['otsu', 'adaptive_gaussian', 'triangle']
        
        print(f"\n{'='*60}")
        print(f"    COMPARING THRESHOLDING METHODS")
        print(f"{'='*60}")
        
        results = {}
        
        for method in methods:
            print(f"\n🔬 Testing {method}...")
            
            generator = MaskGenerator(threshold_method=method, verbose=False)
            result = generator.generate_mask(difference_map)
            
            results[method] = {
                'threshold_value': result.threshold_value,
                'num_regions': result.num_regions,
                'total_area': result.total_defect_area,
                'mask_coverage': result.stats['mask_coverage'],
                'processing_time': result.processing_time
            }
            
            print(f"   Threshold: {result.threshold_value:.0f}" 
                  if result.threshold_value >= 0 else "   Threshold: adaptive")
            print(f"   Regions: {result.num_regions}")
            print(f"   Coverage: {result.stats['mask_coverage']:.4f}%")
        
        # Determine best (most balanced)
        valid_results = {k: v for k, v in results.items() 
                        if v['num_regions'] > 0 and v['mask_coverage'] < 10}
        
        if valid_results:
            best = min(valid_results.keys(), 
                      key=lambda k: abs(valid_results[k]['mask_coverage'] - 1))
        else:
            best = 'otsu'
        
        results['best_method'] = best
        
        print(f"\n🏆 Recommended: {best}")
        
        return results


# ============================================================
# MAIN / CLI
# ============================================================

def main():
    """Command-line interface for mask generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='PCB Mask Generation from Difference Maps'
    )
    
    parser.add_argument(
        'input',
        type=str,
        help='Path to difference map image'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='outputs',
        help='Output directory'
    )
    
    parser.add_argument(
        '-m', '--method',
        type=str,
        choices=['otsu', 'adaptive_mean', 'adaptive_gaussian', 'binary', 'triangle'],
        default='otsu',
        help='Thresholding method'
    )
    
    parser.add_argument(
        '-t', '--threshold',
        type=int,
        default=30,
        help='Manual threshold value (for binary method)'
    )
    
    parser.add_argument(
        '--min-area',
        type=int,
        default=50,
        help='Minimum contour area'
    )
    
    parser.add_argument(
        '-v', '--visualize',
        action='store_true',
        help='Display visualization'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare all thresholding methods'
    )
    
    args = parser.parse_args()
    
    # Load difference map
    diff_map = load_image(args.input, color_mode='gray')
    
    if args.compare:
        results = MaskGenerator.compare_thresholds(
            diff_map,
            output_dir=args.output
        )
    else:
        generator = MaskGenerator(
            threshold_method=args.method,
            threshold_value=args.threshold,
            min_area=args.min_area
        )
        
        result = generator.process_difference_map(
            diff_map,
            output_dir=args.output,
            visualize=args.visualize
        )
        
        print(f"\nFound {result.num_regions} potential defect regions")
    
    return 0


if __name__ == '__main__':
    exit(main())
