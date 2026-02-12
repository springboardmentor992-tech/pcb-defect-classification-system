"""
Contour Detector Module
=======================

Advanced contour detection and analysis for PCB defect regions.
Provides comprehensive tools for finding, filtering, analyzing, and 
visualizing contours in binary defect masks.

Features:
---------
- Multiple contour retrieval modes (external, tree, list)
- Advanced contour filtering (area, perimeter, shape)
- Comprehensive contour property analysis
- Shape feature extraction (circularity, solidity, convexity)
- Contour approximation and smoothing
- Multi-level visualization with annotations
- Batch processing capabilities
- Integration with Module 1 mask outputs

Classes:
--------
- ContourMode: Enum for contour retrieval modes
- ContourProperties: Dataclass for contour measurements
- ContourDetector: Main class for contour detection and analysis

Author: PCB Defect Detection Team
Version: 1.0.0

Usage:
------
>>> from detection.contour_detector import ContourDetector
>>> detector = ContourDetector(min_area=50, max_area=10000)
>>> contours = detector.find_contours(binary_mask)
>>> properties = detector.analyze_contours(contours)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union, Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import warnings

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# ENUMS AND CONSTANTS
# ============================================================

class ContourMode(Enum):
    """Contour retrieval modes."""
    EXTERNAL = cv2.RETR_EXTERNAL      # Only outermost contours
    LIST = cv2.RETR_LIST              # All contours, no hierarchy
    TREE = cv2.RETR_TREE              # Full hierarchy
    CCOMP = cv2.RETR_CCOMP            # Two-level hierarchy


class ContourApproximation(Enum):
    """Contour approximation methods."""
    NONE = cv2.CHAIN_APPROX_NONE      # All points
    SIMPLE = cv2.CHAIN_APPROX_SIMPLE  # Compress horizontal/vertical/diagonal
    TC89_L1 = cv2.CHAIN_APPROX_TC89_L1
    TC89_KCOS = cv2.CHAIN_APPROX_TC89_KCOS


class ShapeType(Enum):
    """Shape classification based on contour properties."""
    CIRCULAR = "circular"        # High circularity (> 0.7)
    RECTANGULAR = "rectangular"  # High solidity, low circularity
    IRREGULAR = "irregular"      # Low solidity
    ELONGATED = "elongated"      # High aspect ratio
    COMPACT = "compact"          # Low aspect ratio, moderate circularity


# Default color palette for visualization
COLORS = {
    'contour': (0, 255, 0),       # Green
    'bbox': (255, 0, 0),          # Blue (BGR)
    'centroid': (0, 0, 255),      # Red
    'hull': (255, 255, 0),        # Cyan
    'text': (255, 255, 255),      # White
    'highlight': (0, 165, 255),   # Orange
}


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class ContourProperties:
    """
    Comprehensive properties of a single contour.
    
    Attributes
    ----------
    contour : np.ndarray
        The raw contour points
    area : float
        Area enclosed by the contour
    perimeter : float
        Arc length of the contour
    bounding_box : Tuple[int, int, int, int]
        (x, y, width, height) of bounding rectangle
    centroid : Tuple[int, int]
        (x, y) center of mass
    circularity : float
        4π × area / perimeter², 1.0 = perfect circle
    solidity : float
        area / convex_hull_area, 1.0 = convex shape
    aspect_ratio : float
        width / height of bounding box
    extent : float
        area / bounding_box_area, fill ratio
    convexity : float
        convex_hull_perimeter / contour_perimeter
    orientation : float
        Angle of the fitted ellipse (degrees)
    convex_hull : np.ndarray
        Points of the convex hull
    moments : Dict
        Raw image moments
    hu_moments : np.ndarray
        Hu moment invariants (7 values)
    fitted_ellipse : Optional[Tuple]
        ((cx, cy), (major, minor), angle) or None
    min_enclosing_circle : Tuple[Tuple[int, int], float]
        ((cx, cy), radius) of minimum enclosing circle
    shape_type : ShapeType
        Classified shape type
    """
    
    contour: np.ndarray
    area: float
    perimeter: float
    bounding_box: Tuple[int, int, int, int]
    centroid: Tuple[int, int]
    circularity: float
    solidity: float
    aspect_ratio: float
    extent: float
    convexity: float
    orientation: float
    convex_hull: np.ndarray
    moments: Dict
    hu_moments: np.ndarray
    fitted_ellipse: Optional[Tuple]
    min_enclosing_circle: Tuple[Tuple[int, int], float]
    shape_type: ShapeType
    index: int = 0
    
    @property
    def x(self) -> int:
        """X coordinate of bounding box."""
        return self.bounding_box[0]
    
    @property
    def y(self) -> int:
        """Y coordinate of bounding box."""
        return self.bounding_box[1]
    
    @property
    def width(self) -> int:
        """Width of bounding box."""
        return self.bounding_box[2]
    
    @property
    def height(self) -> int:
        """Height of bounding box."""
        return self.bounding_box[3]
    
    @property
    def bbox_area(self) -> int:
        """Area of bounding box."""
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        """Center of bounding box."""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def bbox_xyxy(self) -> Tuple[int, int, int, int]:
        """Bounding box as (x1, y1, x2, y2)."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'index': self.index,
            'area': round(self.area, 2),
            'perimeter': round(self.perimeter, 2),
            'bounding_box': {
                'x': self.x,
                'y': self.y,
                'width': self.width,
                'height': self.height
            },
            'centroid': {'x': self.centroid[0], 'y': self.centroid[1]},
            'circularity': round(self.circularity, 4),
            'solidity': round(self.solidity, 4),
            'aspect_ratio': round(self.aspect_ratio, 4),
            'extent': round(self.extent, 4),
            'convexity': round(self.convexity, 4),
            'orientation': round(self.orientation, 2),
            'shape_type': self.shape_type.value,
            'hu_moments': self.hu_moments.tolist() if self.hu_moments is not None else None
        }
    
    def __repr__(self) -> str:
        return (f"ContourProperties(index={self.index}, area={self.area:.1f}, "
                f"shape={self.shape_type.value}, bbox={self.bounding_box})")


@dataclass
class DetectionResult:
    """
    Results from contour detection on a single image.
    
    Attributes
    ----------
    contours : List[np.ndarray]
        Raw contour arrays
    properties : List[ContourProperties]
        Analyzed properties for each contour
    hierarchy : np.ndarray
        Contour hierarchy (if applicable)
    image_shape : Tuple[int, int]
        (height, width) of source image
    processing_time : float
        Time taken for detection
    detection_params : Dict
        Parameters used for detection
    """
    
    contours: List[np.ndarray]
    properties: List[ContourProperties]
    hierarchy: Optional[np.ndarray]
    image_shape: Tuple[int, int]
    processing_time: float
    detection_params: Dict = field(default_factory=dict)
    
    @property
    def num_contours(self) -> int:
        return len(self.contours)
    
    @property
    def total_area(self) -> float:
        return sum(p.area for p in self.properties)
    
    @property
    def coverage_percent(self) -> float:
        """Percentage of image covered by contours."""
        img_area = self.image_shape[0] * self.image_shape[1]
        return (self.total_area / img_area) * 100 if img_area > 0 else 0
    
    def get_largest(self, n: int = 1) -> List[ContourProperties]:
        """Get n largest contours by area."""
        sorted_props = sorted(self.properties, key=lambda p: p.area, reverse=True)
        return sorted_props[:n]
    
    def filter_by_shape(self, shape_type: ShapeType) -> List[ContourProperties]:
        """Filter contours by shape type."""
        return [p for p in self.properties if p.shape_type == shape_type]
    
    def to_dict(self) -> Dict:
        return {
            'num_contours': self.num_contours,
            'total_area': round(self.total_area, 2),
            'coverage_percent': round(self.coverage_percent, 4),
            'image_shape': {'height': self.image_shape[0], 'width': self.image_shape[1]},
            'processing_time': round(self.processing_time, 4),
            'detection_params': self.detection_params,
            'contours': [p.to_dict() for p in self.properties]
        }


# ============================================================
# MAIN CONTOUR DETECTOR CLASS
# ============================================================

class ContourDetector:
    """
    Advanced contour detection and analysis for PCB defect regions.
    
    This class provides comprehensive tools for detecting, filtering,
    analyzing, and visualizing contours in binary defect masks.
    
    Attributes
    ----------
    min_area : int
        Minimum contour area to keep (filters noise)
    max_area : int
        Maximum contour area to keep (filters large regions)
    mode : ContourMode
        Contour retrieval mode
    approximation : ContourApproximation
        Contour approximation method
    
    Examples
    --------
    >>> detector = ContourDetector(min_area=50, max_area=10000)
    >>> result = detector.detect(binary_mask)
    >>> print(f"Found {result.num_contours} contours")
    >>> 
    >>> # With filtering
    >>> detector.set_filters(min_circularity=0.5)
    >>> circular_result = detector.detect(mask)
    """
    
    # Default parameters
    DEFAULT_MIN_AREA = 50
    DEFAULT_MAX_AREA = 50000
    
    def __init__(
        self,
        min_area: int = DEFAULT_MIN_AREA,
        max_area: int = DEFAULT_MAX_AREA,
        mode: Union[str, ContourMode] = ContourMode.EXTERNAL,
        approximation: Union[str, ContourApproximation] = ContourApproximation.SIMPLE,
        verbose: bool = True
    ):
        """
        Initialize the ContourDetector.
        
        Parameters
        ----------
        min_area : int
            Minimum contour area to keep (filters noise)
        max_area : int
            Maximum contour area to keep (filters large false positives)
        mode : ContourMode or str
            Contour retrieval mode: 'external', 'list', 'tree', 'ccomp'
        approximation : ContourApproximation or str
            Contour approximation: 'none', 'simple', 'tc89_l1', 'tc89_kcos'
        verbose : bool
            Print progress messages
        """
        self.min_area = min_area
        self.max_area = max_area
        
        # Parse mode
        if isinstance(mode, str):
            mode = ContourMode[mode.upper()]
        self.mode = mode
        
        # Parse approximation
        if isinstance(approximation, str):
            approximation = ContourApproximation[approximation.upper()]
        self.approximation = approximation
        
        self.verbose = verbose
        
        # Advanced filters (optional)
        self._filters = {
            'min_circularity': None,
            'max_circularity': None,
            'min_solidity': None,
            'min_aspect_ratio': None,
            'max_aspect_ratio': None,
            'min_extent': None
        }
        
        if self.verbose:
            print(f"✓ ContourDetector initialized")
            print(f"   Area range: [{min_area}, {max_area}]")
            print(f"   Mode: {self.mode.name}")
    
    # ========================================================
    # FILTER CONFIGURATION
    # ========================================================
    
    def set_filters(
        self,
        min_circularity: float = None,
        max_circularity: float = None,
        min_solidity: float = None,
        min_aspect_ratio: float = None,
        max_aspect_ratio: float = None,
        min_extent: float = None
    ) -> 'ContourDetector':
        """
        Set advanced shape-based filters.
        
        Parameters
        ----------
        min_circularity : float, optional
            Minimum circularity (0-1)
        max_circularity : float, optional
            Maximum circularity (0-1)
        min_solidity : float, optional
            Minimum solidity (0-1)
        min_aspect_ratio : float, optional
            Minimum width/height ratio
        max_aspect_ratio : float, optional
            Maximum width/height ratio
        min_extent : float, optional
            Minimum extent (area/bbox_area)
            
        Returns
        -------
        ContourDetector
            Self for method chaining
        """
        self._filters.update({
            'min_circularity': min_circularity,
            'max_circularity': max_circularity,
            'min_solidity': min_solidity,
            'min_aspect_ratio': min_aspect_ratio,
            'max_aspect_ratio': max_aspect_ratio,
            'min_extent': min_extent
        })
        return self
    
    def clear_filters(self) -> 'ContourDetector':
        """Clear all advanced filters."""
        self._filters = {k: None for k in self._filters}
        return self
    
    # ========================================================
    # CORE DETECTION
    # ========================================================
    
    def find_contours(
        self,
        binary_mask: np.ndarray,
        apply_filters: bool = True
    ) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
        """
        Find contours in a binary mask.
        
        Parameters
        ----------
        binary_mask : np.ndarray
            Binary mask image (0 or 255)
        apply_filters : bool
            Whether to apply area and shape filters
            
        Returns
        -------
        tuple
            (list of contours, hierarchy array)
        """
        # Ensure proper format
        if len(binary_mask.shape) == 3:
            binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
        
        # Ensure binary
        if binary_mask.max() <= 1:
            binary_mask = (binary_mask * 255).astype(np.uint8)
        
        # Find contours
        contours, hierarchy = cv2.findContours(
            binary_mask.astype(np.uint8),
            self.mode.value,
            self.approximation.value
        )
        
        if not apply_filters:
            return list(contours), hierarchy
        
        # Apply area filter
        filtered = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            if self.max_area and area > self.max_area:
                continue
            filtered.append(contour)
        
        return filtered, hierarchy
    
    def analyze_contour(
        self,
        contour: np.ndarray,
        index: int = 0
    ) -> ContourProperties:
        """
        Analyze a single contour and extract all properties.
        
        Parameters
        ----------
        contour : np.ndarray
            Single contour array
        index : int
            Index for labeling
            
        Returns
        -------
        ContourProperties
            Comprehensive contour properties
        """
        # Area and perimeter
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Moments
        moments = cv2.moments(contour)
        
        # Centroid
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
        else:
            cx, cy = x + w // 2, y + h // 2
        
        # Convex hull
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        hull_perimeter = cv2.arcLength(hull, True)
        
        # Shape metrics
        circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-8)
        solidity = area / (hull_area + 1e-8)
        aspect_ratio = w / (h + 1e-8)
        extent = area / ((w * h) + 1e-8)
        convexity = hull_perimeter / (perimeter + 1e-8)
        
        # Hu moments
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Fitted ellipse (needs at least 5 points)
        fitted_ellipse = None
        orientation = 0.0
        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                fitted_ellipse = ellipse
                orientation = ellipse[2]  # Angle in degrees
            except cv2.error:
                pass
        
        # Minimum enclosing circle
        (enc_cx, enc_cy), enc_radius = cv2.minEnclosingCircle(contour)
        min_enclosing_circle = ((int(enc_cx), int(enc_cy)), enc_radius)
        
        # Classify shape
        shape_type = self._classify_shape(circularity, solidity, aspect_ratio, extent)
        
        return ContourProperties(
            contour=contour,
            area=area,
            perimeter=perimeter,
            bounding_box=(x, y, w, h),
            centroid=(cx, cy),
            circularity=circularity,
            solidity=solidity,
            aspect_ratio=aspect_ratio,
            extent=extent,
            convexity=convexity,
            orientation=orientation,
            convex_hull=hull,
            moments=moments,
            hu_moments=hu_moments,
            fitted_ellipse=fitted_ellipse,
            min_enclosing_circle=min_enclosing_circle,
            shape_type=shape_type,
            index=index
        )
    
    def _classify_shape(
        self,
        circularity: float,
        solidity: float,
        aspect_ratio: float,
        extent: float
    ) -> ShapeType:
        """Classify shape based on metrics."""
        if circularity > 0.7 and solidity > 0.85:
            return ShapeType.CIRCULAR
        elif aspect_ratio > 3.0 or aspect_ratio < 0.33:
            return ShapeType.ELONGATED
        elif solidity < 0.6:
            return ShapeType.IRREGULAR
        elif solidity > 0.85 and 0.7 < aspect_ratio < 1.4:
            return ShapeType.RECTANGULAR
        else:
            return ShapeType.COMPACT
    
    def _apply_shape_filters(
        self,
        props: ContourProperties
    ) -> bool:
        """Check if contour passes shape filters."""
        f = self._filters
        
        if f['min_circularity'] and props.circularity < f['min_circularity']:
            return False
        if f['max_circularity'] and props.circularity > f['max_circularity']:
            return False
        if f['min_solidity'] and props.solidity < f['min_solidity']:
            return False
        if f['min_aspect_ratio'] and props.aspect_ratio < f['min_aspect_ratio']:
            return False
        if f['max_aspect_ratio'] and props.aspect_ratio > f['max_aspect_ratio']:
            return False
        if f['min_extent'] and props.extent < f['min_extent']:
            return False
        
        return True
    
    # ========================================================
    # MAIN DETECTION PIPELINE
    # ========================================================
    
    def detect(
        self,
        binary_mask: np.ndarray,
        analyze: bool = True
    ) -> DetectionResult:
        """
        Detect and analyze contours in a binary mask.
        
        This is the main entry point for contour detection.
        
        Parameters
        ----------
        binary_mask : np.ndarray
            Binary mask image
        analyze : bool
            Whether to compute full properties for each contour
            
        Returns
        -------
        DetectionResult
            Complete detection results
        """
        start_time = time.time()
        
        if self.verbose:
            print(f"\n🔍 Detecting contours...")
        
        # Find contours
        contours, hierarchy = self.find_contours(binary_mask, apply_filters=True)
        
        if self.verbose:
            print(f"   Found {len(contours)} contours (after area filter)")
        
        # Analyze each contour
        all_properties = []
        
        if analyze:
            for idx, contour in enumerate(contours):
                props = self.analyze_contour(contour, index=idx + 1)
                
                # Apply shape filters
                if self._apply_shape_filters(props):
                    all_properties.append(props)
            
            if self.verbose and len(all_properties) < len(contours):
                print(f"   After shape filters: {len(all_properties)} contours")
        
        elapsed = time.time() - start_time
        
        # Collect filtered contours based on properties
        filtered_contours = [p.contour for p in all_properties] if analyze else contours
        
        result = DetectionResult(
            contours=filtered_contours,
            properties=all_properties,
            hierarchy=hierarchy,
            image_shape=binary_mask.shape[:2],
            processing_time=elapsed,
            detection_params={
                'min_area': self.min_area,
                'max_area': self.max_area,
                'mode': self.mode.name,
                'filters': {k: v for k, v in self._filters.items() if v is not None}
            }
        )
        
        if self.verbose:
            print(f"   ✓ Detection complete in {elapsed:.3f}s")
            if all_properties:
                print(f"   Total contour area: {result.total_area:.0f} px")
                print(f"   Image coverage: {result.coverage_percent:.4f}%")
        
        return result
    
    # ========================================================
    # VISUALIZATION
    # ========================================================
    
    def visualize_contours(
        self,
        image: np.ndarray,
        result: DetectionResult,
        draw_contours: bool = True,
        draw_bboxes: bool = True,
        draw_centroids: bool = True,
        draw_hulls: bool = False,
        draw_labels: bool = True,
        contour_color: Tuple[int, int, int] = COLORS['contour'],
        bbox_color: Tuple[int, int, int] = COLORS['bbox'],
        contour_thickness: int = 2,
        bbox_thickness: int = 1
    ) -> np.ndarray:
        """
        Visualize detected contours on an image.
        
        Parameters
        ----------
        image : np.ndarray
            Image to draw on (will be copied)
        result : DetectionResult
            Detection results
        draw_contours : bool
            Draw contour outlines
        draw_bboxes : bool
            Draw bounding boxes
        draw_centroids : bool
            Draw centroid points
        draw_hulls : bool
            Draw convex hulls
        draw_labels : bool
            Draw labels with properties
        contour_color : tuple
            BGR color for contours
        bbox_color : tuple
            BGR color for bounding boxes
        contour_thickness : int
            Contour line thickness
        bbox_thickness : int
            Bounding box line thickness
            
        Returns
        -------
        np.ndarray
            Annotated image
        """
        # Convert to BGR if grayscale
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()
        
        for props in result.properties:
            # Draw contour
            if draw_contours:
                cv2.drawContours(vis, [props.contour], 0, contour_color, contour_thickness)
            
            # Draw convex hull
            if draw_hulls:
                cv2.drawContours(vis, [props.convex_hull], 0, COLORS['hull'], 1)
            
            # Draw bounding box
            if draw_bboxes:
                x, y, w, h = props.bounding_box
                cv2.rectangle(vis, (x, y), (x + w, y + h), bbox_color, bbox_thickness)
            
            # Draw centroid
            if draw_centroids:
                cx, cy = props.centroid
                cv2.circle(vis, (cx, cy), 4, COLORS['centroid'], -1)
            
            # Draw label
            if draw_labels:
                x, y, w, h = props.bounding_box
                label = f"#{props.index} A:{int(props.area)}"
                
                # Background for text
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                )
                cv2.rectangle(
                    vis, 
                    (x, y - text_h - 6), 
                    (x + text_w + 4, y), 
                    (0, 0, 0), 
                    -1
                )
                cv2.putText(
                    vis, label,
                    (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    COLORS['text'], 1
                )
        
        return vis
    
    def create_analysis_visualization(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        result: DetectionResult,
        figsize: Tuple[int, int] = (18, 12)
    ) -> 'matplotlib.figure.Figure':
        """
        Create comprehensive analysis visualization.
        
        Parameters
        ----------
        image : np.ndarray
            Original image
        mask : np.ndarray
            Binary mask
        result : DetectionResult
            Detection results
        figsize : tuple
            Figure size
            
        Returns
        -------
        matplotlib.figure.Figure
            Analysis figure
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        # Convert images
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # 1. Original image
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('Original Image', fontweight='bold')
        axes[0, 0].axis('off')
        
        # 2. Binary mask
        axes[0, 1].imshow(mask, cmap='gray')
        axes[0, 1].set_title('Binary Mask', fontweight='bold')
        axes[0, 1].axis('off')
        
        # 3. Detected contours
        contour_vis = self.visualize_contours(
            image_rgb.copy(), result,
            draw_bboxes=True, draw_labels=True
        )
        axes[0, 2].imshow(cv2.cvtColor(contour_vis, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title(f'Detected Contours ({result.num_contours})', fontweight='bold')
        axes[0, 2].axis('off')
        
        # 4. Shape distribution
        if result.properties:
            shape_counts = {}
            for p in result.properties:
                shape = p.shape_type.value
                shape_counts[shape] = shape_counts.get(shape, 0) + 1
            
            shapes = list(shape_counts.keys())
            counts = list(shape_counts.values())
            colors_bar = plt.cm.Set2(np.linspace(0, 1, len(shapes)))
            
            axes[1, 0].bar(shapes, counts, color=colors_bar)
            axes[1, 0].set_title('Shape Distribution', fontweight='bold')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].tick_params(axis='x', rotation=45)
        else:
            axes[1, 0].text(0.5, 0.5, 'No contours detected', 
                           ha='center', va='center')
            axes[1, 0].set_title('Shape Distribution', fontweight='bold')
        
        # 5. Area histogram
        if result.properties:
            areas = [p.area for p in result.properties]
            axes[1, 1].hist(areas, bins=min(20, len(areas)), color='steelblue', 
                           edgecolor='white', alpha=0.7)
            axes[1, 1].axvline(np.mean(areas), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(areas):.0f}')
            axes[1, 1].set_title('Area Distribution', fontweight='bold')
            axes[1, 1].set_xlabel('Area (px)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'No contours detected', 
                           ha='center', va='center')
            axes[1, 1].set_title('Area Distribution', fontweight='bold')
        
        # 6. Summary statistics
        axes[1, 2].axis('off')
        summary_lines = [
            'DETECTION SUMMARY',
            '─' * 30,
            f'Contours found: {result.num_contours}',
            f'Total area: {result.total_area:.0f} px',
            f'Coverage: {result.coverage_percent:.4f}%',
            f'Processing time: {result.processing_time:.3f}s',
            '',
            'PARAMETERS',
            '─' * 30,
            f'Min area: {self.min_area}',
            f'Max area: {self.max_area}',
            f'Mode: {self.mode.name}',
        ]
        
        if result.properties:
            summary_lines.extend([
                '',
                'CONTOUR STATS',
                '─' * 30,
                f'Min area: {min(p.area for p in result.properties):.0f}',
                f'Max area: {max(p.area for p in result.properties):.0f}',
                f'Avg circularity: {np.mean([p.circularity for p in result.properties]):.3f}',
                f'Avg solidity: {np.mean([p.solidity for p in result.properties]):.3f}',
            ])
        
        axes[1, 2].text(
            0.1, 0.95, '\n'.join(summary_lines),
            transform=axes[1, 2].transAxes,
            fontfamily='monospace',
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8)
        )
        
        plt.suptitle('Contour Detection Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    # ========================================================
    # UTILITY METHODS
    # ========================================================
    
    def print_contour_table(self, result: DetectionResult) -> None:
        """Print formatted table of contour properties."""
        print(f"\n{'='*90}")
        print("CONTOUR ANALYSIS")
        print(f"{'='*90}")
        print(f"{'#':<4} {'Area':<10} {'Perim':<10} {'BBox (x,y,w,h)':<22} "
              f"{'Circ':<8} {'Solid':<8} {'Shape':<12}")
        print(f"{'-'*90}")
        
        for p in result.properties:
            bbox_str = f"({p.x},{p.y},{p.width},{p.height})"
            print(f"{p.index:<4} {p.area:<10.1f} {p.perimeter:<10.1f} {bbox_str:<22} "
                  f"{p.circularity:<8.3f} {p.solidity:<8.3f} {p.shape_type.value:<12}")
        
        print(f"{'='*90}")
        print(f"Total: {result.num_contours} contours, {result.total_area:.0f} px total area")
    
    def get_contour_centers(self, result: DetectionResult) -> np.ndarray:
        """Get array of contour centroids."""
        return np.array([p.centroid for p in result.properties])
    
    def get_bounding_boxes(self, result: DetectionResult) -> List[Tuple[int, int, int, int]]:
        """Get list of bounding boxes."""
        return [p.bounding_box for p in result.properties]
    
    def expand_bounding_box(
        self,
        bbox: Tuple[int, int, int, int],
        expand_pixels: int = 10,
        image_shape: Tuple[int, int] = None
    ) -> Tuple[int, int, int, int]:
        """
        Expand bounding box by fixed pixels.
        
        Parameters
        ----------
        bbox : tuple
            (x, y, w, h) bounding box
        expand_pixels : int
            Pixels to add on each side
        image_shape : tuple, optional
            (height, width) to clip to boundaries
            
        Returns
        -------
        tuple
            Expanded bounding box
        """
        x, y, w, h = bbox
        
        x_new = max(0, x - expand_pixels)
        y_new = max(0, y - expand_pixels)
        w_new = w + 2 * expand_pixels
        h_new = h + 2 * expand_pixels
        
        if image_shape:
            img_h, img_w = image_shape
            w_new = min(w_new, img_w - x_new)
            h_new = min(h_new, img_h - y_new)
        
        return (x_new, y_new, w_new, h_new)
    
    def save_results(
        self,
        result: DetectionResult,
        output_path: Union[str, Path]
    ) -> None:
        """Save detection results to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        if self.verbose:
            print(f"   ✓ Results saved to: {output_path}")


# ============================================================
# MAIN / CLI
# ============================================================

def main():
    """Command-line interface for contour detection."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='PCB Defect Contour Detection'
    )
    
    parser.add_argument(
        'mask_path',
        type=str,
        help='Path to binary mask image'
    )
    
    parser.add_argument(
        '-i', '--image',
        type=str,
        help='Path to original image for visualization'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='outputs/contours',
        help='Output directory'
    )
    
    parser.add_argument(
        '--min-area',
        type=int,
        default=50,
        help='Minimum contour area'
    )
    
    parser.add_argument(
        '--max-area',
        type=int,
        default=10000,
        help='Maximum contour area'
    )
    
    parser.add_argument(
        '-v', '--visualize',
        action='store_true',
        help='Show visualization'
    )
    
    args = parser.parse_args()
    
    # Load mask
    mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Could not load mask from {args.mask_path}")
        return 1
    
    # Load image if provided
    image = None
    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            print(f"Warning: Could not load image from {args.image}")
            image = mask
    else:
        image = mask
    
    # Detect
    detector = ContourDetector(
        min_area=args.min_area,
        max_area=args.max_area,
        verbose=True
    )
    
    result = detector.detect(mask)
    
    # Print table
    detector.print_contour_table(result)
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    detector.save_results(result, output_dir / 'detection_results.json')
    
    # Visualization
    vis = detector.visualize_contours(image, result)
    cv2.imwrite(str(output_dir / 'contours_visualization.png'), vis)
    
    if args.visualize:
        import matplotlib.pyplot as plt
        fig = detector.create_analysis_visualization(image, mask, result)
        plt.show()
    
    return 0


if __name__ == '__main__':
    exit(main())
