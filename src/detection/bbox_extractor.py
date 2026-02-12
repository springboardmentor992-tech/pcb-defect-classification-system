"""
Bounding Box Extractor Module
=============================

Advanced bounding box extraction and manipulation for PCB defect regions.
Provides comprehensive tools for extracting, formatting, expanding, merging,
and visualizing bounding boxes around detected defects.

Features:
---------
- Multiple bounding box formats (xywh, xyxy, cxcywh)
- Box expansion with configurable padding
- IoU (Intersection over Union) calculations
- Non-Maximum Suppression (NMS) for overlapping boxes
- Merging nearby/overlapping boxes
- Rich visualization with labels and colors
- Batch processing capabilities
- Integration with ContourDetector

Classes:
--------
- BBoxFormat: Enum for bounding box formats
- BoundingBox: Dataclass representing a single bounding box
- BoundingBoxExtractor: Main class for bbox extraction and manipulation

Author: PCB Defect Detection Team
Version: 1.0.0

Usage:
------
>>> from detection.bbox_extractor import BoundingBoxExtractor
>>> extractor = BoundingBoxExtractor(expand_pixels=10)
>>> boxes = extractor.extract_from_contours(contours, image_shape)
>>> vis = extractor.visualize(image, boxes)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union, Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# ENUMS AND CONSTANTS
# ============================================================

class BBoxFormat(Enum):
    """Bounding box format types."""
    XYWH = "xywh"           # (x, y, width, height)
    XYXY = "xyxy"           # (x1, y1, x2, y2)
    CXCYWH = "cxcywh"       # (center_x, center_y, width, height)
    

class ExpansionMode(Enum):
    """Bounding box expansion modes."""
    FIXED = "fixed"         # Fixed pixel amount
    PERCENT = "percent"     # Percentage of box size
    ASPECT = "aspect"       # Maintain aspect ratio


# Default visualization colors (BGR format)
BBOX_COLORS = {
    'default': (0, 255, 0),      # Green
    'Missing_hole': (255, 0, 0),  # Blue
    'Mouse_bite': (0, 0, 255),    # Red
    'Open_circuit': (255, 255, 0), # Cyan
    'Short': (0, 255, 255),       # Yellow
    'Spur': (255, 0, 255),        # Magenta
    'Spurious_copper': (128, 0, 128),  # Purple
}


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class BoundingBox:
    """
    Represents a single bounding box with comprehensive properties.
    
    Attributes
    ----------
    x : int
        Left edge x-coordinate
    y : int
        Top edge y-coordinate
    width : int
        Box width
    height : int
        Box height
    label : str
        Optional label/class name
    confidence : float
        Optional confidence score (0-1)
    index : int
        Box index for tracking
    contour_index : int
        Index of source contour (-1 if not from contour)
    area : float
        Area of the original contour (if available)
    expanded : bool
        Whether box has been expanded
    """
    
    x: int
    y: int
    width: int
    height: int
    label: str = ""
    confidence: float = 1.0
    index: int = 0
    contour_index: int = -1
    area: float = 0.0
    expanded: bool = False
    
    @property
    def x1(self) -> int:
        """Left edge (same as x)."""
        return self.x
    
    @property
    def y1(self) -> int:
        """Top edge (same as y)."""
        return self.y
    
    @property
    def x2(self) -> int:
        """Right edge."""
        return self.x + self.width
    
    @property
    def y2(self) -> int:
        """Bottom edge."""
        return self.y + self.height
    
    @property
    def center_x(self) -> int:
        """Center x-coordinate."""
        return self.x + self.width // 2
    
    @property
    def center_y(self) -> int:
        """Center y-coordinate."""
        return self.y + self.height // 2
    
    @property
    def center(self) -> Tuple[int, int]:
        """Center point (x, y)."""
        return (self.center_x, self.center_y)
    
    @property 
    def box_area(self) -> int:
        """Area of bounding box."""
        return self.width * self.height
    
    @property
    def aspect_ratio(self) -> float:
        """Width / Height ratio."""
        return self.width / max(self.height, 1)
    
    def to_xywh(self) -> Tuple[int, int, int, int]:
        """Return as (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)
    
    def to_xyxy(self) -> Tuple[int, int, int, int]:
        """Return as (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)
    
    def to_cxcywh(self) -> Tuple[int, int, int, int]:
        """Return as (center_x, center_y, width, height)."""
        return (self.center_x, self.center_y, self.width, self.height)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'index': self.index,
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'x2': self.x2,
            'y2': self.y2,
            'center': {'x': self.center_x, 'y': self.center_y},
            'box_area': self.box_area,
            'aspect_ratio': round(self.aspect_ratio, 3),
            'label': self.label,
            'confidence': round(self.confidence, 3),
            'contour_area': round(self.area, 1),
            'expanded': self.expanded
        }
    
    def contains_point(self, x: int, y: int) -> bool:
        """Check if box contains a point."""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2
    
    def iou(self, other: 'BoundingBox') -> float:
        """Calculate Intersection over Union with another box."""
        # Calculate intersection
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = self.box_area + other.box_area - intersection
        
        return intersection / max(union, 1e-8)
    
    def overlap_ratio(self, other: 'BoundingBox') -> float:
        """Calculate overlap ratio (intersection / smaller box area)."""
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        smaller_area = min(self.box_area, other.box_area)
        
        return intersection / max(smaller_area, 1e-8)
    
    def distance_to(self, other: 'BoundingBox') -> float:
        """Calculate center-to-center distance to another box."""
        dx = self.center_x - other.center_x
        dy = self.center_y - other.center_y
        return np.sqrt(dx**2 + dy**2)
    
    def __repr__(self) -> str:
        return (f"BoundingBox(x={self.x}, y={self.y}, "
                f"w={self.width}, h={self.height}, label='{self.label}')")


@dataclass
class ExtractionResult:
    """
    Results from bounding box extraction.
    
    Attributes
    ----------
    boxes : List[BoundingBox]
        Extracted bounding boxes
    image_shape : Tuple[int, int]
        (height, width) of source image
    processing_time : float
        Time taken for extraction
    params : Dict
        Parameters used for extraction
    """
    
    boxes: List[BoundingBox]
    image_shape: Tuple[int, int]
    processing_time: float
    params: Dict = field(default_factory=dict)
    
    @property
    def num_boxes(self) -> int:
        return len(self.boxes)
    
    @property
    def total_area(self) -> int:
        return sum(b.box_area for b in self.boxes)
    
    @property
    def coverage_percent(self) -> float:
        img_area = self.image_shape[0] * self.image_shape[1]
        return (self.total_area / img_area) * 100 if img_area > 0 else 0
    
    def get_by_label(self, label: str) -> List[BoundingBox]:
        """Get boxes with specific label."""
        return [b for b in self.boxes if b.label == label]
    
    def get_largest(self, n: int = 1) -> List[BoundingBox]:
        """Get n largest boxes by area."""
        sorted_boxes = sorted(self.boxes, key=lambda b: b.box_area, reverse=True)
        return sorted_boxes[:n]
    
    def to_dict(self) -> Dict:
        return {
            'num_boxes': self.num_boxes,
            'total_area': self.total_area,
            'coverage_percent': round(self.coverage_percent, 4),
            'image_shape': {'height': self.image_shape[0], 'width': self.image_shape[1]},
            'processing_time': round(self.processing_time, 4),
            'params': self.params,
            'boxes': [b.to_dict() for b in self.boxes]
        }


# ============================================================
# MAIN BOUNDING BOX EXTRACTOR CLASS
# ============================================================

class BoundingBoxExtractor:
    """
    Advanced bounding box extraction and manipulation for PCB defects.
    
    This class provides comprehensive tools for extracting bounding boxes
    from contours, manipulating box coordinates, and visualizing results.
    
    Attributes
    ----------
    expand_pixels : int
        Fixed pixel expansion on each side (default: 10)
    expand_percent : float
        Percentage expansion relative to box size (default: 0)
    min_size : Tuple[int, int]
        Minimum box size (width, height) after extraction
    image_shape : Tuple[int, int]
        Reference image shape for clipping
    
    Examples
    --------
    >>> extractor = BoundingBoxExtractor(expand_pixels=15)
    >>> result = extractor.extract_from_detection(detection_result)
    >>> vis = extractor.visualize(image, result)
    """
    
    def __init__(
        self,
        expand_pixels: int = 10,
        expand_percent: float = 0.0,
        min_size: Tuple[int, int] = (16, 16),
        verbose: bool = True
    ):
        """
        Initialize the BoundingBoxExtractor.
        
        Parameters
        ----------
        expand_pixels : int
            Fixed pixels to add on each side of box
        expand_percent : float  
            Percentage of box size to add (0.1 = 10%)
        min_size : tuple
            Minimum (width, height) for extracted boxes
        verbose : bool
            Print progress messages
        """
        self.expand_pixels = expand_pixels
        self.expand_percent = expand_percent
        self.min_size = min_size
        self.verbose = verbose
        self.image_shape = None
        
        if self.verbose:
            print(f"✓ BoundingBoxExtractor initialized")
            print(f"   Expansion: {expand_pixels}px fixed, {expand_percent*100:.0f}% relative")
            print(f"   Min size: {min_size}")
    
    # ========================================================
    # EXTRACTION METHODS
    # ========================================================
    
    def extract_from_contour(
        self,
        contour: np.ndarray,
        index: int = 0,
        label: str = "",
        contour_area: float = 0.0,
        expand: bool = True
    ) -> BoundingBox:
        """
        Extract bounding box from a single contour.
        
        Parameters
        ----------
        contour : np.ndarray
            OpenCV contour array
        index : int
            Box index
        label : str
            Label/class for the box
        contour_area : float
            Pre-computed contour area
        expand : bool
            Whether to apply expansion
            
        Returns
        -------
        BoundingBox
            Extracted bounding box
        """
        x, y, w, h = cv2.boundingRect(contour)
        
        if contour_area == 0:
            contour_area = cv2.contourArea(contour)
        
        box = BoundingBox(
            x=x, y=y, width=w, height=h,
            label=label, index=index,
            contour_index=index,
            area=contour_area,
            expanded=False
        )
        
        if expand:
            box = self.expand_box(box)
        
        return box
    
    def extract_from_contours(
        self,
        contours: List[np.ndarray],
        image_shape: Tuple[int, int] = None,
        labels: List[str] = None,
        expand: bool = True
    ) -> ExtractionResult:
        """
        Extract bounding boxes from multiple contours.
        
        Parameters
        ----------
        contours : list
            List of OpenCV contour arrays
        image_shape : tuple
            (height, width) of image for clipping
        labels : list
            Optional labels for each contour
        expand : bool
            Whether to apply expansion
            
        Returns
        -------
        ExtractionResult
            Extraction results with all boxes
        """
        start_time = time.time()
        
        if image_shape:
            self.image_shape = image_shape
        
        boxes = []
        for i, contour in enumerate(contours):
            label = labels[i] if labels and i < len(labels) else ""
            box = self.extract_from_contour(
                contour, index=i+1, label=label, expand=expand
            )
            boxes.append(box)
        
        elapsed = time.time() - start_time
        
        img_shape = image_shape if image_shape else (0, 0)
        
        result = ExtractionResult(
            boxes=boxes,
            image_shape=img_shape,
            processing_time=elapsed,
            params={
                'expand_pixels': self.expand_pixels,
                'expand_percent': self.expand_percent,
                'min_size': self.min_size,
                'expand_enabled': expand
            }
        )
        
        if self.verbose:
            print(f"✓ Extracted {len(boxes)} bounding boxes in {elapsed:.3f}s")
        
        return result
    
    def extract_from_detection(
        self,
        detection_result: 'DetectionResult',
        expand: bool = True
    ) -> ExtractionResult:
        """
        Extract bounding boxes from ContourDetector results.
        
        Parameters
        ----------
        detection_result : DetectionResult
            Output from ContourDetector.detect()
        expand : bool
            Whether to apply expansion
            
        Returns
        -------
        ExtractionResult
            Extraction results with all boxes
        """
        start_time = time.time()
        
        self.image_shape = detection_result.image_shape
        
        boxes = []
        for prop in detection_result.properties:
            x, y, w, h = prop.bounding_box
            
            box = BoundingBox(
                x=x, y=y, width=w, height=h,
                label=prop.shape_type.value,
                index=prop.index,
                contour_index=prop.index,
                area=prop.area,
                expanded=False
            )
            
            if expand:
                box = self.expand_box(box)
            
            boxes.append(box)
        
        elapsed = time.time() - start_time
        
        result = ExtractionResult(
            boxes=boxes,
            image_shape=detection_result.image_shape,
            processing_time=elapsed,
            params={
                'expand_pixels': self.expand_pixels,
                'expand_percent': self.expand_percent,
                'source': 'ContourDetector'
            }
        )
        
        if self.verbose:
            print(f"✓ Extracted {len(boxes)} bounding boxes from detection result")
        
        return result
    
    # ========================================================
    # BOX MANIPULATION
    # ========================================================
    
    def expand_box(
        self,
        box: BoundingBox,
        pixels: int = None,
        percent: float = None
    ) -> BoundingBox:
        """
        Expand a bounding box by adding padding.
        
        Parameters
        ----------
        box : BoundingBox
            Box to expand
        pixels : int, optional
            Fixed pixels to add (uses instance default if None)
        percent : float, optional
            Percentage to add (uses instance default if None)
            
        Returns
        -------
        BoundingBox
            Expanded box
        """
        if pixels is None:
            pixels = self.expand_pixels
        if percent is None:
            percent = self.expand_percent
        
        # Calculate expansion amounts
        expand_w = pixels + int(box.width * percent)
        expand_h = pixels + int(box.height * percent)
        
        # Calculate new coordinates
        new_x = box.x - expand_w
        new_y = box.y - expand_h
        new_w = box.width + 2 * expand_w
        new_h = box.height + 2 * expand_h
        
        # Clip to image boundaries if available
        if self.image_shape:
            img_h, img_w = self.image_shape
            new_x = max(0, new_x)
            new_y = max(0, new_y)
            new_w = min(new_w, img_w - new_x)
            new_h = min(new_h, img_h - new_y)
        
        # Enforce minimum size
        new_w = max(new_w, self.min_size[0])
        new_h = max(new_h, self.min_size[1])
        
        return BoundingBox(
            x=new_x, y=new_y, width=new_w, height=new_h,
            label=box.label,
            confidence=box.confidence,
            index=box.index,
            contour_index=box.contour_index,
            area=box.area,
            expanded=True
        )
    
    def expand_all(
        self,
        result: ExtractionResult,
        pixels: int = None,
        percent: float = None
    ) -> ExtractionResult:
        """Expand all boxes in a result."""
        expanded_boxes = [
            self.expand_box(box, pixels, percent) 
            for box in result.boxes
        ]
        
        return ExtractionResult(
            boxes=expanded_boxes,
            image_shape=result.image_shape,
            processing_time=result.processing_time,
            params={**result.params, 'additional_expansion': True}
        )
    
    def make_square(
        self,
        box: BoundingBox,
        mode: str = 'expand'
    ) -> BoundingBox:
        """
        Convert box to square.
        
        Parameters
        ----------
        box : BoundingBox
            Box to convert
        mode : str
            'expand' = use larger dimension
            'shrink' = use smaller dimension
            'center' = average dimensions
            
        Returns
        -------
        BoundingBox
            Square bounding box
        """
        if mode == 'expand':
            side = max(box.width, box.height)
        elif mode == 'shrink':
            side = min(box.width, box.height)
        else:  # center
            side = (box.width + box.height) // 2
        
        # Center the square on the original box center
        new_x = box.center_x - side // 2
        new_y = box.center_y - side // 2
        
        # Clip to image boundaries
        if self.image_shape:
            img_h, img_w = self.image_shape
            new_x = max(0, min(new_x, img_w - side))
            new_y = max(0, min(new_y, img_h - side))
        
        return BoundingBox(
            x=max(0, new_x), y=max(0, new_y), 
            width=side, height=side,
            label=box.label,
            index=box.index,
            contour_index=box.contour_index,
            area=box.area,
            expanded=box.expanded
        )
    
    # ========================================================
    # BOX MERGING AND NMS
    # ========================================================
    
    def merge_overlapping(
        self,
        boxes: List[BoundingBox],
        iou_threshold: float = 0.5
    ) -> List[BoundingBox]:
        """
        Merge overlapping bounding boxes.
        
        Parameters
        ----------
        boxes : list
            List of bounding boxes
        iou_threshold : float
            IoU threshold for merging (0-1)
            
        Returns
        -------
        list
            Merged bounding boxes
        """
        if len(boxes) <= 1:
            return boxes
        
        # Track which boxes have been merged
        merged = [False] * len(boxes)
        result = []
        
        for i, box_i in enumerate(boxes):
            if merged[i]:
                continue
            
            # Find all overlapping boxes
            to_merge = [box_i]
            for j, box_j in enumerate(boxes[i+1:], start=i+1):
                if merged[j]:
                    continue
                if box_i.iou(box_j) >= iou_threshold:
                    to_merge.append(box_j)
                    merged[j] = True
            
            # Merge by taking union
            if len(to_merge) > 1:
                x1 = min(b.x1 for b in to_merge)
                y1 = min(b.y1 for b in to_merge)
                x2 = max(b.x2 for b in to_merge)
                y2 = max(b.y2 for b in to_merge)
                
                merged_box = BoundingBox(
                    x=x1, y=y1, width=x2-x1, height=y2-y1,
                    label=box_i.label,
                    index=len(result) + 1,
                    area=sum(b.area for b in to_merge),
                    expanded=any(b.expanded for b in to_merge)
                )
                result.append(merged_box)
            else:
                box_i.index = len(result) + 1
                result.append(box_i)
        
        if self.verbose:
            print(f"   Merged {len(boxes)} boxes to {len(result)} boxes")
        
        return result
    
    def non_max_suppression(
        self,
        boxes: List[BoundingBox],
        iou_threshold: float = 0.5,
        score_key: str = 'area'
    ) -> List[BoundingBox]:
        """
        Apply Non-Maximum Suppression to remove overlapping boxes.
        
        Parameters
        ----------
        boxes : list
            List of bounding boxes
        iou_threshold : float
            IoU threshold for suppression
        score_key : str
            'area' or 'confidence' for scoring
            
        Returns
        -------
        list
            Filtered bounding boxes
        """
        if len(boxes) <= 1:
            return boxes
        
        # Sort by score (descending)
        if score_key == 'area':
            sorted_boxes = sorted(boxes, key=lambda b: b.area, reverse=True)
        else:
            sorted_boxes = sorted(boxes, key=lambda b: b.confidence, reverse=True)
        
        keep = []
        suppressed = [False] * len(sorted_boxes)
        
        for i, box_i in enumerate(sorted_boxes):
            if suppressed[i]:
                continue
            
            keep.append(box_i)
            
            # Suppress overlapping boxes with lower score
            for j in range(i + 1, len(sorted_boxes)):
                if suppressed[j]:
                    continue
                if box_i.iou(sorted_boxes[j]) >= iou_threshold:
                    suppressed[j] = True
        
        # Re-index
        for i, box in enumerate(keep):
            box.index = i + 1
        
        if self.verbose:
            print(f"   NMS: {len(boxes)} → {len(keep)} boxes")
        
        return keep
    
    def filter_by_size(
        self,
        boxes: List[BoundingBox],
        min_area: int = 0,
        max_area: int = None,
        min_aspect: float = None,
        max_aspect: float = None
    ) -> List[BoundingBox]:
        """
        Filter boxes by size and aspect ratio.
        
        Parameters
        ----------
        boxes : list
            List of bounding boxes
        min_area : int
            Minimum box area
        max_area : int
            Maximum box area
        min_aspect : float
            Minimum aspect ratio
        max_aspect : float
            Maximum aspect ratio
            
        Returns
        -------
        list
            Filtered boxes
        """
        filtered = []
        for box in boxes:
            if box.box_area < min_area:
                continue
            if max_area and box.box_area > max_area:
                continue
            if min_aspect and box.aspect_ratio < min_aspect:
                continue
            if max_aspect and box.aspect_ratio > max_aspect:
                continue
            filtered.append(box)
        
        # Re-index
        for i, box in enumerate(filtered):
            box.index = i + 1
        
        return filtered
    
    # ========================================================
    # FORMAT CONVERSIONS
    # ========================================================
    
    def to_format(
        self,
        boxes: List[BoundingBox],
        format_type: BBoxFormat
    ) -> List[Tuple[int, int, int, int]]:
        """
        Convert boxes to specific format.
        
        Parameters
        ----------
        boxes : list
            List of BoundingBox objects
        format_type : BBoxFormat
            Target format (XYWH, XYXY, CXCYWH)
            
        Returns
        -------
        list
            List of coordinate tuples
        """
        if format_type == BBoxFormat.XYWH:
            return [b.to_xywh() for b in boxes]
        elif format_type == BBoxFormat.XYXY:
            return [b.to_xyxy() for b in boxes]
        else:  # CXCYWH
            return [b.to_cxcywh() for b in boxes]
    
    def from_format(
        self,
        coords_list: List[Tuple[int, int, int, int]],
        format_type: BBoxFormat,
        labels: List[str] = None
    ) -> List[BoundingBox]:
        """
        Create BoundingBox objects from coordinate tuples.
        
        Parameters
        ----------
        coords_list : list
            List of coordinate tuples
        format_type : BBoxFormat
            Input format
        labels : list
            Optional labels
            
        Returns
        -------
        list
            List of BoundingBox objects
        """
        boxes = []
        for i, coords in enumerate(coords_list):
            if format_type == BBoxFormat.XYWH:
                x, y, w, h = coords
            elif format_type == BBoxFormat.XYXY:
                x1, y1, x2, y2 = coords
                x, y = x1, y1
                w, h = x2 - x1, y2 - y1
            else:  # CXCYWH
                cx, cy, w, h = coords
                x, y = cx - w // 2, cy - h // 2
            
            label = labels[i] if labels and i < len(labels) else ""
            boxes.append(BoundingBox(
                x=x, y=y, width=w, height=h,
                label=label, index=i+1
            ))
        
        return boxes
    
    # ========================================================
    # VISUALIZATION
    # ========================================================
    
    def visualize(
        self,
        image: np.ndarray,
        result: Union[ExtractionResult, List[BoundingBox]],
        color: Tuple[int, int, int] = None,
        thickness: int = 2,
        draw_labels: bool = True,
        draw_index: bool = True,
        draw_area: bool = False,
        font_scale: float = 0.5,
        use_class_colors: bool = True
    ) -> np.ndarray:
        """
        Visualize bounding boxes on an image.
        
        Parameters
        ----------
        image : np.ndarray
            Image to draw on (will be copied)
        result : ExtractionResult or list
            Boxes to draw
        color : tuple
            Default BGR color (None = use class colors)
        thickness : int
            Line thickness
        draw_labels : bool
            Draw class labels
        draw_index : bool
            Draw box indices
        draw_area : bool
            Draw area values
        font_scale : float
            Font size multiplier
        use_class_colors : bool
            Use different colors per class
            
        Returns
        -------
        np.ndarray
            Annotated image
        """
        # Convert grayscale to BGR
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()
        
        # Get boxes list
        boxes = result.boxes if isinstance(result, ExtractionResult) else result
        
        for box in boxes:
            # Determine color
            if color:
                box_color = color
            elif use_class_colors and box.label in BBOX_COLORS:
                box_color = BBOX_COLORS[box.label]
            else:
                box_color = BBOX_COLORS['default']
            
            # Draw rectangle
            cv2.rectangle(
                vis, 
                (box.x, box.y), 
                (box.x2, box.y2), 
                box_color, 
                thickness
            )
            
            # Build label text
            label_parts = []
            if draw_index:
                label_parts.append(f"#{box.index}")
            if draw_labels and box.label:
                label_parts.append(box.label)
            if draw_area:
                label_parts.append(f"A:{box.area:.0f}")
            
            if label_parts:
                label_text = " ".join(label_parts)
                
                # Calculate text size
                (text_w, text_h), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                )
                
                # Draw background rectangle
                cv2.rectangle(
                    vis,
                    (box.x, box.y - text_h - 8),
                    (box.x + text_w + 4, box.y),
                    box_color,
                    -1
                )
                
                # Draw text
                cv2.putText(
                    vis, label_text,
                    (box.x + 2, box.y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    1
                )
        
        return vis
    
    def visualize_comparison(
        self,
        image: np.ndarray,
        original_boxes: List[BoundingBox],
        expanded_boxes: List[BoundingBox],
        figsize: Tuple[int, int] = (16, 8)
    ) -> 'matplotlib.figure.Figure':
        """
        Create side-by-side comparison of original vs expanded boxes.
        
        Parameters
        ----------
        image : np.ndarray
            Source image
        original_boxes : list
            Original bounding boxes
        expanded_boxes : list
            Expanded bounding boxes
        figsize : tuple
            Figure size
            
        Returns
        -------
        matplotlib.figure.Figure
            Comparison figure
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Original boxes
        vis_orig = self.visualize(
            image, original_boxes, 
            color=(0, 255, 0), draw_area=True
        )
        axes[0].imshow(cv2.cvtColor(vis_orig, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'Original Boxes ({len(original_boxes)})', fontweight='bold')
        axes[0].axis('off')
        
        # Expanded boxes
        vis_exp = self.visualize(
            image, expanded_boxes,
            color=(255, 0, 0), draw_area=True
        )
        axes[1].imshow(cv2.cvtColor(vis_exp, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Expanded Boxes (+{self.expand_pixels}px)', fontweight='bold')
        axes[1].axis('off')
        
        plt.suptitle('Bounding Box Expansion Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    # ========================================================
    # UTILITY METHODS
    # ========================================================
    
    def print_box_table(
        self,
        result: Union[ExtractionResult, List[BoundingBox]]
    ) -> None:
        """Print formatted table of bounding boxes."""
        boxes = result.boxes if isinstance(result, ExtractionResult) else result
        
        print(f"\n{'='*90}")
        print("BOUNDING BOX SUMMARY")
        print(f"{'='*90}")
        print(f"{'#':<4} {'(x,y)':<15} {'Size (w×h)':<15} {'Area':<10} "
              f"{'Aspect':<8} {'Label':<15} {'Exp':<5}")
        print(f"{'-'*90}")
        
        for box in boxes:
            pos = f"({box.x},{box.y})"
            size = f"{box.width}×{box.height}"
            exp = "Yes" if box.expanded else "No"
            print(f"{box.index:<4} {pos:<15} {size:<15} {box.box_area:<10} "
                  f"{box.aspect_ratio:<8.2f} {box.label:<15} {exp:<5}")
        
        print(f"{'='*90}")
        
        if isinstance(result, ExtractionResult):
            print(f"Total: {result.num_boxes} boxes, {result.total_area} px² total area")
    
    def get_roi_coordinates(
        self,
        boxes: List[BoundingBox]
    ) -> List[Dict[str, int]]:
        """
        Get ROI coordinates for image cropping.
        
        Parameters
        ----------
        boxes : list
            List of bounding boxes
            
        Returns
        -------
        list
            List of dicts with x1, y1, x2, y2 coordinates
        """
        return [
            {
                'index': box.index,
                'x1': box.x1,
                'y1': box.y1,
                'x2': box.x2,
                'y2': box.y2,
                'label': box.label
            }
            for box in boxes
        ]
    
    def save_results(
        self,
        result: ExtractionResult,
        output_path: Union[str, Path]
    ) -> None:
        """Save extraction results to JSON."""
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
    """Example usage of BoundingBoxExtractor."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='PCB Defect Bounding Box Extraction'
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
        default='outputs/bboxes',
        help='Output directory'
    )
    
    parser.add_argument(
        '-e', '--expand',
        type=int,
        default=10,
        help='Expansion pixels'
    )
    
    args = parser.parse_args()
    
    # Import contour detector
    from contour_detector import ContourDetector
    
    # Load mask
    mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Could not load mask from {args.mask_path}")
        return 1
    
    # Detect contours
    detector = ContourDetector(min_area=50, verbose=True)
    detection = detector.detect(mask)
    
    # Extract bounding boxes
    extractor = BoundingBoxExtractor(expand_pixels=args.expand, verbose=True)
    result = extractor.extract_from_detection(detection)
    
    # Print table
    extractor.print_box_table(result)
    
    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    extractor.save_results(result, output_dir / 'bbox_results.json')
    
    # Visualize
    image = mask if args.image is None else cv2.imread(args.image)
    vis = extractor.visualize(image, result)
    cv2.imwrite(str(output_dir / 'bbox_visualization.png'), vis)
    
    print(f"\n✓ Results saved to: {output_dir}")
    return 0


if __name__ == '__main__':
    exit(main())
