"""
Detection Module
================

Contains tools for detecting, analyzing, and extracting defect regions
from PCB images.

Classes:
--------
- ContourDetector: Advanced contour detection and analysis
- BoundingBoxExtractor: Bounding box extraction and manipulation
- ROIExtractor: Extract and manage Regions of Interest

Enums:
------
- ContourMode: Contour retrieval modes
- ContourApproximation: Contour approximation methods
- ShapeType: Shape classification types
- BBoxFormat: Bounding box format types
- DatasetSplit: Dataset split types (train/val/test)

Data Classes:
-------------
- ContourProperties: Comprehensive contour measurements
- DetectionResult: Results from contour detection
- BoundingBox: Single bounding box with properties
- ExtractionResult: Results from bbox extraction
- ROIMetadata: ROI information and paths
- ExtractionStats: ROI extraction statistics
"""

from .contour_detector import (
    ContourDetector,
    ContourMode,
    ContourApproximation,
    ShapeType,
    ContourProperties,
    DetectionResult
)

from .bbox_extractor import (
    BoundingBoxExtractor,
    BBoxFormat,
    BoundingBox,
    ExtractionResult
)

from .roi_extractor import (
    ROIExtractor,
    DatasetSplit,
    ROIMetadata,
    ExtractionStats
)

__all__ = [
    # Contour Detection
    'ContourDetector',
    'ContourMode',
    'ContourApproximation',
    'ShapeType',
    'ContourProperties',
    'DetectionResult',
    # Bounding Box Extraction
    'BoundingBoxExtractor',
    'BBoxFormat',
    'BoundingBox',
    'ExtractionResult',
    # ROI Extraction
    'ROIExtractor',
    'DatasetSplit',
    'ROIMetadata',
    'ExtractionStats'
]

