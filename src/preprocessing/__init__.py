"""
Preprocessing Module
====================

Contains tools for loading, preparing, and processing PCB images.

Classes:
--------
- PCBDatasetLoader: Load and manage the PCB defect dataset
- ImageSubtractor: Perform image alignment and subtraction
- MaskGenerator: Create binary masks and detect defect regions
"""

from .pcb_dataset_loader import PCBDatasetLoader
from .image_subtraction import ImageSubtractor, AlignmentMethod
from .mask_generation import MaskGenerator, ThresholdMethod

__all__ = [
    'PCBDatasetLoader', 
    'ImageSubtractor', 
    'AlignmentMethod',
    'MaskGenerator',
    'ThresholdMethod'
]
