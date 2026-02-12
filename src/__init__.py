"""
PCB Defect Detection System
============================

A comprehensive system for detecting and classifying defects in 
Printed Circuit Boards (PCBs) using computer vision and deep learning.

Modules:
--------
- preprocessing: Image loading, alignment, and subtraction
- utils: Utility functions for image operations
"""

__version__ = "1.0.0"
__author__ = "PCB Defect Detection Team"

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Output directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Defect classes
DEFECT_CLASSES = [
    "Missing_hole",
    "Mouse_bite", 
    "Open_circuit",
    "Short",
    "Spur",
    "Spurious_copper"
]

# Image settings
DEFAULT_IMAGE_SIZE = (640, 640)
SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
