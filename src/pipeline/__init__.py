"""
PCB Defect Detection - Pipeline Module
=======================================

This module provides the end-to-end inference pipeline for
PCB defect detection.

Main Components:
    - PCBDefectPipeline: End-to-end detection pipeline
    - DetectionResult: Result container with all detection info
    - DetectedDefect: Single defect representation
    - PipelineConfig: Configuration dataclass

Usage:
    >>> from pipeline import create_pipeline
    >>> pipeline = create_pipeline()
    >>> result = pipeline.process(template, test)
"""

from .inference_pipeline import (
    PCBDefectPipeline,
    PipelineConfig,
    DetectionResult,
    DetectedDefect,
    create_pipeline
)

__all__ = [
    'PCBDefectPipeline',
    'PipelineConfig', 
    'DetectionResult',
    'DetectedDefect',
    'create_pipeline'
]
