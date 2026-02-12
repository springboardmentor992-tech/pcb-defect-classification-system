"""
Utilities Module
================

Common utility functions for image processing, visualization, and export.

Modules:
--------
- image_utils: Core image processing functions
- export_utils: Export detection results in various formats
- visualization: Charts, graphs, and visualization utilities

Functions:
----------
- load_image: Load image from path
- save_image: Save image to path
- visualize_comparison: Display multiple images side by side
- apply_gaussian_blur: Apply Gaussian smoothing
- create_difference_map: Calculate absolute difference between images
- apply_morphological_operations: Apply morphological transforms
"""

from .image_utils import (
    load_image,
    save_image,
    visualize_comparison,
    apply_gaussian_blur,
    create_difference_map,
    apply_morphological_operations,
    resize_image,
    convert_to_grayscale
)

# Export utilities
try:
    from .export_utils import (
        export_defects_csv,
        export_summary_csv,
        export_results_json,
        export_text_report,
        numpy_to_bytes,
        create_comparison_image,
        export_all_formats,
        get_timestamp_filename
    )
except ImportError:
    pass

# Visualization utilities
try:
    from .visualization import (
        create_bar_chart_svg,
        create_pie_chart_svg,
        create_confidence_histogram_svg,
        calculate_statistics,
        create_stats_card_html,
        create_defect_heatmap,
        draw_enhanced_annotations,
        DEFECT_COLORS,
        DEFECT_DISPLAY_NAMES
    )
except ImportError:
    pass

__all__ = [
    # Image utilities
    'load_image',
    'save_image', 
    'visualize_comparison',
    'apply_gaussian_blur',
    'create_difference_map',
    'apply_morphological_operations',
    'resize_image',
    'convert_to_grayscale',
    # Export utilities
    'export_defects_csv',
    'export_summary_csv',
    'export_results_json',
    'export_text_report',
    'numpy_to_bytes',
    'create_comparison_image',
    'export_all_formats',
    'get_timestamp_filename',
    # Visualization utilities
    'create_bar_chart_svg',
    'create_pie_chart_svg',
    'create_confidence_histogram_svg',
    'calculate_statistics',
    'create_stats_card_html',
    'create_defect_heatmap',
    'draw_enhanced_annotations',
    'DEFECT_COLORS',
    'DEFECT_DISPLAY_NAMES'
]
