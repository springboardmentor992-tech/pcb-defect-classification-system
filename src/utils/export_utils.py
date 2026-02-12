"""
PCB Defect Detection - Export Utilities
========================================

Utilities for exporting detection results in various formats:
- Annotated images (PNG/JPEG)
- CSV reports
- JSON data
- Text summaries
- PDF reports (optional)

Author: PCB Defect Detection Team
Version: 1.0.0
"""

import io
import csv
import json
import base64
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import numpy as np
import cv2
from PIL import Image


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class ExportConfig:
    """Configuration for export operations."""
    include_timestamp: bool = True
    image_format: str = 'png'  # 'png' or 'jpeg'
    image_quality: int = 95
    csv_delimiter: str = ','
    json_indent: int = 2


@dataclass
class DefectSummary:
    """Summary of defects for export."""
    index: int
    defect_type: str
    confidence: float
    x: int
    y: int
    width: int
    height: int
    area: int
    
    @classmethod
    def from_api_response(cls, defect: Dict) -> 'DefectSummary':
        """Create from API response dict."""
        bbox = defect.get('bbox', {})
        return cls(
            index=defect.get('index', 0),
            defect_type=defect.get('class', 'Unknown'),
            confidence=defect.get('confidence', 0.0),
            x=bbox.get('x', 0),
            y=bbox.get('y', 0),
            width=bbox.get('width', 0),
            height=bbox.get('height', 0),
            area=defect.get('area', 0)
        )


# ============================================================
# IMAGE EXPORT
# ============================================================

def numpy_to_bytes(
    image: np.ndarray,
    format: str = 'png',
    quality: int = 95
) -> bytes:
    """
    Convert numpy array to image bytes.
    
    Args:
        image: Image as numpy array (RGB or BGR)
        format: Output format ('png' or 'jpeg')
        quality: JPEG quality (1-100)
        
    Returns:
        Image as bytes
    """
    # Convert to PIL Image
    if len(image.shape) == 2:
        # Grayscale
        pil_image = Image.fromarray(image)
    elif image.shape[2] == 3:
        # Assume RGB
        pil_image = Image.fromarray(image)
    elif image.shape[2] == 4:
        # RGBA
        pil_image = Image.fromarray(image)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    
    # Save to bytes
    buffer = io.BytesIO()
    
    if format.lower() == 'png':
        pil_image.save(buffer, format='PNG')
    elif format.lower() in ('jpg', 'jpeg'):
        # Convert to RGB if RGBA
        if pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB')
        pil_image.save(buffer, format='JPEG', quality=quality)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return buffer.getvalue()


def create_comparison_image(
    original: np.ndarray,
    annotated: np.ndarray,
    add_labels: bool = True
) -> np.ndarray:
    """
    Create a side-by-side comparison image.
    
    Args:
        original: Original image
        annotated: Annotated image with detections
        add_labels: Whether to add "Original" and "Detected" labels
        
    Returns:
        Combined comparison image
    """
    # Ensure same dimensions
    h1, w1 = original.shape[:2]
    h2, w2 = annotated.shape[:2]
    
    if h1 != h2 or w1 != w2:
        # Resize to match
        annotated = cv2.resize(annotated, (w1, h1))
    
    # Ensure both are 3-channel
    if len(original.shape) == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    if len(annotated.shape) == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2RGB)
    
    # Create separator
    separator_width = 10
    separator = np.ones((h1, separator_width, 3), dtype=np.uint8) * 128
    
    # Combine images
    combined = np.hstack([original, separator, annotated])
    
    if add_labels:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = min(w1, h1) / 500
        thickness = max(1, int(font_scale * 2))
        
        # Add "Original" label
        cv2.putText(combined, "Original", (10, 30), font, font_scale, 
                    (255, 255, 255), thickness + 2, cv2.LINE_AA)
        cv2.putText(combined, "Original", (10, 30), font, font_scale, 
                    (0, 0, 0), thickness, cv2.LINE_AA)
        
        # Add "Detected" label
        cv2.putText(combined, "Detected", (w1 + separator_width + 10, 30), font, 
                    font_scale, (255, 255, 255), thickness + 2, cv2.LINE_AA)
        cv2.putText(combined, "Detected", (w1 + separator_width + 10, 30), font, 
                    font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    
    return combined


def create_defect_grid(
    defects: List[Dict],
    source_image: np.ndarray,
    grid_size: Tuple[int, int] = (4, 4),
    cell_size: Tuple[int, int] = (128, 128)
) -> Optional[np.ndarray]:
    """
    Create a grid of cropped defect regions.
    
    Args:
        defects: List of detected defects
        source_image: Original image to crop from
        grid_size: (rows, cols) grid dimensions
        cell_size: Size of each cell in pixels
        
    Returns:
        Grid image or None if no defects
    """
    if not defects:
        return None
    
    max_defects = grid_size[0] * grid_size[1]
    selected = defects[:max_defects]
    
    # Create grid
    grid_h = grid_size[0] * cell_size[1]
    grid_w = grid_size[1] * cell_size[0]
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 32  # Dark background
    
    for idx, defect in enumerate(selected):
        row = idx // grid_size[1]
        col = idx % grid_size[1]
        
        # Get bounding box
        bbox = defect.get('bbox', {})
        x, y = bbox.get('x', 0), bbox.get('y', 0)
        w, h = bbox.get('width', 64), bbox.get('height', 64)
        
        # Expand slightly
        pad = 10
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(source_image.shape[1] - x, w + 2 * pad)
        h = min(source_image.shape[0] - y, h + 2 * pad)
        
        # Crop region
        crop = source_image[y:y+h, x:x+w]
        
        if crop.size == 0:
            continue
        
        # Resize to cell size
        crop_resized = cv2.resize(crop, cell_size)
        
        # Place in grid
        y_start = row * cell_size[1]
        x_start = col * cell_size[0]
        grid[y_start:y_start+cell_size[1], x_start:x_start+cell_size[0]] = crop_resized
        
        # Add label
        label = f"#{idx+1}: {defect.get('class', '?')[:8]}"
        cv2.putText(grid, label, (x_start + 5, y_start + cell_size[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    
    return grid


# ============================================================
# CSV EXPORT
# ============================================================

def export_defects_csv(
    defects: List[Dict],
    metadata: Optional[Dict] = None,
    config: Optional[ExportConfig] = None
) -> str:
    """
    Export defects to CSV format.
    
    Args:
        defects: List of detected defects
        metadata: Optional metadata (timestamp, image info, etc.)
        config: Export configuration
        
    Returns:
        CSV content as string
    """
    config = config or ExportConfig()
    
    output = io.StringIO()
    
    # Write metadata as comments
    if metadata:
        output.write(f"# PCB Defect Detection Report\n")
        output.write(f"# Generated: {metadata.get('timestamp', datetime.now().isoformat())}\n")
        output.write(f"# Total Defects: {len(defects)}\n")
        if 'processing_time' in metadata:
            output.write(f"# Processing Time: {metadata['processing_time']:.2f}s\n")
        output.write(f"#\n")
    
    # Create CSV writer
    writer = csv.writer(output, delimiter=config.csv_delimiter)
    
    # Write header
    header = ['Index', 'Defect Type', 'Confidence (%)', 'X', 'Y', 'Width', 'Height', 'Area']
    writer.writerow(header)
    
    # Write defects
    for defect in defects:
        bbox = defect.get('bbox', {})
        row = [
            defect.get('index', 0),
            defect.get('class', 'Unknown'),
            f"{defect.get('confidence', 0) * 100:.1f}",
            bbox.get('x', 0),
            bbox.get('y', 0),
            bbox.get('width', 0),
            bbox.get('height', 0),
            defect.get('area', 0)
        ]
        writer.writerow(row)
    
    return output.getvalue()


def export_summary_csv(
    defects: List[Dict],
    metadata: Optional[Dict] = None
) -> str:
    """
    Export defect summary (counts by type) to CSV.
    
    Args:
        defects: List of detected defects
        metadata: Optional metadata
        
    Returns:
        CSV content as string
    """
    # Count by type
    from collections import Counter
    type_counts = Counter(d.get('class', 'Unknown') for d in defects)
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['Defect Type', 'Count', 'Percentage'])
    
    # Write summary
    total = len(defects)
    for defect_type, count in sorted(type_counts.items()):
        percentage = (count / total * 100) if total > 0 else 0
        writer.writerow([defect_type, count, f"{percentage:.1f}%"])
    
    # Write total
    writer.writerow(['TOTAL', total, '100%'])
    
    return output.getvalue()


# ============================================================
# JSON EXPORT
# ============================================================

def export_results_json(
    results: Dict,
    config: Optional[ExportConfig] = None
) -> str:
    """
    Export complete results to JSON format.
    
    Args:
        results: Detection results dictionary
        config: Export configuration
        
    Returns:
        JSON content as string
    """
    config = config or ExportConfig()
    
    # Create export structure
    export_data = {
        'export_info': {
            'format': 'PCB Defect Detection Export',
            'version': '1.0',
            'timestamp': datetime.now().isoformat(),
        },
        'detection_results': {
            'success': results.get('success', False),
            'num_defects': results.get('num_defects', 0),
            'processing_time': results.get('processing_time', 0.0),
            'image_size': results.get('image_size', {}),
        },
        'summary': results.get('summary', {}),
        'defects': results.get('defects', [])
    }
    
    return json.dumps(export_data, indent=config.json_indent, default=str)


# ============================================================
# TEXT REPORT EXPORT
# ============================================================

def export_text_report(
    results: Dict,
    include_defect_list: bool = True
) -> str:
    """
    Export a human-readable text report.
    
    Args:
        results: Detection results dictionary
        include_defect_list: Whether to include individual defect details
        
    Returns:
        Text report as string
    """
    lines = []
    separator = "=" * 60
    
    # Header
    lines.append(separator)
    lines.append("PCB DEFECT DETECTION REPORT")
    lines.append(separator)
    lines.append("")
    
    # Timestamp
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Summary
    lines.append("-" * 40)
    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Total Defects Found:   {results.get('num_defects', 0)}")
    lines.append(f"Processing Time:       {results.get('processing_time', 0):.2f} seconds")
    
    image_size = results.get('image_size', {})
    if image_size:
        lines.append(f"Image Dimensions:      {image_size.get('width', 0)} x {image_size.get('height', 0)} px")
    
    # Summary by type
    summary = results.get('summary', {})
    defect_types = summary.get('defect_types', [])
    if defect_types:
        lines.append(f"Defect Types:          {', '.join(defect_types)}")
    
    avg_conf = summary.get('avg_confidence', 0)
    if avg_conf:
        lines.append(f"Average Confidence:    {avg_conf:.1%}")
    
    lines.append("")
    
    # Defect counts by type
    defects = results.get('defects', [])
    if defects:
        from collections import Counter
        type_counts = Counter(d.get('class', 'Unknown') for d in defects)
        
        lines.append("-" * 40)
        lines.append("DEFECTS BY TYPE")
        lines.append("-" * 40)
        
        for dtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            display_name = dtype.replace('_', ' ')
            lines.append(f"  {display_name:<20} {count:>5}")
        
        lines.append(f"  {'TOTAL':<20} {len(defects):>5}")
        lines.append("")
    
    # Individual defects
    if include_defect_list and defects:
        lines.append("-" * 40)
        lines.append("INDIVIDUAL DEFECTS")
        lines.append("-" * 40)
        
        for defect in defects[:50]:  # Limit to first 50
            bbox = defect.get('bbox', {})
            lines.append(
                f"  #{defect.get('index', 0):3d}  "
                f"{defect.get('class', 'Unknown'):<18}  "
                f"Conf: {defect.get('confidence', 0):.0%}  "
                f"Pos: ({bbox.get('x', 0)}, {bbox.get('y', 0)})"
            )
        
        if len(defects) > 50:
            lines.append(f"  ... and {len(defects) - 50} more defects")
        lines.append("")
    
    # Footer
    lines.append(separator)
    lines.append("End of Report")
    lines.append(separator)
    
    return "\n".join(lines)


# ============================================================
# DOWNLOAD HELPERS
# ============================================================

def create_download_link(
    data: Union[str, bytes],
    filename: str,
    mime_type: str = 'application/octet-stream',
    label: str = 'Download'
) -> str:
    """
    Create an HTML download link for Streamlit.
    
    Args:
        data: Data to download (string or bytes)
        filename: Suggested filename
        mime_type: MIME type of the data
        label: Link text
        
    Returns:
        HTML string with download link
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    b64 = base64.b64encode(data).decode()
    
    return f'''
    <a href="data:{mime_type};base64,{b64}" 
       download="{filename}" 
       style="
           display: inline-block;
           padding: 0.5rem 1rem;
           background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
           color: white;
           text-decoration: none;
           border-radius: 8px;
           font-weight: 500;
           transition: all 0.3s ease;
           box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
       "
       onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(102, 126, 234, 0.5)';"
       onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 15px rgba(102, 126, 234, 0.4)';">
       {label}
    </a>
    '''


def get_timestamp_filename(base: str, extension: str) -> str:
    """
    Create a timestamped filename.
    
    Args:
        base: Base filename (without extension)
        extension: File extension (without dot)
        
    Returns:
        Filename with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{timestamp}.{extension}"


# ============================================================
# BATCH EXPORT
# ============================================================

def export_all_formats(
    results: Dict,
    annotated_image: Optional[np.ndarray] = None,
    original_image: Optional[np.ndarray] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Union[str, bytes]]:
    """
    Export results in all available formats.
    
    Args:
        results: Detection results
        annotated_image: Annotated image array
        original_image: Original image array
        output_dir: Optional directory to save files
        
    Returns:
        Dictionary with format names and data
    """
    exports = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON export
    exports['json'] = {
        'filename': f'pcb_detection_{timestamp}.json',
        'data': export_results_json(results),
        'mime': 'application/json'
    }
    
    # CSV export (detailed)
    exports['csv_detailed'] = {
        'filename': f'pcb_defects_{timestamp}.csv',
        'data': export_defects_csv(
            results.get('defects', []),
            {'timestamp': timestamp, 'processing_time': results.get('processing_time', 0)}
        ),
        'mime': 'text/csv'
    }
    
    # CSV export (summary)
    exports['csv_summary'] = {
        'filename': f'pcb_summary_{timestamp}.csv',
        'data': export_summary_csv(results.get('defects', [])),
        'mime': 'text/csv'
    }
    
    # Text report
    exports['text'] = {
        'filename': f'pcb_report_{timestamp}.txt',
        'data': export_text_report(results),
        'mime': 'text/plain'
    }
    
    # Annotated image
    if annotated_image is not None:
        exports['annotated_image'] = {
            'filename': f'pcb_annotated_{timestamp}.png',
            'data': numpy_to_bytes(annotated_image, 'png'),
            'mime': 'image/png'
        }
    
    # Comparison image
    if annotated_image is not None and original_image is not None:
        comparison = create_comparison_image(original_image, annotated_image)
        exports['comparison_image'] = {
            'filename': f'pcb_comparison_{timestamp}.png',
            'data': numpy_to_bytes(comparison, 'png'),
            'mime': 'image/png'
        }
    
    # Save to disk if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for key, export in exports.items():
            filepath = output_dir / export['filename']
            data = export['data']
            
            if isinstance(data, str):
                with open(filepath, 'w') as f:
                    f.write(data)
            else:
                with open(filepath, 'wb') as f:
                    f.write(data)
    
    return exports


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    # Test with sample data
    sample_results = {
        'success': True,
        'num_defects': 5,
        'processing_time': 2.5,
        'image_size': {'width': 1920, 'height': 1080},
        'summary': {
            'defect_types': ['Short', 'Missing_hole', 'Spur'],
            'avg_confidence': 0.85
        },
        'defects': [
            {'index': 1, 'class': 'Short', 'confidence': 0.92, 
             'bbox': {'x': 100, 'y': 200, 'width': 50, 'height': 40}, 'area': 1500},
            {'index': 2, 'class': 'Missing_hole', 'confidence': 0.88,
             'bbox': {'x': 300, 'y': 150, 'width': 30, 'height': 30}, 'area': 700},
        ]
    }
    
    print("Testing Export Utilities...")
    
    # Test JSON export
    json_data = export_results_json(sample_results)
    print(f"\n✓ JSON export: {len(json_data)} bytes")
    
    # Test CSV export
    csv_data = export_defects_csv(sample_results['defects'])
    print(f"✓ CSV export: {len(csv_data)} bytes")
    
    # Test text report
    text_data = export_text_report(sample_results)
    print(f"✓ Text report: {len(text_data)} bytes")
    print("\nSample text report:")
    print(text_data[:500])
    
    print("\n✓ All export tests passed!")
