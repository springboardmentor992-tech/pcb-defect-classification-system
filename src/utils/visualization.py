"""
PCB Defect Detection - Visualization Utilities
===============================================

Advanced visualization utilities for PCB defect detection results:
- Interactive charts and graphs
- Defect heatmaps
- Confidence distributions
- Statistics dashboards

Author: PCB Defect Detection Team
Version: 1.0.0
"""

import io
import base64
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# COLOR SCHEMES
# ============================================================

# Defect class colors (RGB)
DEFECT_COLORS = {
    'Missing_hole': (239, 68, 68),      # Red
    'Mouse_bite': (34, 197, 94),        # Green
    'Open_circuit': (59, 130, 246),     # Blue
    'Short': (234, 179, 8),             # Yellow/Amber
    'Spur': (168, 85, 247),             # Purple
    'Spurious_copper': (6, 182, 212),   # Cyan
    'Unknown': (156, 163, 175)          # Gray
}

# Display names
DEFECT_DISPLAY_NAMES = {
    'Missing_hole': 'Missing Hole',
    'Mouse_bite': 'Mouse Bite',
    'Open_circuit': 'Open Circuit',
    'Short': 'Short',
    'Spur': 'Spur',
    'Spurious_copper': 'Spurious Copper',
    'Unknown': 'Unknown'
}

# Confidence color gradient (low to high)
CONFIDENCE_COLORS = [
    (239, 68, 68),   # Red (low)
    (234, 179, 8),   # Yellow (medium)
    (34, 197, 94),   # Green (high)
]


# ============================================================
# CHART GENERATION
# ============================================================

def create_bar_chart_svg(
    data: Dict[str, int],
    width: int = 400,
    height: int = 250,
    title: str = "Defects by Type",
    show_values: bool = True
) -> str:
    """
    Create an SVG bar chart.
    
    Args:
        data: Dictionary of {label: value}
        width: Chart width in pixels
        height: Chart height in pixels
        title: Chart title
        show_values: Show value labels on bars
        
    Returns:
        SVG string
    """
    if not data:
        return f'<svg width="{width}" height="{height}"><text x="{width//2}" y="{height//2}" text-anchor="middle">No data</text></svg>'
    
    margin = {'top': 40, 'right': 20, 'bottom': 60, 'left': 50}
    chart_width = width - margin['left'] - margin['right']
    chart_height = height - margin['top'] - margin['bottom']
    
    labels = list(data.keys())
    values = list(data.values())
    max_value = max(values) if values else 1
    
    bar_width = chart_width / len(labels) * 0.8
    bar_gap = chart_width / len(labels) * 0.2
    
    svg_parts = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        f'<style>',
        f'  .title {{ font: bold 14px sans-serif; fill: #f8fafc; }}',
        f'  .label {{ font: 10px sans-serif; fill: #94a3b8; }}',
        f'  .value {{ font: bold 11px sans-serif; fill: #f8fafc; }}',
        f'  .axis {{ stroke: #475569; stroke-width: 1; }}',
        f'</style>',
        f'<rect width="{width}" height="{height}" fill="#1e293b" rx="8"/>',
        f'<text x="{width//2}" y="25" text-anchor="middle" class="title">{title}</text>',
    ]
    
    # Draw bars
    for i, (label, value) in enumerate(zip(labels, values)):
        x = margin['left'] + i * (bar_width + bar_gap) + bar_gap / 2
        bar_height = (value / max_value) * chart_height if max_value > 0 else 0
        y = margin['top'] + chart_height - bar_height
        
        # Get color for this defect type
        color = DEFECT_COLORS.get(label, (100, 116, 139))
        color_hex = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
        
        # Bar with gradient
        svg_parts.append(
            f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" '
            f'fill="{color_hex}" rx="4" opacity="0.9"/>'
        )
        
        # Value label
        if show_values:
            svg_parts.append(
                f'<text x="{x + bar_width/2}" y="{y - 5}" text-anchor="middle" class="value">{value}</text>'
            )
        
        # X-axis label
        display_name = label.replace('_', ' ')[:10]
        svg_parts.append(
            f'<text x="{x + bar_width/2}" y="{height - margin["bottom"] + 15}" '
            f'text-anchor="middle" transform="rotate(-30, {x + bar_width/2}, {height - margin["bottom"] + 15})" '
            f'class="label">{display_name}</text>'
        )
    
    # X-axis line
    svg_parts.append(
        f'<line x1="{margin["left"]}" y1="{height - margin["bottom"]}" '
        f'x2="{width - margin["right"]}" y2="{height - margin["bottom"]}" class="axis"/>'
    )
    
    svg_parts.append('</svg>')
    
    return '\n'.join(svg_parts)


def create_pie_chart_svg(
    data: Dict[str, int],
    width: int = 300,
    height: int = 300,
    title: str = "Distribution"
) -> str:
    """
    Create an SVG pie/donut chart.
    
    Args:
        data: Dictionary of {label: value}
        width: Chart width in pixels  
        height: Chart height in pixels
        title: Chart title
        
    Returns:
        SVG string
    """
    if not data or sum(data.values()) == 0:
        return f'<svg width="{width}" height="{height}"><text x="{width//2}" y="{height//2}" text-anchor="middle" fill="#94a3b8">No data</text></svg>'
    
    cx, cy = width // 2, height // 2 + 15
    outer_radius = min(width, height) // 2 - 40
    inner_radius = outer_radius * 0.5  # Donut style
    
    total = sum(data.values())
    
    svg_parts = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        f'<style>',
        f'  .title {{ font: bold 14px sans-serif; fill: #f8fafc; }}',
        f'  .legend {{ font: 11px sans-serif; fill: #e2e8f0; }}',
        f'</style>',
        f'<rect width="{width}" height="{height}" fill="#1e293b" rx="8"/>',
        f'<text x="{width//2}" y="22" text-anchor="middle" class="title">{title}</text>',
    ]
    
    # Draw segments
    start_angle = -90  # Start from top
    
    for i, (label, value) in enumerate(data.items()):
        if value == 0:
            continue
            
        angle = (value / total) * 360
        end_angle = start_angle + angle
        
        # Calculate arc path
        large_arc = 1 if angle > 180 else 0
        
        start_rad = np.radians(start_angle)
        end_rad = np.radians(end_angle)
        
        x1_outer = cx + outer_radius * np.cos(start_rad)
        y1_outer = cy + outer_radius * np.sin(start_rad)
        x2_outer = cx + outer_radius * np.cos(end_rad)
        y2_outer = cy + outer_radius * np.sin(end_rad)
        
        x1_inner = cx + inner_radius * np.cos(end_rad)
        y1_inner = cy + inner_radius * np.sin(end_rad)
        x2_inner = cx + inner_radius * np.cos(start_rad)
        y2_inner = cy + inner_radius * np.sin(start_rad)
        
        # Get color
        color = DEFECT_COLORS.get(label, (100, 116, 139))
        color_hex = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
        
        # Arc path
        path = (
            f'M {x1_outer} {y1_outer} '
            f'A {outer_radius} {outer_radius} 0 {large_arc} 1 {x2_outer} {y2_outer} '
            f'L {x1_inner} {y1_inner} '
            f'A {inner_radius} {inner_radius} 0 {large_arc} 0 {x2_inner} {y2_inner} Z'
        )
        
        svg_parts.append(f'<path d="{path}" fill="{color_hex}" stroke="#1e293b" stroke-width="2"/>')
        
        start_angle = end_angle
    
    # Center text
    svg_parts.append(
        f'<text x="{cx}" y="{cy - 5}" text-anchor="middle" class="title">{total}</text>'
    )
    svg_parts.append(
        f'<text x="{cx}" y="{cy + 15}" text-anchor="middle" style="font: 11px sans-serif; fill: #94a3b8;">Total</text>'
    )
    
    svg_parts.append('</svg>')
    
    return '\n'.join(svg_parts)


def create_confidence_histogram_svg(
    confidences: List[float],
    width: int = 400,
    height: int = 200,
    bins: int = 10
) -> str:
    """
    Create an SVG histogram of confidence scores.
    
    Args:
        confidences: List of confidence values (0-1)
        width: Chart width
        height: Chart height
        bins: Number of histogram bins
        
    Returns:
        SVG string
    """
    if not confidences:
        return f'<svg width="{width}" height="{height}"><text x="{width//2}" y="{height//2}" text-anchor="middle" fill="#94a3b8">No data</text></svg>'
    
    margin = {'top': 40, 'right': 20, 'bottom': 40, 'left': 50}
    chart_width = width - margin['left'] - margin['right']
    chart_height = height - margin['top'] - margin['bottom']
    
    # Calculate histogram
    hist, bin_edges = np.histogram(confidences, bins=bins, range=(0, 1))
    max_count = max(hist) if len(hist) > 0 else 1
    
    bar_width = chart_width / bins
    
    svg_parts = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        f'<style>',
        f'  .title {{ font: bold 14px sans-serif; fill: #f8fafc; }}',
        f'  .label {{ font: 10px sans-serif; fill: #94a3b8; }}',
        f'  .axis {{ stroke: #475569; stroke-width: 1; }}',
        f'</style>',
        f'<rect width="{width}" height="{height}" fill="#1e293b" rx="8"/>',
        f'<text x="{width//2}" y="25" text-anchor="middle" class="title">Confidence Distribution</text>',
    ]
    
    # Draw bars
    for i, count in enumerate(hist):
        x = margin['left'] + i * bar_width
        bar_height = (count / max_count) * chart_height if max_count > 0 else 0
        y = margin['top'] + chart_height - bar_height
        
        # Color based on confidence level
        confidence = (i + 0.5) / bins
        if confidence < 0.5:
            color = '#ef4444'  # Red
        elif confidence < 0.75:
            color = '#eab308'  # Yellow
        else:
            color = '#22c55e'  # Green
        
        svg_parts.append(
            f'<rect x="{x}" y="{y}" width="{bar_width - 2}" height="{bar_height}" '
            f'fill="{color}" opacity="0.8" rx="2"/>'
        )
    
    # X-axis labels
    for i in range(0, bins + 1, 2):
        x = margin['left'] + i * bar_width
        label = f'{i * 10}%'
        svg_parts.append(
            f'<text x="{x}" y="{height - margin["bottom"] + 15}" text-anchor="middle" class="label">{label}</text>'
        )
    
    # Axes
    svg_parts.append(
        f'<line x1="{margin["left"]}" y1="{height - margin["bottom"]}" '
        f'x2="{width - margin["right"]}" y2="{height - margin["bottom"]}" class="axis"/>'
    )
    
    svg_parts.append('</svg>')
    
    return '\n'.join(svg_parts)


# ============================================================
# HEATMAP GENERATION
# ============================================================

def create_defect_heatmap(
    defects: List[Dict],
    image_size: Tuple[int, int],
    output_size: Tuple[int, int] = (400, 300),
    blur_kernel: int = 51
) -> np.ndarray:
    """
    Create a heatmap showing defect concentration.
    
    Args:
        defects: List of detected defects
        image_size: (width, height) of original image
        output_size: (width, height) of output heatmap
        blur_kernel: Gaussian blur kernel size
        
    Returns:
        Heatmap as RGB numpy array
    """
    # Create accumulator at original size
    heatmap = np.zeros((image_size[1], image_size[0]), dtype=np.float32)
    
    for defect in defects:
        bbox = defect.get('bbox', {})
        cx = bbox.get('x', 0) + bbox.get('width', 0) // 2
        cy = bbox.get('y', 0) + bbox.get('height', 0) // 2
        
        # Weight by confidence
        weight = defect.get('confidence', 0.5)
        
        # Add Gaussian blob at defect location
        radius = max(bbox.get('width', 30), bbox.get('height', 30))
        
        y1, y2 = max(0, cy - radius), min(image_size[1], cy + radius)
        x1, x2 = max(0, cx - radius), min(image_size[0], cx + radius)
        
        if y2 > y1 and x2 > x1:
            heatmap[y1:y2, x1:x2] += weight
    
    # Normalize
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    # Apply Gaussian blur
    if blur_kernel > 0:
        heatmap = cv2.GaussianBlur(heatmap, (blur_kernel, blur_kernel), 0)
    
    # Convert to colormap
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Resize to output size
    if output_size:
        heatmap_color = cv2.resize(heatmap_color, output_size)
    
    return heatmap_color


def create_overlay_heatmap(
    image: np.ndarray,
    defects: List[Dict],
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create a heatmap overlaid on the original image.
    
    Args:
        image: Original image (RGB)
        defects: List of detected defects
        alpha: Blend factor (0-1)
        
    Returns:
        Image with heatmap overlay
    """
    h, w = image.shape[:2]
    heatmap = create_defect_heatmap(defects, (w, h), (w, h))
    
    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    
    return overlay


# ============================================================
# STATISTICS VISUALIZATION
# ============================================================

def calculate_statistics(results: Dict) -> Dict:
    """
    Calculate comprehensive statistics from detection results.
    
    Args:
        results: Detection results dictionary
        
    Returns:
        Dictionary of statistics
    """
    defects = results.get('defects', [])
    
    if not defects:
        return {
            'total_defects': 0,
            'defect_types': {},
            'confidence': {'min': 0, 'max': 0, 'mean': 0, 'std': 0},
            'area': {'min': 0, 'max': 0, 'mean': 0, 'total': 0},
            'spatial': {'x_mean': 0, 'y_mean': 0}
        }
    
    # Count by type
    type_counts = Counter(d.get('class', 'Unknown') for d in defects)
    
    # Confidence stats
    confidences = [d.get('confidence', 0) for d in defects]
    
    # Area stats
    areas = [d.get('area', 0) for d in defects]
    
    # Spatial stats
    centers_x = [d.get('bbox', {}).get('x', 0) + d.get('bbox', {}).get('width', 0) // 2 for d in defects]
    centers_y = [d.get('bbox', {}).get('y', 0) + d.get('bbox', {}).get('height', 0) // 2 for d in defects]
    
    return {
        'total_defects': len(defects),
        'defect_types': dict(type_counts),
        'confidence': {
            'min': float(min(confidences)),
            'max': float(max(confidences)),
            'mean': float(np.mean(confidences)),
            'std': float(np.std(confidences))
        },
        'area': {
            'min': int(min(areas)),
            'max': int(max(areas)),
            'mean': float(np.mean(areas)),
            'total': int(sum(areas))
        },
        'spatial': {
            'x_mean': float(np.mean(centers_x)),
            'y_mean': float(np.mean(centers_y))
        }
    }


def create_stats_card_html(
    label: str,
    value: str,
    icon: str = "📊",
    color: str = "#667eea"
) -> str:
    """
    Create an HTML stats card.
    
    Args:
        label: Card label
        value: Card value
        icon: Emoji icon
        color: Accent color
        
    Returns:
        HTML string
    """
    return f'''
    <div style="
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(100, 116, 139, 0.3);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    ">
        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>
        <div style="font-size: 1.75rem; font-weight: bold; color: {color};">{value}</div>
        <div style="font-size: 0.875rem; color: #94a3b8; margin-top: 0.25rem;">{label}</div>
    </div>
    '''


# ============================================================
# ENHANCED ANNOTATION
# ============================================================

def draw_enhanced_annotations(
    image: np.ndarray,
    defects: List[Dict],
    show_confidence: bool = True,
    show_index: bool = True,
    box_thickness: int = 2,
    font_scale: float = 0.5
) -> np.ndarray:
    """
    Draw enhanced annotations on an image.
    
    Args:
        image: Input image (RGB or BGR)
        defects: List of detected defects
        show_confidence: Show confidence percentage
        show_index: Show defect index numbers
        box_thickness: Bounding box line thickness
        font_scale: Font scale factor
        
    Returns:
        Annotated image
    """
    result = image.copy()
    
    for defect in defects:
        bbox = defect.get('bbox', {})
        x, y = bbox.get('x', 0), bbox.get('y', 0)
        w, h = bbox.get('width', 0), bbox.get('height', 0)
        
        defect_type = defect.get('class', 'Unknown')
        confidence = defect.get('confidence', 0)
        index = defect.get('index', 0)
        
        # Get color
        color = DEFECT_COLORS.get(defect_type, (156, 163, 175))
        
        # Draw bounding box with rounded corners effect
        cv2.rectangle(result, (x, y), (x + w, y + h), color, box_thickness)
        
        # Draw corner accents
        corner_len = min(w, h) // 4
        for cx, cy in [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]:
            dx = 1 if cx == x else -1
            dy = 1 if cy == y else -1
            cv2.line(result, (cx, cy), (cx + dx * corner_len, cy), color, box_thickness + 1)
            cv2.line(result, (cx, cy), (cx, cy + dy * corner_len), color, box_thickness + 1)
        
        # Build label
        label_parts = []
        if show_index:
            label_parts.append(f"#{index}")
        label_parts.append(DEFECT_DISPLAY_NAMES.get(defect_type, defect_type))
        if show_confidence:
            label_parts.append(f"{confidence:.0%}")
        label = " ".join(label_parts)
        
        # Calculate label size
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        
        # Label background
        label_y = y - 5 if y > text_h + 10 else y + h + text_h + 5
        label_x = x
        
        # Semi-transparent background
        overlay = result.copy()
        cv2.rectangle(
            overlay,
            (label_x - 2, label_y - text_h - 5),
            (label_x + text_w + 5, label_y + 5),
            color,
            -1
        )
        cv2.addWeighted(overlay, 0.8, result, 0.2, 0, result)
        
        # Draw text
        cv2.putText(
            result,
            label,
            (label_x + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        
        # Draw center dot
        center_x = x + w // 2
        center_y = y + h // 2
        cv2.circle(result, (center_x, center_y), 3, color, -1)
    
    return result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("Testing Visualization Utilities...")
    
    # Sample data
    sample_data = {
        'Short': 45,
        'Missing_hole': 23,
        'Spur': 18,
        'Open_circuit': 12,
        'Mouse_bite': 8,
        'Spurious_copper': 5
    }
    
    # Test bar chart
    bar_svg = create_bar_chart_svg(sample_data)
    print(f"✓ Bar chart SVG: {len(bar_svg)} bytes")
    
    # Test pie chart
    pie_svg = create_pie_chart_svg(sample_data)
    print(f"✓ Pie chart SVG: {len(pie_svg)} bytes")
    
    # Test histogram
    sample_confidences = np.random.beta(5, 2, 100).tolist()
    hist_svg = create_confidence_histogram_svg(sample_confidences)
    print(f"✓ Histogram SVG: {len(hist_svg)} bytes")
    
    # Test stats card
    card_html = create_stats_card_html("Total Defects", "123", "🔴")
    print(f"✓ Stats card HTML: {len(card_html)} bytes")
    
    print("\n✓ All visualization tests passed!")
