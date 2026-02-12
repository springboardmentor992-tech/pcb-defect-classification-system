"""
Image Utilities Module
======================

Provides core image processing functions for PCB defect detection.

This module contains all the fundamental image operations needed
throughout the PCB defect detection pipeline.

Author: PCB Defect Detection Team
Version: 1.0.0
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union, List
import matplotlib.pyplot as plt


# ============================================================
# IMAGE I/O OPERATIONS
# ============================================================

def load_image(
    image_path: Union[str, Path],
    color_mode: str = 'color',
    target_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Load an image from disk with optional color conversion and resizing.
    
    Parameters
    ----------
    image_path : str or Path
        Path to the image file
    color_mode : str, optional
        Color mode: 'color' (BGR), 'rgb', 'gray', 'grayscale'
        Default is 'color'
    target_size : tuple, optional
        Target size (width, height) for resizing
        
    Returns
    -------
    np.ndarray
        Loaded image as numpy array
        
    Raises
    ------
    FileNotFoundError
        If image file does not exist
    ValueError
        If image cannot be loaded
        
    Examples
    --------
    >>> img = load_image('path/to/image.jpg')
    >>> gray_img = load_image('path/to/image.jpg', color_mode='gray')
    >>> resized = load_image('path/to/image.jpg', target_size=(640, 640))
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Determine OpenCV flag based on color mode
    if color_mode in ['gray', 'grayscale']:
        cv2_flag = cv2.IMREAD_GRAYSCALE
    else:
        cv2_flag = cv2.IMREAD_COLOR
    
    # Load image
    image = cv2.imread(str(image_path), cv2_flag)
    
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert BGR to RGB if requested
    if color_mode == 'rgb' and len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize if target size specified
    if target_size is not None:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    return image


def save_image(
    image: np.ndarray,
    output_path: Union[str, Path],
    create_dirs: bool = True,
    quality: int = 95
) -> bool:
    """
    Save an image to disk.
    
    Parameters
    ----------
    image : np.ndarray
        Image to save
    output_path : str or Path
        Output file path
    create_dirs : bool, optional
        Create parent directories if they don't exist
        Default is True
    quality : int, optional
        JPEG quality (0-100), default is 95
        
    Returns
    -------
    bool
        True if save was successful
        
    Examples
    --------
    >>> save_image(img, 'outputs/result.jpg')
    >>> save_image(img, 'outputs/high_quality.jpg', quality=100)
    """
    output_path = Path(output_path)
    
    if create_dirs:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set compression parameters based on extension
    ext = output_path.suffix.lower()
    if ext in ['.jpg', '.jpeg']:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == '.png':
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    else:
        params = []
    
    return cv2.imwrite(str(output_path), image, params)


# ============================================================
# IMAGE TRANSFORMATION OPERATIONS
# ============================================================

def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    interpolation: str = 'area'
) -> np.ndarray:
    """
    Resize an image to target dimensions.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    target_size : tuple
        Target size as (width, height)
    interpolation : str, optional
        Interpolation method: 'linear', 'cubic', 'area', 'nearest'
        Default is 'area' (best for downscaling)
        
    Returns
    -------
    np.ndarray
        Resized image
    """
    interpolation_map = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'area': cv2.INTER_AREA,
        'lanczos': cv2.INTER_LANCZOS4
    }
    
    interp = interpolation_map.get(interpolation.lower(), cv2.INTER_AREA)
    return cv2.resize(image, target_size, interpolation=interp)


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale.
    
    Parameters
    ----------
    image : np.ndarray
        Input image (BGR or RGB)
        
    Returns
    -------
    np.ndarray
        Grayscale image
    """
    if len(image.shape) == 2:
        return image  # Already grayscale
    
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Unexpected number of channels: {image.shape[2]}")


def normalize_image(
    image: np.ndarray,
    method: str = 'minmax'
) -> np.ndarray:
    """
    Normalize image pixel values.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    method : str, optional
        Normalization method: 'minmax' (0-255), 'standard' (z-score),
        'unit' (0-1)
        
    Returns
    -------
    np.ndarray
        Normalized image
    """
    image = image.astype(np.float32)
    
    if method == 'minmax':
        min_val = image.min()
        max_val = image.max()
        if max_val - min_val > 0:
            image = (image - min_val) / (max_val - min_val) * 255
        return image.astype(np.uint8)
    
    elif method == 'standard':
        mean = image.mean()
        std = image.std()
        if std > 0:
            image = (image - mean) / std
        return image
    
    elif method == 'unit':
        return image / 255.0
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


# ============================================================
# IMAGE FILTERING OPERATIONS
# ============================================================

def apply_gaussian_blur(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 0
) -> np.ndarray:
    """
    Apply Gaussian blur to reduce noise.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    kernel_size : int, optional
        Size of the Gaussian kernel (must be odd)
        Default is 5
    sigma : float, optional
        Standard deviation. If 0, calculated from kernel size.
        Default is 0
        
    Returns
    -------
    np.ndarray
        Blurred image
        
    Notes
    -----
    Gaussian blur is essential before subtraction to reduce
    sensor noise and minor variations.
    
    Examples
    --------
    >>> blurred = apply_gaussian_blur(img, kernel_size=5)
    >>> heavily_blurred = apply_gaussian_blur(img, kernel_size=9)
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def apply_bilateral_filter(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75
) -> np.ndarray:
    """
    Apply bilateral filter for edge-preserving smoothing.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    d : int, optional
        Diameter of each pixel neighborhood
    sigma_color : float, optional
        Filter sigma in the color space
    sigma_space : float, optional
        Filter sigma in the coordinate space
        
    Returns
    -------
    np.ndarray
        Filtered image
        
    Notes
    -----
    Bilateral filtering is slower but preserves edges better
    than Gaussian blur, making it useful when edge sharpness
    is important.
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def apply_median_blur(
    image: np.ndarray,
    kernel_size: int = 5
) -> np.ndarray:
    """
    Apply median blur to remove salt-and-pepper noise.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    kernel_size : int, optional
        Size of the kernel (must be odd)
        
    Returns
    -------
    np.ndarray
        Filtered image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(image, kernel_size)


# ============================================================
# IMAGE SUBTRACTION OPERATIONS
# ============================================================

def create_difference_map(
    image1: np.ndarray,
    image2: np.ndarray,
    method: str = 'absolute'
) -> np.ndarray:
    """
    Create a difference map between two images.
    
    Parameters
    ----------
    image1 : np.ndarray
        First image (typically template/reference)
    image2 : np.ndarray
        Second image (typically test image)
    method : str, optional
        Difference method:
        - 'absolute': |image1 - image2| (default)
        - 'squared': (image1 - image2)^2
        - 'signed': image1 - image2 (can be negative)
        
    Returns
    -------
    np.ndarray
        Difference map
        
    Raises
    ------
    ValueError
        If images have different shapes
        
    Examples
    --------
    >>> diff = create_difference_map(template, test)
    >>> diff_squared = create_difference_map(template, test, method='squared')
    """
    if image1.shape != image2.shape:
        raise ValueError(
            f"Image shapes must match. Got {image1.shape} and {image2.shape}"
        )
    
    # Convert to grayscale if color images
    if len(image1.shape) == 3:
        image1 = convert_to_grayscale(image1)
    if len(image2.shape) == 3:
        image2 = convert_to_grayscale(image2)
    
    # Calculate difference
    if method == 'absolute':
        diff = cv2.absdiff(image1, image2)
    elif method == 'squared':
        diff = np.square(image1.astype(np.float32) - image2.astype(np.float32))
        diff = normalize_image(diff)
    elif method == 'signed':
        diff = image1.astype(np.float32) - image2.astype(np.float32)
    else:
        raise ValueError(f"Unknown difference method: {method}")
    
    return diff


# ============================================================
# MORPHOLOGICAL OPERATIONS
# ============================================================

def apply_morphological_operations(
    image: np.ndarray,
    operation: str = 'open',
    kernel_size: int = 3,
    kernel_shape: str = 'rect',
    iterations: int = 1
) -> np.ndarray:
    """
    Apply morphological operations to clean up binary masks.
    
    Parameters
    ----------
    image : np.ndarray
        Input binary image/mask
    operation : str, optional
        Morphological operation:
        - 'erode': Shrink white regions
        - 'dilate': Expand white regions
        - 'open': Erosion then dilation (removes noise)
        - 'close': Dilation then erosion (fills holes)
        - 'gradient': Dilation - erosion (edge detection)
        - 'tophat': Original - opening (bright spots)
        - 'blackhat': Closing - original (dark spots)
    kernel_size : int, optional
        Size of the structuring element
    kernel_shape : str, optional
        Shape of kernel: 'rect', 'ellipse', 'cross'
    iterations : int, optional
        Number of times to apply operation
        
    Returns
    -------
    np.ndarray
        Processed image
        
    Examples
    --------
    >>> # Remove small noise
    >>> clean = apply_morphological_operations(mask, 'open', kernel_size=3)
    >>> # Fill small holes
    >>> filled = apply_morphological_operations(clean, 'close', kernel_size=5)
    """
    # Create structuring element
    shape_map = {
        'rect': cv2.MORPH_RECT,
        'ellipse': cv2.MORPH_ELLIPSE,
        'cross': cv2.MORPH_CROSS
    }
    
    shape = shape_map.get(kernel_shape.lower(), cv2.MORPH_RECT)
    kernel = cv2.getStructuringElement(shape, (kernel_size, kernel_size))
    
    # Operation mapping
    op_map = {
        'erode': cv2.MORPH_ERODE,
        'dilate': cv2.MORPH_DILATE,
        'open': cv2.MORPH_OPEN,
        'close': cv2.MORPH_CLOSE,
        'gradient': cv2.MORPH_GRADIENT,
        'tophat': cv2.MORPH_TOPHAT,
        'blackhat': cv2.MORPH_BLACKHAT
    }
    
    morph_op = op_map.get(operation.lower())
    
    if morph_op is None:
        raise ValueError(f"Unknown morphological operation: {operation}")
    
    return cv2.morphologyEx(image, morph_op, kernel, iterations=iterations)


def clean_binary_mask(
    mask: np.ndarray,
    min_area: int = 100,
    open_kernel: int = 3,
    close_kernel: int = 5
) -> np.ndarray:
    """
    Clean a binary mask by removing noise and filling holes.
    
    This is a convenience function that combines common cleaning
    operations used in defect detection.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask to clean
    min_area : int, optional
        Minimum contour area to keep (removes small noise)
    open_kernel : int, optional
        Kernel size for opening operation
    close_kernel : int, optional
        Kernel size for closing operation
        
    Returns
    -------
    np.ndarray
        Cleaned binary mask
    """
    # Step 1: Opening to remove small noise
    cleaned = apply_morphological_operations(
        mask, 'open', kernel_size=open_kernel
    )
    
    # Step 2: Closing to fill small holes
    cleaned = apply_morphological_operations(
        cleaned, 'close', kernel_size=close_kernel
    )
    
    # Step 3: Remove small contours
    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Create mask for large contours only
    result = np.zeros_like(mask)
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(result, [contour], -1, 255, -1)
    
    return result


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def visualize_comparison(
    images: List[np.ndarray],
    titles: List[str],
    figsize: Tuple[int, int] = (15, 5),
    cmap: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> None:
    """
    Display multiple images side by side for comparison.
    
    Parameters
    ----------
    images : list of np.ndarray
        Images to display
    titles : list of str
        Title for each image
    figsize : tuple, optional
        Figure size
    cmap : str, optional
        Colormap for grayscale images (e.g., 'gray', 'hot')
    save_path : str or Path, optional
        Path to save the figure
    show : bool, optional
        Whether to display the figure
        
    Examples
    --------
    >>> visualize_comparison(
    ...     [template, test, diff, mask],
    ...     ['Template', 'Test', 'Difference', 'Mask'],
    ...     save_path='comparison.png'
    ... )
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    
    if n == 1:
        axes = [axes]
    
    for i, (img, title) in enumerate(zip(images, titles)):
        # Convert BGR to RGB for display if color image
        if len(img.shape) == 3:
            display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i].imshow(display_img)
        else:
            axes[i].imshow(img, cmap=cmap or 'gray')
        
        axes[i].set_title(title, fontsize=12, fontweight='bold')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Figure saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def create_overlay(
    background: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create an overlay visualization of a mask on an image.
    
    Parameters
    ----------
    background : np.ndarray
        Background image
    mask : np.ndarray
        Binary mask to overlay
    color : tuple, optional
        BGR color for the overlay (default: red)
    alpha : float, optional
        Transparency of overlay (0-1)
        
    Returns
    -------
    np.ndarray
        Image with mask overlay
    """
    # Ensure background is color
    if len(background.shape) == 2:
        background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    
    # Create colored overlay
    overlay = background.copy()
    overlay[mask > 0] = color
    
    # Blend
    result = cv2.addWeighted(background, 1 - alpha, overlay, alpha, 0)
    
    return result


def draw_bounding_boxes(
    image: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    labels: Optional[List[str]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.7
) -> np.ndarray:
    """
    Draw bounding boxes on an image.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    boxes : list of tuples
        Bounding boxes as (xmin, ymin, xmax, ymax)
    labels : list of str, optional
        Labels for each box
    color : tuple, optional
        BGR color for boxes
    thickness : int, optional
        Line thickness
    font_scale : float, optional
        Font scale for labels
        
    Returns
    -------
    np.ndarray
        Image with drawn boxes
    """
    result = image.copy()
    
    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box
        
        # Draw rectangle
        cv2.rectangle(result, (xmin, ymin), (xmax, ymax), color, thickness)
        
        # Draw label if provided
        if labels and i < len(labels):
            label = labels[i]
            # Get text size for background
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            # Draw background rectangle
            cv2.rectangle(
                result,
                (xmin, ymin - text_h - 10),
                (xmin + text_w + 5, ymin),
                color,
                -1
            )
            # Draw text
            cv2.putText(
                result,
                label,
                (xmin + 2, ymin - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness
            )
    
    return result


# ============================================================
# HISTOGRAM AND ANALYSIS FUNCTIONS
# ============================================================

def calculate_histogram(
    image: np.ndarray,
    bins: int = 256,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculate histogram of an image.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    bins : int, optional
        Number of histogram bins
    mask : np.ndarray, optional
        Binary mask to limit histogram calculation
        
    Returns
    -------
    np.ndarray
        Histogram values
    """
    if len(image.shape) == 3:
        image = convert_to_grayscale(image)
    
    hist = cv2.calcHist([image], [0], mask, [bins], [0, 256])
    return hist.flatten()


def analyze_image_statistics(image: np.ndarray) -> dict:
    """
    Calculate various statistics for an image.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
        
    Returns
    -------
    dict
        Dictionary containing image statistics
    """
    if len(image.shape) == 3:
        gray = convert_to_grayscale(image)
    else:
        gray = image
    
    return {
        'shape': image.shape,
        'dtype': str(image.dtype),
        'min': float(gray.min()),
        'max': float(gray.max()),
        'mean': float(gray.mean()),
        'std': float(gray.std()),
        'median': float(np.median(gray))
    }


# ============================================================
# MAIN / TESTING
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("IMAGE UTILITIES MODULE TEST")
    print("="*60)
    
    # Test with a sample image (if exists)
    test_path = Path(__file__).parent.parent.parent.parent / "PCB_DATASET" / "PCB_USED" / "01.JPG"
    
    if test_path.exists():
        print(f"\n✓ Found test image: {test_path.name}")
        
        # Test loading
        img = load_image(test_path)
        print(f"✓ Loaded image: {img.shape}")
        
        # Test grayscale conversion
        gray = convert_to_grayscale(img)
        print(f"✓ Converted to grayscale: {gray.shape}")
        
        # Test blur
        blurred = apply_gaussian_blur(gray, kernel_size=5)
        print(f"✓ Applied Gaussian blur")
        
        # Test statistics
        stats = analyze_image_statistics(img)
        print(f"✓ Image statistics: mean={stats['mean']:.1f}, std={stats['std']:.1f}")
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
    else:
        print(f"\n⚠ Test image not found at: {test_path}")
        print("Please ensure PCB_DATASET is in the parent directory.")
