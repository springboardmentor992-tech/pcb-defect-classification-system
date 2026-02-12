"""
Image Subtraction Module
========================

Performs image alignment and subtraction for PCB defect detection.
This module implements the core image processing pipeline for comparing
test PCB images against reference templates.

Features:
---------
- Multiple alignment methods (ORB, AKAZE, SIFT)
- Homography-based image registration
- Difference map computation
- Gaussian/Bilateral noise filtering
- Quality metrics for alignment evaluation

Author: PCB Defect Detection Team
Version: 1.0.0

Usage:
------
>>> from image_subtraction import ImageSubtractor
>>> subtractor = ImageSubtractor(alignment_method='ORB')
>>> result = subtractor.process_image_pair(template_path, test_path)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union, Dict, List
from dataclasses import dataclass, field
from enum import Enum
import time
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_utils import (
    load_image,
    save_image,
    convert_to_grayscale,
    apply_gaussian_blur,
    create_difference_map,
    visualize_comparison,
    normalize_image
)


# ============================================================
# ENUMS AND DATA CLASSES
# ============================================================

class AlignmentMethod(Enum):
    """Supported feature detection methods for image alignment."""
    ORB = "ORB"       # Oriented FAST and Rotated BRIEF - Fast, good for real-time
    AKAZE = "AKAZE"   # Accelerated-KAZE - Good balance of speed and accuracy
    SIFT = "SIFT"     # Scale-Invariant Feature Transform - Most accurate, slower


@dataclass
class AlignmentResult:
    """Results from image alignment operation."""
    aligned_image: np.ndarray
    homography_matrix: Optional[np.ndarray]
    num_matches: int
    num_good_matches: int
    inliers: int
    alignment_time: float
    method: str
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary (excluding large arrays)."""
        return {
            'num_matches': self.num_matches,
            'num_good_matches': self.num_good_matches,
            'inliers': self.inliers,
            'alignment_time': round(self.alignment_time, 4),
            'method': self.method,
            'success': self.success,
            'error_message': self.error_message
        }


@dataclass
class SubtractionResult:
    """Complete result from image subtraction pipeline."""
    template: np.ndarray
    test_original: np.ndarray
    test_aligned: np.ndarray
    difference_map: np.ndarray
    difference_map_enhanced: np.ndarray
    alignment_result: AlignmentResult
    processing_time: float
    quality_metrics: Dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'alignment': self.alignment_result.to_dict(),
            'processing_time': round(self.processing_time, 4),
            'quality_metrics': self.quality_metrics
        }


# ============================================================
# MAIN IMAGE SUBTRACTOR CLASS
# ============================================================

class ImageSubtractor:
    """
    Image alignment and subtraction for PCB defect detection.
    
    This class implements a complete pipeline for comparing test PCB
    images against reference templates using feature-based alignment
    and pixel-wise subtraction.
    
    Attributes
    ----------
    alignment_method : AlignmentMethod
        Feature detection method to use
    blur_kernel : int
        Gaussian blur kernel size for preprocessing
    match_ratio : float
        Ratio for Lowe's ratio test in feature matching
    min_matches : int
        Minimum good matches required for alignment
    
    Examples
    --------
    >>> subtractor = ImageSubtractor(alignment_method='ORB')
    >>> result = subtractor.process_image_pair(
    ...     'template.jpg', 'test.jpg',
    ...     output_dir='outputs'
    ... )
    >>> print(f"Alignment quality: {result.quality_metrics['alignment_score']:.2f}")
    """
    
    # Default configuration
    DEFAULT_BLUR_KERNEL = 5
    DEFAULT_MATCH_RATIO = 0.75
    DEFAULT_MIN_MATCHES = 10
    
    # ORB configuration
    ORB_FEATURES = 10000
    ORB_SCALE_FACTOR = 1.2
    ORB_LEVELS = 8
    
    # AKAZE configuration
    AKAZE_THRESHOLD = 0.001
    
    def __init__(
        self,
        alignment_method: Union[str, AlignmentMethod] = 'ORB',
        blur_kernel: int = DEFAULT_BLUR_KERNEL,
        match_ratio: float = DEFAULT_MATCH_RATIO,
        min_matches: int = DEFAULT_MIN_MATCHES,
        verbose: bool = True
    ):
        """
        Initialize the ImageSubtractor.
        
        Parameters
        ----------
        alignment_method : str or AlignmentMethod
            Feature detection method: 'ORB', 'AKAZE', or 'SIFT'
        blur_kernel : int
            Gaussian blur kernel size (odd number)
        match_ratio : float
            Ratio threshold for Lowe's ratio test (0.0-1.0)
        min_matches : int
            Minimum good matches required for valid alignment
        verbose : bool
            Print progress messages
        """
        # Set alignment method
        if isinstance(alignment_method, str):
            alignment_method = AlignmentMethod(alignment_method.upper())
        self.alignment_method = alignment_method
        
        # Configuration
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.match_ratio = match_ratio
        self.min_matches = min_matches
        self.verbose = verbose
        
        # Initialize feature detector
        self.detector = self._create_detector()
        
        # Initialize matcher
        if self.alignment_method == AlignmentMethod.SIFT:
            # FLANN-based matcher for SIFT (faster for large feature sets)
            index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            # BFMatcher for ORB/AKAZE (binary descriptors)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        if self.verbose:
            print(f"✓ ImageSubtractor initialized with {self.alignment_method.value}")
    
    def _create_detector(self):
        """Create the appropriate feature detector."""
        if self.alignment_method == AlignmentMethod.ORB:
            return cv2.ORB_create(
                nfeatures=self.ORB_FEATURES,
                scaleFactor=self.ORB_SCALE_FACTOR,
                nlevels=self.ORB_LEVELS,
                edgeThreshold=31,
                firstLevel=0,
                WTA_K=2,
                patchSize=31,
                fastThreshold=20
            )
        
        elif self.alignment_method == AlignmentMethod.AKAZE:
            return cv2.AKAZE_create(
                threshold=self.AKAZE_THRESHOLD
            )
        
        elif self.alignment_method == AlignmentMethod.SIFT:
            return cv2.SIFT_create(
                nfeatures=5000,
                contrastThreshold=0.04,
                edgeThreshold=10,
                sigma=1.6
            )
        
        else:
            raise ValueError(f"Unknown alignment method: {self.alignment_method}")
    
    # ========================================================
    # PREPROCESSING
    # ========================================================
    
    def preprocess_images(
        self,
        template_path: Union[str, Path],
        test_path: Union[str, Path],
        target_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess template and test images.
        
        Parameters
        ----------
        template_path : str or Path
            Path to template (reference) image
        test_path : str or Path
            Path to test (defect) image
        target_size : tuple, optional
            Resize images to (width, height)
            
        Returns
        -------
        tuple
            (template_gray, test_gray) preprocessed grayscale images
        """
        # Load images
        template = load_image(template_path, color_mode='color', target_size=target_size)
        test = load_image(test_path, color_mode='color', target_size=target_size)
        
        # Convert to grayscale for feature detection
        template_gray = convert_to_grayscale(template)
        test_gray = convert_to_grayscale(test)
        
        return template_gray, test_gray
    
    def preprocess_for_subtraction(
        self,
        image: np.ndarray,
        blur_kernel: Optional[int] = None
    ) -> np.ndarray:
        """
        Preprocess image for subtraction (noise reduction).
        
        Parameters
        ----------
        image : np.ndarray
            Input image
        blur_kernel : int, optional
            Blur kernel size (uses default if not specified)
            
        Returns
        -------
        np.ndarray
            Preprocessed image
        """
        kernel = blur_kernel or self.blur_kernel
        
        # Ensure grayscale
        if len(image.shape) == 3:
            image = convert_to_grayscale(image)
        
        # Apply Gaussian blur to reduce noise
        blurred = apply_gaussian_blur(image, kernel_size=kernel)
        
        return blurred
    
    # ========================================================
    # FEATURE DETECTION AND MATCHING
    # ========================================================
    
    def detect_and_match(
        self,
        template_gray: np.ndarray,
        test_gray: np.ndarray
    ) -> Tuple[List, List, List]:
        """
        Detect features and find matches between images.
        
        Parameters
        ----------
        template_gray : np.ndarray
            Grayscale template image
        test_gray : np.ndarray
            Grayscale test image
            
        Returns
        -------
        tuple
            (keypoints_template, keypoints_test, good_matches)
        """
        # Detect keypoints and compute descriptors
        kp1, desc1 = self.detector.detectAndCompute(template_gray, None)
        kp2, desc2 = self.detector.detectAndCompute(test_gray, None)
        
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return kp1, kp2, []
        
        # Convert descriptors for FLANN if using SIFT
        if self.alignment_method == AlignmentMethod.SIFT:
            desc1 = desc1.astype(np.float32)
            desc2 = desc2.astype(np.float32)
        
        # Match descriptors using kNN
        try:
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
        except cv2.error as e:
            if self.verbose:
                print(f"⚠ Matching error: {e}")
            return kp1, kp2, []
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.match_ratio * n.distance:
                    good_matches.append(m)
        
        # Sort by distance
        good_matches = sorted(good_matches, key=lambda x: x.distance)
        
        return kp1, kp2, good_matches
    
    def visualize_matches(
        self,
        template_gray: np.ndarray,
        test_gray: np.ndarray,
        kp1: List,
        kp2: List,
        matches: List,
        max_matches: int = 100,
        save_path: Optional[Union[str, Path]] = None
    ) -> np.ndarray:
        """
        Visualize feature matches between images.
        
        Parameters
        ----------
        template_gray : np.ndarray
            Template image
        test_gray : np.ndarray
            Test image
        kp1, kp2 : list
            Keypoints from each image
        matches : list
            Good matches
        max_matches : int
            Maximum matches to draw
        save_path : str or Path, optional
            Path to save visualization
            
        Returns
        -------
        np.ndarray
            Match visualization image
        """
        # Draw top matches
        matches_to_draw = matches[:max_matches]
        
        match_img = cv2.drawMatches(
            template_gray, kp1,
            test_gray, kp2,
            matches_to_draw,
            None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        if save_path:
            save_image(match_img, save_path)
            if self.verbose:
                print(f"✓ Match visualization saved to: {save_path}")
        
        return match_img
    
    # ========================================================
    # IMAGE ALIGNMENT
    # ========================================================
    
    def compute_homography(
        self,
        kp1: List,
        kp2: List,
        matches: List
    ) -> Tuple[Optional[np.ndarray], int]:
        """
        Compute homography matrix from matched keypoints.
        
        Parameters
        ----------
        kp1, kp2 : list
            Keypoints from template and test images
        matches : list
            Good matches
            
        Returns
        -------
        tuple
            (homography_matrix, num_inliers)
        """
        if len(matches) < self.min_matches:
            return None, 0
        
        # Extract matched point coordinates
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Compute homography using RANSAC
        H, mask = cv2.findHomography(
            dst_pts, src_pts,  # Note: dst -> src to align test to template
            cv2.RANSAC,
            ransacReprojThreshold=5.0,
            maxIters=2000,
            confidence=0.995
        )
        
        # Count inliers
        num_inliers = int(mask.sum()) if mask is not None else 0
        
        return H, num_inliers
    
    def align_images(
        self,
        template_gray: np.ndarray,
        test_gray: np.ndarray,
        test_color: Optional[np.ndarray] = None
    ) -> AlignmentResult:
        """
        Align test image to template using feature-based registration.
        
        Parameters
        ----------
        template_gray : np.ndarray
            Grayscale template image
        test_gray : np.ndarray
            Grayscale test image
        test_color : np.ndarray, optional
            Color version of test image to also align
            
        Returns
        -------
        AlignmentResult
            Complete alignment results
        """
        start_time = time.time()
        
        # Detect and match features
        kp1, kp2, good_matches = self.detect_and_match(template_gray, test_gray)
        
        num_matches = len(kp1) + len(kp2)
        num_good_matches = len(good_matches)
        
        if self.verbose:
            print(f"   Keypoints: template={len(kp1)}, test={len(kp2)}")
            print(f"   Good matches: {num_good_matches}")
        
        # Check minimum matches
        if num_good_matches < self.min_matches:
            elapsed = time.time() - start_time
            return AlignmentResult(
                aligned_image=test_gray,
                homography_matrix=None,
                num_matches=num_matches,
                num_good_matches=num_good_matches,
                inliers=0,
                alignment_time=elapsed,
                method=self.alignment_method.value,
                success=False,
                error_message=f"Insufficient matches: {num_good_matches} < {self.min_matches}"
            )
        
        # Compute homography
        H, num_inliers = self.compute_homography(kp1, kp2, good_matches)
        
        if H is None:
            elapsed = time.time() - start_time
            return AlignmentResult(
                aligned_image=test_gray,
                homography_matrix=None,
                num_matches=num_matches,
                num_good_matches=num_good_matches,
                inliers=0,
                alignment_time=elapsed,
                method=self.alignment_method.value,
                success=False,
                error_message="Failed to compute homography"
            )
        
        if self.verbose:
            print(f"   Inliers: {num_inliers}")
        
        # Warp test image to align with template
        h, w = template_gray.shape[:2]
        aligned_gray = cv2.warpPerspective(
            test_gray, H, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        # Also warp color image if provided
        if test_color is not None:
            aligned_color = cv2.warpPerspective(
                test_color, H, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )
            aligned_image = aligned_color
        else:
            aligned_image = aligned_gray
        
        elapsed = time.time() - start_time
        
        return AlignmentResult(
            aligned_image=aligned_image,
            homography_matrix=H,
            num_matches=num_matches,
            num_good_matches=num_good_matches,
            inliers=num_inliers,
            alignment_time=elapsed,
            method=self.alignment_method.value,
            success=True
        )
    
    # ========================================================
    # IMAGE SUBTRACTION
    # ========================================================
    
    def compute_difference(
        self,
        template: np.ndarray,
        aligned_test: np.ndarray,
        blur_kernel: Optional[int] = None,
        enhance: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute difference map between template and aligned test image.
        
        Parameters
        ----------
        template : np.ndarray
            Template image
        aligned_test : np.ndarray
            Aligned test image
        blur_kernel : int, optional
            Blur kernel for preprocessing
        enhance : bool
            Apply contrast enhancement to difference map
            
        Returns
        -------
        tuple
            (difference_map, enhanced_difference_map)
        """
        kernel = blur_kernel or self.blur_kernel
        
        # Preprocess for subtraction
        template_prep = self.preprocess_for_subtraction(template, kernel)
        test_prep = self.preprocess_for_subtraction(aligned_test, kernel)
        
        # Compute absolute difference
        diff = create_difference_map(template_prep, test_prep, method='absolute')
        
        # Enhanced version with contrast stretching
        if enhance:
            # Histogram equalization for better contrast
            diff_enhanced = cv2.equalizeHist(diff)
            
            # Optional: Apply CLAHE for better local contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            diff_enhanced = clahe.apply(diff)
        else:
            diff_enhanced = diff
        
        return diff, diff_enhanced
    
    def calculate_quality_metrics(
        self,
        template: np.ndarray,
        aligned: np.ndarray,
        difference: np.ndarray,
        alignment_result: AlignmentResult
    ) -> Dict:
        """
        Calculate quality metrics for alignment and subtraction.
        
        Parameters
        ----------
        template : np.ndarray
            Template image
        aligned : np.ndarray
            Aligned test image
        difference : np.ndarray
            Difference map
        alignment_result : AlignmentResult
            Alignment results
            
        Returns
        -------
        dict
            Quality metrics
        """
        # Ensure grayscale
        if len(template.shape) == 3:
            template = convert_to_grayscale(template)
        if len(aligned.shape) == 3:
            aligned = convert_to_grayscale(aligned)
        
        # Structural Similarity Index (SSIM-inspired metric)
        # Higher = better alignment
        template_f = template.astype(np.float64)
        aligned_f = aligned.astype(np.float64)
        
        # Mean values
        mu1 = template_f.mean()
        mu2 = aligned_f.mean()
        
        # Variances and covariance
        var1 = template_f.var()
        var2 = aligned_f.var()
        covar = ((template_f - mu1) * (aligned_f - mu2)).mean()
        
        # SSIM constants
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        ssim = ((2 * mu1 * mu2 + C1) * (2 * covar + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (var1 + var2 + C2))
        
        # Normalized Cross-Correlation
        ncc = covar / (np.sqrt(var1 * var2) + 1e-8)
        
        # Difference map statistics
        diff_mean = float(difference.mean())
        diff_std = float(difference.std())
        diff_max = float(difference.max())
        
        # Alignment score (0-100)
        # Based on inlier ratio and SSIM
        inlier_ratio = alignment_result.inliers / max(alignment_result.num_good_matches, 1)
        alignment_score = 0.5 * (ssim * 100) + 0.5 * (inlier_ratio * 100)
        
        return {
            'ssim': round(float(ssim), 4),
            'ncc': round(float(ncc), 4),
            'alignment_score': round(float(alignment_score), 2),
            'inlier_ratio': round(float(inlier_ratio), 4),
            'difference_mean': round(diff_mean, 2),
            'difference_std': round(diff_std, 2),
            'difference_max': round(diff_max, 2)
        }
    
    # ========================================================
    # COMPLETE PIPELINE
    # ========================================================
    
    def process_image_pair(
        self,
        template_path: Union[str, Path],
        test_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        save_intermediate: bool = True,
        visualize: bool = False,
        target_size: Optional[Tuple[int, int]] = None,
        output_prefix: Optional[str] = None
    ) -> SubtractionResult:
        """
        Process a template-test pair through the complete pipeline.
        
        This is the main entry point for processing image pairs.
        
        Parameters
        ----------
        template_path : str or Path
            Path to template (reference) image
        test_path : str or Path
            Path to test (defect) image
        output_dir : str or Path, optional
            Directory to save outputs
        save_intermediate : bool
            Save intermediate results (aligned, diff, etc.)
        visualize : bool
            Display visualizations
        target_size : tuple, optional
            Resize images to (width, height)
        output_prefix : str, optional
            Prefix for output filenames
            
        Returns
        -------
        SubtractionResult
            Complete processing results
            
        Examples
        --------
        >>> result = subtractor.process_image_pair(
        ...     'template.jpg', 'test.jpg',
        ...     output_dir='outputs',
        ...     visualize=True
        ... )
        >>> print(f"Alignment score: {result.quality_metrics['alignment_score']}")
        """
        start_time = time.time()
        
        template_path = Path(template_path)
        test_path = Path(test_path)
        
        if self.verbose:
            print(f"\n{'─'*60}")
            print(f"Processing: {test_path.name}")
            print(f"Template:   {template_path.name}")
            print(f"Method:     {self.alignment_method.value}")
            print(f"{'─'*60}")
        
        # Load images (color)
        template_color = load_image(template_path, color_mode='color', target_size=target_size)
        test_color = load_image(test_path, color_mode='color', target_size=target_size)
        
        # Convert to grayscale for processing
        template_gray = convert_to_grayscale(template_color)
        test_gray = convert_to_grayscale(test_color)
        
        if self.verbose:
            print(f"   Image size: {template_gray.shape[1]}x{template_gray.shape[0]}")
        
        # Align images
        if self.verbose:
            print(f"\n🔄 Aligning images...")
        
        alignment_result = self.align_images(
            template_gray, test_gray, test_color
        )
        
        if not alignment_result.success:
            if self.verbose:
                print(f"   ⚠ Alignment failed: {alignment_result.error_message}")
            # Continue with unaligned image
            aligned_gray = test_gray
            aligned_color = test_color
        else:
            if self.verbose:
                print(f"   ✓ Alignment successful ({alignment_result.alignment_time:.3f}s)")
            aligned_color = alignment_result.aligned_image
            aligned_gray = convert_to_grayscale(aligned_color)
        
        # Compute difference
        if self.verbose:
            print(f"\n📊 Computing difference map...")
        
        diff_map, diff_enhanced = self.compute_difference(
            template_gray, aligned_gray
        )
        
        if self.verbose:
            print(f"   ✓ Difference range: {diff_map.min()}-{diff_map.max()}")
        
        # Calculate quality metrics
        if self.verbose:
            print(f"\n📈 Calculating quality metrics...")
        
        metrics = self.calculate_quality_metrics(
            template_gray, aligned_gray, diff_map, alignment_result
        )
        
        if self.verbose:
            print(f"   SSIM: {metrics['ssim']:.4f}")
            print(f"   Alignment Score: {metrics['alignment_score']:.2f}/100")
        
        # Create result
        elapsed = time.time() - start_time
        
        result = SubtractionResult(
            template=template_color,
            test_original=test_color,
            test_aligned=aligned_color,
            difference_map=diff_map,
            difference_map_enhanced=diff_enhanced,
            alignment_result=alignment_result,
            processing_time=elapsed,
            quality_metrics=metrics
        )
        
        # Save outputs
        if output_dir:
            self._save_outputs(result, output_dir, test_path.stem, 
                              output_prefix, save_intermediate)
        
        # Visualize
        if visualize:
            self._visualize_result(result)
        
        if self.verbose:
            print(f"\n✓ Processing complete ({elapsed:.3f}s)")
        
        return result
    
    def _save_outputs(
        self,
        result: SubtractionResult,
        output_dir: Union[str, Path],
        test_name: str,
        prefix: Optional[str],
        save_intermediate: bool
    ) -> None:
        """Save processing outputs to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        name_prefix = f"{prefix}_" if prefix else ""
        
        if save_intermediate:
            # Save aligned image
            aligned_path = output_dir / f"{name_prefix}{test_name}_aligned.png"
            save_image(result.test_aligned, aligned_path)
            
            # Save difference map
            diff_path = output_dir / f"{name_prefix}{test_name}_diff.png"
            save_image(result.difference_map, diff_path)
            
            # Save enhanced difference
            diff_enh_path = output_dir / f"{name_prefix}{test_name}_diff_enhanced.png"
            save_image(result.difference_map_enhanced, diff_enh_path)
        
        # Save metadata
        meta_path = output_dir / f"{name_prefix}{test_name}_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        if self.verbose:
            print(f"   ✓ Outputs saved to: {output_dir}")
    
    def _visualize_result(self, result: SubtractionResult) -> None:
        """Display processing results."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Row 1: Template, Test Original, Test Aligned
        axes[0, 0].imshow(cv2.cvtColor(result.template, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Template (Reference)', fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(cv2.cvtColor(result.test_original, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Test (Original)', fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(cv2.cvtColor(result.test_aligned, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title('Test (Aligned)', fontweight='bold')
        axes[0, 2].axis('off')
        
        # Row 2: Difference maps and metrics
        axes[1, 0].imshow(result.difference_map, cmap='gray')
        axes[1, 0].set_title('Difference Map', fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(result.difference_map_enhanced, cmap='hot')
        axes[1, 1].set_title('Enhanced Difference (Hot)', fontweight='bold')
        axes[1, 1].axis('off')
        
        # Metrics panel
        axes[1, 2].axis('off')
        metrics_text = [
            "QUALITY METRICS",
            "─" * 30,
            f"Method: {result.alignment_result.method}",
            f"Matches: {result.alignment_result.num_good_matches}",
            f"Inliers: {result.alignment_result.inliers}",
            f"",
            f"SSIM: {result.quality_metrics['ssim']:.4f}",
            f"NCC: {result.quality_metrics['ncc']:.4f}",
            f"Alignment Score: {result.quality_metrics['alignment_score']:.1f}/100",
            f"",
            f"Diff Mean: {result.quality_metrics['difference_mean']:.1f}",
            f"Diff Std: {result.quality_metrics['difference_std']:.1f}",
            f"",
            f"Time: {result.processing_time:.3f}s"
        ]
        axes[1, 2].text(
            0.1, 0.9, '\n'.join(metrics_text),
            transform=axes[1, 2].transAxes,
            fontfamily='monospace',
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8)
        )
        
        plt.suptitle('Image Subtraction Pipeline Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    # ========================================================
    # BATCH PROCESSING
    # ========================================================
    
    def process_batch(
        self,
        pairs: List[Dict],
        output_dir: Union[str, Path],
        save_intermediate: bool = True,
        visualize: bool = False,
        target_size: Optional[Tuple[int, int]] = None
    ) -> List[Dict]:
        """
        Process multiple template-test pairs.
        
        Parameters
        ----------
        pairs : list of dict
            List of pair dictionaries with 'template' and 'test' paths
        output_dir : str or Path
            Output directory
        save_intermediate : bool
            Save intermediate results
        visualize : bool
            Display visualizations for each pair
        target_size : tuple, optional
            Resize images
            
        Returns
        -------
        list of dict
            Processing results for each pair
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        total = len(pairs)
        successful = 0
        failed = 0
        
        print(f"\n{'='*60}")
        print(f"    BATCH PROCESSING: {total} pairs")
        print(f"{'='*60}")
        
        for i, pair in enumerate(pairs):
            pair_id = pair.get('id', i + 1)
            template_path = pair['template']
            test_path = pair['test']
            defect_class = pair.get('defect_class', 'unknown')
            
            print(f"\n[{i+1}/{total}] Processing pair {pair_id} ({defect_class})...")
            
            try:
                result = self.process_image_pair(
                    template_path=template_path,
                    test_path=test_path,
                    output_dir=output_dir,
                    save_intermediate=save_intermediate,
                    visualize=visualize,
                    target_size=target_size,
                    output_prefix=f"pair_{pair_id:03d}"
                )
                
                results.append({
                    'pair_id': pair_id,
                    'defect_class': defect_class,
                    'status': 'success',
                    'alignment_success': result.alignment_result.success,
                    'quality_metrics': result.quality_metrics,
                    'processing_time': result.processing_time
                })
                
                if result.alignment_result.success:
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                results.append({
                    'pair_id': pair_id,
                    'defect_class': defect_class,
                    'status': 'failed',
                    'error': str(e)
                })
                failed += 1
                print(f"   ❌ Error: {e}")
        
        # Save batch summary
        summary = {
            'total_pairs': total,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total * 100 if total > 0 else 0,
            'alignment_method': self.alignment_method.value,
            'results': results
        }
        
        summary_path = output_dir / 'batch_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"    BATCH PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"   Successful: {successful}/{total}")
        print(f"   Failed: {failed}/{total}")
        print(f"   Success Rate: {summary['success_rate']:.1f}%")
        print(f"   Summary saved to: {summary_path}")
        print(f"{'='*60}\n")
        
        return results
    
    # ========================================================
    # METHOD COMPARISON
    # ========================================================
    
    @staticmethod
    def compare_methods(
        template_path: Union[str, Path],
        test_path: Union[str, Path],
        methods: List[str] = None,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict:
        """
        Compare different alignment methods on a single pair.
        
        Parameters
        ----------
        template_path : str or Path
            Path to template image
        test_path : str or Path
            Path to test image
        methods : list, optional
            Methods to compare (default: ['ORB', 'AKAZE'])
        output_dir : str or Path, optional
            Directory for comparison outputs
            
        Returns
        -------
        dict
            Comparison results for each method
        """
        if methods is None:
            methods = ['ORB', 'AKAZE']
        
        print(f"\n{'='*60}")
        print(f"    COMPARING ALIGNMENT METHODS")
        print(f"{'='*60}")
        
        results = {}
        
        for method in methods:
            print(f"\n🔬 Testing {method}...")
            
            subtractor = ImageSubtractor(
                alignment_method=method,
                verbose=False
            )
            
            try:
                result = subtractor.process_image_pair(
                    template_path,
                    test_path,
                    output_dir=output_dir,
                    save_intermediate=output_dir is not None,
                    output_prefix=method.lower()
                )
                
                results[method] = {
                    'success': result.alignment_result.success,
                    'matches': result.alignment_result.num_good_matches,
                    'inliers': result.alignment_result.inliers,
                    'ssim': result.quality_metrics['ssim'],
                    'alignment_score': result.quality_metrics['alignment_score'],
                    'time': result.processing_time
                }
                
                print(f"   ✓ Score: {result.quality_metrics['alignment_score']:.1f}, "
                      f"Matches: {result.alignment_result.num_good_matches}, "
                      f"Time: {result.processing_time:.3f}s")
                
            except Exception as e:
                results[method] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"   ❌ Failed: {e}")
        
        # Determine best method
        best_method = None
        best_score = -1
        
        for method, data in results.items():
            if data.get('success', False):
                score = data.get('alignment_score', 0)
                if score > best_score:
                    best_score = score
                    best_method = method
        
        print(f"\n🏆 Best Method: {best_method} (score: {best_score:.1f})")
        
        results['best_method'] = best_method
        results['best_score'] = best_score
        
        return results


# ============================================================
# MAIN / CLI
# ============================================================

def main():
    """Command-line interface for image subtraction."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='PCB Image Alignment and Subtraction',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'template',
        type=str,
        help='Path to template (reference) image'
    )
    
    parser.add_argument(
        'test',
        type=str,
        help='Path to test image'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='outputs',
        help='Output directory (default: outputs)'
    )
    
    parser.add_argument(
        '-m', '--method',
        type=str,
        choices=['ORB', 'AKAZE', 'SIFT'],
        default='ORB',
        help='Alignment method (default: ORB)'
    )
    
    parser.add_argument(
        '-v', '--visualize',
        action='store_true',
        help='Display visualization'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare all alignment methods'
    )
    
    args = parser.parse_args()
    
    if args.compare:
        results = ImageSubtractor.compare_methods(
            args.template,
            args.test,
            output_dir=args.output
        )
    else:
        subtractor = ImageSubtractor(alignment_method=args.method)
        result = subtractor.process_image_pair(
            args.template,
            args.test,
            output_dir=args.output,
            visualize=args.visualize
        )
        
        print(f"\nAlignment Score: {result.quality_metrics['alignment_score']:.2f}/100")
    
    return 0


if __name__ == '__main__':
    exit(main())
