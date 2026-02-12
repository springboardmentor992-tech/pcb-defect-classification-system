"""
PCB Defect Detection - API Client
==================================

Client library for communicating with the FastAPI backend server.

This client provides a clean interface for the Streamlit frontend
to interact with the detection API.

Features:
---------
- Async and sync methods for flexibility
- Automatic retry on connection errors
- Image encoding/decoding utilities
- Type-safe response handling
- Connection health checks

Usage:
------
    >>> from src.api.client import PCBDetectionClient
    >>> client = PCBDetectionClient()
    >>> result = client.detect(template_image, test_image)

Author: PCB Defect Detection Team
Version: 1.0.0
"""

import io
import base64
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging

import numpy as np
import cv2
from PIL import Image
import requests
from requests.exceptions import ConnectionError, Timeout, RequestException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class APIHealthStatus:
    """Health check result."""
    healthy: bool
    model_loaded: bool
    timestamp: str
    version: str
    error: Optional[str] = None


@dataclass
class DefectInfo:
    """Information about a detected defect."""
    index: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    center: Tuple[int, int]
    area: int
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DefectInfo':
        """Create from API response dict."""
        return cls(
            index=data['index'],
            class_name=data['class'],
            confidence=data['confidence'],
            bbox=(
                data['bbox']['x'],
                data['bbox']['y'],
                data['bbox']['width'],
                data['bbox']['height']
            ),
            center=(data['center']['x'], data['center']['y']),
            area=data['area']
        )


@dataclass
class DetectionResponse:
    """Complete detection response."""
    success: bool
    num_defects: int
    defects: List[DefectInfo]
    processing_time: float
    image_size: Tuple[int, int]
    timestamp: str
    error_message: Optional[str] = None
    
    # Decoded images (numpy arrays)
    annotated_image: Optional[np.ndarray] = None
    difference_map: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    
    # Summary statistics
    defect_types: List[str] = None
    avg_confidence: float = 0.0
    
    def __post_init__(self):
        if self.defect_types is None:
            self.defect_types = []


@dataclass
class ModelInfo:
    """Model information."""
    model_path: str
    device: str
    num_classes: int
    class_names: List[str]
    input_size: Tuple[int, int]
    confidence_threshold: float


@dataclass
class DefectClass:
    """Defect class information."""
    name: str
    display_name: str
    color: str
    description: str


# ============================================================
# API CLIENT
# ============================================================

class PCBDetectionClient:
    """
    Client for the PCB Defect Detection API.
    
    This client handles all communication with the FastAPI backend,
    including image encoding, request handling, and response parsing.
    
    Example:
        >>> client = PCBDetectionClient("http://localhost:8000")
        >>> 
        >>> # Check server health
        >>> health = client.health_check()
        >>> print(f"Server healthy: {health.healthy}")
        >>> 
        >>> # Detect defects
        >>> result = client.detect(template_image, test_image)
        >>> print(f"Found {result.num_defects} defects")
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the API server
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Session for connection pooling
        self.session = requests.Session()
        
        logger.info(f"PCBDetectionClient initialized: {self.base_url}")
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> requests.Response:
        """
        Make HTTP request with retry logic.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            **kwargs: Additional request arguments
            
        Returns:
            Response object
            
        Raises:
            RequestException: If all retries fail
        """
        url = f"{self.base_url}{endpoint}"
        kwargs.setdefault('timeout', self.timeout)
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.request(method, url, **kwargs)
                response.raise_for_status()
                return response
                
            except ConnectionError as e:
                last_error = e
                logger.warning(f"Connection error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    
            except Timeout as e:
                last_error = e
                logger.warning(f"Timeout (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    
            except RequestException as e:
                # Don't retry on client errors (4xx)
                if hasattr(e, 'response') and e.response is not None:
                    if 400 <= e.response.status_code < 500:
                        raise
                last_error = e
                logger.warning(f"Request error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        raise last_error or RequestException("All retry attempts failed")
    
    def _encode_image(self, image: Union[np.ndarray, Image.Image, bytes]) -> bytes:
        """
        Encode image to bytes for upload.
        
        Args:
            image: Image as numpy array, PIL Image, or bytes
            
        Returns:
            JPEG-encoded bytes
        """
        if isinstance(image, bytes):
            return image
        
        if isinstance(image, Image.Image):
            # PIL Image
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=95)
            return buffer.getvalue()
        
        if isinstance(image, np.ndarray):
            # Numpy array
            success, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not success:
                raise ValueError("Failed to encode image")
            return buffer.tobytes()
        
        raise TypeError(f"Unsupported image type: {type(image)}")
    
    def _decode_base64_image(self, base64_str: str) -> np.ndarray:
        """
        Decode base64 string to numpy array.
        
        Args:
            base64_str: Base64-encoded image string
            
        Returns:
            Decoded image as numpy array
        """
        img_bytes = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    
    # ========================================================
    # PUBLIC API METHODS
    # ========================================================
    
    def is_available(self) -> bool:
        """
        Check if the API server is available.
        
        Returns:
            True if server responds, False otherwise
        """
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def health_check(self) -> APIHealthStatus:
        """
        Perform health check on the API server.
        
        Returns:
            APIHealthStatus with server status
        """
        try:
            response = self._make_request('GET', '/health')
            data = response.json()
            
            return APIHealthStatus(
                healthy=data.get('status') == 'healthy',
                model_loaded=data.get('model_loaded', False),
                timestamp=data.get('timestamp', ''),
                version=data.get('version', '1.0.0')
            )
            
        except Exception as e:
            return APIHealthStatus(
                healthy=False,
                model_loaded=False,
                timestamp='',
                version='',
                error=str(e)
            )
    
    def get_model_info(self) -> Optional[ModelInfo]:
        """
        Get information about the loaded model.
        
        Returns:
            ModelInfo object or None if unavailable
        """
        try:
            response = self._make_request('GET', '/model/info')
            data = response.json()
            
            return ModelInfo(
                model_path=data['model_path'],
                device=data['device'],
                num_classes=data['num_classes'],
                class_names=data['class_names'],
                input_size=tuple(data['input_size']),
                confidence_threshold=data['confidence_threshold']
            )
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return None
    
    def get_defect_classes(self) -> List[DefectClass]:
        """
        Get list of available defect classes.
        
        Returns:
            List of DefectClass objects
        """
        try:
            response = self._make_request('GET', '/classes')
            data = response.json()
            
            return [
                DefectClass(
                    name=cls['name'],
                    display_name=cls['display_name'],
                    color=cls['color'],
                    description=cls['description']
                )
                for cls in data['classes']
            ]
            
        except Exception as e:
            logger.error(f"Failed to get defect classes: {e}")
            return []
    
    def detect(
        self,
        template_image: Union[np.ndarray, Image.Image, bytes],
        test_image: Union[np.ndarray, Image.Image, bytes],
        confidence_threshold: float = 0.5,
        include_images: bool = True
    ) -> DetectionResponse:
        """
        Detect defects in a PCB image pair.
        
        Args:
            template_image: Defect-free reference image
            test_image: Image to inspect
            confidence_threshold: Minimum confidence threshold
            include_images: Whether to include result images
            
        Returns:
            DetectionResponse with all detection results
        """
        try:
            # Encode images
            template_bytes = self._encode_image(template_image)
            test_bytes = self._encode_image(test_image)
            
            # Prepare multipart form data
            files = {
                'template': ('template.jpg', template_bytes, 'image/jpeg'),
                'test': ('test.jpg', test_bytes, 'image/jpeg')
            }
            
            params = {
                'confidence_threshold': confidence_threshold,
                'include_images': include_images
            }
            
            # Make request
            response = self._make_request(
                'POST',
                '/detect',
                files=files,
                params=params
            )
            
            data = response.json()
            
            # Parse defects
            defects = [DefectInfo.from_dict(d) for d in data.get('defects', [])]
            
            # Decode images if included
            annotated_image = None
            difference_map = None
            mask = None
            
            if data.get('annotated_image_base64'):
                annotated_image = self._decode_base64_image(data['annotated_image_base64'])
            
            if data.get('difference_map_base64'):
                difference_map = self._decode_base64_image(data['difference_map_base64'])
            
            if data.get('mask_base64'):
                mask = self._decode_base64_image(data['mask_base64'])
            
            # Build response
            return DetectionResponse(
                success=data.get('success', False),
                num_defects=data.get('num_defects', 0),
                defects=defects,
                processing_time=data.get('processing_time', 0.0),
                image_size=(
                    data.get('image_size', {}).get('width', 0),
                    data.get('image_size', {}).get('height', 0)
                ),
                timestamp=data.get('timestamp', ''),
                error_message=data.get('error_message'),
                annotated_image=annotated_image,
                difference_map=difference_map,
                mask=mask,
                defect_types=data.get('summary', {}).get('defect_types', []),
                avg_confidence=data.get('summary', {}).get('avg_confidence', 0.0)
            )
            
        except RequestException as e:
            logger.error(f"Detection request failed: {e}")
            return DetectionResponse(
                success=False,
                num_defects=0,
                defects=[],
                processing_time=0.0,
                image_size=(0, 0),
                timestamp='',
                error_message=f"API request failed: {str(e)}"
            )
        
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return DetectionResponse(
                success=False,
                num_defects=0,
                defects=[],
                processing_time=0.0,
                image_size=(0, 0),
                timestamp='',
                error_message=str(e)
            )
    
    def close(self):
        """Close the client session."""
        self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def create_client(
    base_url: str = "http://localhost:8000",
    **kwargs
) -> PCBDetectionClient:
    """
    Create and return an API client instance.
    
    Args:
        base_url: API server URL
        **kwargs: Additional client configuration
        
    Returns:
        Configured PCBDetectionClient instance
    """
    return PCBDetectionClient(base_url=base_url, **kwargs)


def quick_detect(
    template_path: str,
    test_path: str,
    base_url: str = "http://localhost:8000",
    confidence_threshold: float = 0.5
) -> DetectionResponse:
    """
    Quick one-shot detection from file paths.
    
    Args:
        template_path: Path to template image
        test_path: Path to test image
        base_url: API server URL
        confidence_threshold: Minimum confidence
        
    Returns:
        DetectionResponse with results
    """
    template = cv2.imread(template_path)
    test = cv2.imread(test_path)
    
    if template is None:
        raise ValueError(f"Failed to load template: {template_path}")
    if test is None:
        raise ValueError(f"Failed to load test image: {test_path}")
    
    with create_client(base_url) as client:
        return client.detect(template, test, confidence_threshold)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("Testing PCBDetectionClient...")
    
    client = create_client()
    
    # Test health check
    print("\n1. Health Check:")
    health = client.health_check()
    print(f"   Healthy: {health.healthy}")
    print(f"   Model loaded: {health.model_loaded}")
    print(f"   Version: {health.version}")
    
    if health.healthy:
        # Test model info
        print("\n2. Model Info:")
        info = client.get_model_info()
        if info:
            print(f"   Device: {info.device}")
            print(f"   Classes: {info.num_classes}")
        
        # Test defect classes
        print("\n3. Defect Classes:")
        classes = client.get_defect_classes()
        for cls in classes:
            print(f"   - {cls.display_name}: {cls.description}")
    
    print("\n✓ Client test complete!")
    client.close()
