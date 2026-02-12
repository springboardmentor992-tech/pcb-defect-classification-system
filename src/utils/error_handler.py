# File: src/utils/error_handler.py

"""
Comprehensive error handling utilities
"""

import logging
from functools import wraps
from pathlib import Path
import traceback
import cv2
import torch


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

try:
    log_file = Path('logs/app.log')
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(str(log_file))
    logging.getLogger().addHandler(file_handler)
except Exception:
    pass # Fallback to stream only

logger = logging.getLogger(__name__)


class PCBDetectionError(Exception):
    """Base exception for PCB detection errors"""
    pass


class ImageLoadError(PCBDetectionError):
    """Error loading images"""
    pass


class AlignmentError(PCBDetectionError):
    """Error during image alignment"""
    pass


class ModelError(PCBDetectionError):
    """Error with model inference"""
    pass


def handle_errors(func):
    """Decorator for error handling"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise ImageLoadError(f"Could not find required file: {e}")
        except cv2.error as e:
            logger.error(f"OpenCV error: {e}")
            raise AlignmentError(f"Image processing failed: {e}")
        except torch.cuda.OutOfMemoryError:
            logger.error("GPU out of memory")
            raise ModelError("GPU out of memory. Try using CPU or smaller batch size.")
        except Exception as e:
            logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
            raise PCBDetectionError(f"An unexpected error occurred: {e}")
    
    return wrapper


def validate_image_path(path: str) -> Path:
    """Validate image path exists and is valid"""
    p = Path(path)
    if not p.exists():
        raise ImageLoadError(f"Image not found: {path}")
    if p.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
        raise ImageLoadError(f"Unsupported image format: {p.suffix}")
    return p


def validate_model_path(path: str) -> Path:
    """Validate model path"""
    p = Path(path)
    if not p.exists():
        raise ModelError(f"Model not found: {path}")
    if p.suffix not in ['.pth', '.pt', '.h5']:
        raise ModelError(f"Unsupported model format: {p.suffix}")
    return p
