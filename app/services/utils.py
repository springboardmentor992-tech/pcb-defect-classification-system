import cv2
import numpy as np
import base64
from fastapi import UploadFile

def read_image_file(file: UploadFile) -> np.ndarray:
    """Read uploaded file as OpenCV image"""
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {file.filename}")
    return img

def encode_image_to_base64(image: np.ndarray) -> str:
    """Encode OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.png', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64
