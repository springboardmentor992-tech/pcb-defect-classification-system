from pydantic import BaseModel
from typing import List, Optional

class DefectResult(BaseModel):
    id: int
    class_name: str
    confidence: float
    bbox: List[int]

class InferenceResponse(BaseModel):
    success: bool
    image_name: str
    template_name: Optional[str]
    num_defects: int
    defects: List[DefectResult]
    processing_time: float
    annotated_image_base64: Optional[str] = None
    threshold_image_base64: Optional[str] = None

class ErrorResponse(BaseModel):
    success: bool
    error: str
