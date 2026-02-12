"""
PCB Defect Detection - API Module
==================================

REST API server for PCB defect detection.

Components:
    - server: FastAPI application with all endpoints
    - client: API client for frontend integration

Usage:
    # Start server
    uvicorn src.api.server:app --host 0.0.0.0 --port 8000
    
    # Or
    python -m src.api.server
"""

from .server import app

__all__ = ['app']
