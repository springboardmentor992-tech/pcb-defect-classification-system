import logging
import sys

def setup_logging():
    """Context manager or setup function for logging"""
    logger = logging.getLogger("pcb_defect_detection")
    logger.setLevel(logging.INFO)

    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    
    # Format: Timestamp - Level - Message
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    # Avoid adding duplicate handlers if setup is called multiple times
    if not logger.handlers:
        logger.addHandler(handler)
    
    return logger

# Create a global logger instance
logger = setup_logging()
