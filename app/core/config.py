import os
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "models/pcb_defect_resnet18.pth")

# Ensure absolute path for safety if needed
if not os.path.isabs(MODEL_PATH):
    # Depending on where the app is run from, this might need adjustment.
    # Assuming run from root of project
    MODEL_PATH = os.path.abspath(MODEL_PATH)

CLASS_NAMES = [
    'Missing_hole',
    'Mouse_bite',
    'Open_circuit',
    'Short',
    'Spur',
    'Spurious_copper'
]
