# PCB Defect Classification System

## Overview
This project is an automated pipeline for detecting and classifying defects on Printed Circuit Boards (PCBs) using computer vision and deep learning. It aligns a defective PCB image with a template, identifies regions of interest (ROIs) that differ, and classifies these defects using a PyTorch ResNet18 model.

## Features
- **Image Processing Pipeline**: Uses OpenCV for image alignment (ORB & Homography), image differencing, and contour detection to isolate potential defects.
- **Deep Learning Classification**: Employs a trained PyTorch model (ResNet18) to classify the extracted regions into one of several defect categories:
  - Missing Hole
  - Mouse Bite
  - Open Circuit
  - Short
  - Spur
  - Spurious Copper
- **FastAPI Backend**: Provides a RESTful API to run the inference pipeline.
- **Streamlit Frontend**: Offers a user-friendly web interface for uploading images and visualizing the detected defects.

## Project Structure
- `app/main.py` - FastAPI backend application handling API requests.
- `frontend.py` - Streamlit frontend application.
- `model_train.py` - Script for training the PyTorch ResNet18 model on PCB defect data.
- `pipeline.py` / `inference` module - Core logic for image alignment, ROI extraction, and defect classification.
- `run.sh` - Bash script to easily launch both the backend and frontend servers simultaneously.
- `requirements.txt` - Python dependencies.

## Setup and Installation

### Prerequisites
- Python 3.8+
- Virtual Environment (recommended)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/springboardmentor992-tech/pcb-defect-classification-system.git
   cd pcb-defect-classification-system
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application (Backend + Frontend)
You can start the entire system using the provided shell script:

```bash
./run.sh
```

This will:
1. Start the FastAPI backend server on `http://localhost:8000`.
2. Launch the Streamlit web interface (usually accessible at `http://localhost:8501`).

### Training the Model
To train the classification model on your dataset, configure the paths in `model_train.py` and run:

```bash
python model_train.py
```

## Technologies Used
- **Python**: Core programming language.
- **OpenCV**: Image processing, alignment, and computer vision tasks.
- **PyTorch**: Deep learning model definition, training, and testing.
- **FastAPI**: Backend API development.
- **Streamlit**: Web frontend UI.
