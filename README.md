# PCB Defect Detection and Classification System

An advanced AI-powered system designed to automate the quality control process for Printed Circuit Boards (PCBs). This project leverages state-of-the-art Deep Learning models (EfficientNet) to identify and classify manufacturing defects with high precision.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-ff4b4b)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)

## 🔍 Overview

Manual inspection of PCBs is time-consuming and error-prone. This system automates defect detection using computer vision, providing a faster and more consistent alternative.

### Key Capabilities
- **Multi-Class Detection**: Identifies 6 specific defect types:
  1.  **Missing Hole**: Absence of required drill holes.
  2.  **Mouse Bite**: Irregular cutouts on the board edge.
  3.  **Open Circuit**: Broken copper traces.
  4.  **Short Circuit**: Unwanted connections between traces.
  5.  **Spur**: Small protrusions of copper.
  6.  **Spurious Copper**: Excess copper material.
- **High Accuracy**: Utilizes **EfficientNet-B3** for robust feature extraction and classification.
- **Real-Time Interface**: A modern, dark-themed dashboard built with **Streamlit** for easy interaction.
- **REST API**: A **FastAPI** backend that allows for seamless integration into manufacturing pipelines.
- **Reporting**: Generates downloadable reports (CSV, JSON) for quality assurance records.

## 🛠️ Technology Stack

- **Frontend**: Streamlit (Premium Dark Theme UI)
- **Backend**: FastAPI (Python)
- **Model**: PyTorch (EfficientNet Architecture)
- **Image Processing**: OpenCV, NumPy, Pillow
- **Data Handling**: Pandas

## 🚀 Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/YOUR_USERNAME/PCB-Defect-Detection-and-Classification-System.git
    cd PCB-Defect-Detection-and-Classification-System
    ```

2.  **Create Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 💻 Running the Application

To run the full system, you need to start both the backend API and the frontend dashboard.

**1. Start the Backend API (Terminal 1)**
This service handles the AI inference.
```bash
uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000
```

**2. Start the Frontend Dashboard (Terminal 2)**
This launches the web interface.
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

## 📂 Project Structure

- `app.py`: Main entry point for the Streamlit dashboard.
- `src/`: Core source code.
    - `api/`: FastAPI server and endpoints.
    - `models/`: Neural network architecture and training logic.
    - `pipeline/`: Inference pipelines.
    - `preprocessing/`: Image augmentation and preparation.
    - `utils/`: Helper functions for visualization and export.
- `tests/`: Unit tests and benchmarks. (Optional)
