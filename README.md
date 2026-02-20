# 🚀 PCB DEFECT DETECTION AND CLASSIFICATION SYSTEM

An advanced AI-powered system designed to automate quality inspection of Printed Circuit Boards (PCBs) using Computer Vision and Deep Learning.


# 📌 OVERVIEW

Manual inspection of PCBs in manufacturing industries is slow, inconsistent, and error-prone.  
This system provides an automated, AI-based inspection pipeline capable of detecting and classifying PCB defects with high accuracy.

The system combines:

- Advanced Image Alignment (ORB + Homography)
- Image Differencing & ROI Extraction
- Deep Learning Classification (PyTorch)
- REST API Backend (FastAPI)
- Scalable Inference Pipeline


# 🎯 KEY FEATURES

## 🔍 Image Processing Pipeline
- PCB alignment using ORB feature matching
- Homography transformation
- Image subtraction and contour detection
- Region of Interest (ROI) extraction

## 🧠 Deep Learning Classification
The model classifies defects into:

1. Missing Hole  
2. Mouse Bite  
3. Open Circuit  
4. Short Circuit  
5. Spur  
6. Spurious Copper  

Built using:
- PyTorch
- Torchvision (ResNet / EfficientNet)

## 🌐 Backend API
- FastAPI-based REST API
- Interactive Swagger documentation (`/docs`)
- JSON response with defect classification

## ⚡ Real-Time Inference
- Upload defective PCB image
- Automatic alignment with template
- Detection + classification
- Output visualization


# 🏗️ SYSTEM ARCHITECTURE

1️⃣ Input Reference PCB  
2️⃣ Input Defective PCB  
3️⃣ Image Alignment (ORB + Homography)  
4️⃣ Image Subtraction  
5️⃣ ROI Extraction  
6️⃣ Deep Learning Classification  
7️⃣ API Response Output  


🚀 Installation & Setup

1. Clone the Repository
	   git clone https://github.com/springboardmentor992-tech/pcb-defect-classification-system/tree/rajesh
     cd pcb-defect-detection 
2. Create Virtual Environment
     python -m venv venv
     python3 -m uvicorn main:app --reload     
4. Install Dependencies
     pip install -r requirements.txt


💻 Running the Application

To run the full system, you need to start both the backend API and the frontend dashboard.
1. Start the Backend API (Terminal 1) This service handles the AI inference.
      cd backend
      python3 -m uvicorn app_backend:app --host 127.0.0.1 --port 8000
2. Start the Frontend Dashboard (Terminal 2) This launches the web interface.
      cd ~/Desktop/pcb_defect_system/frontend
      streamlit run app_frontend.py
Open your browser to http://localhost:8501.
