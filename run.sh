#!/bin/bash

# Function to kill the backend when this script exits (Ctrl+C)
cleanup() {
    echo ""
    echo "🛑 Stopping Backend (PID: $BACKEND_PID)..."
    kill $BACKEND_PID
    exit
}
trap cleanup EXIT INT

# Check if venv exists and activate
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "⚠️  Virtual environment 'venv' not found."
    echo "Please ensure you have created it and installed requirements."
    exit 1
fi

echo "🚀 Starting PCB Defect Detection System"
echo "======================================="

# 1. Start FastAPI Backend in background
echo "Starting Backend Server..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait briefly for backend to initialize
echo "Waiting for backend to be ready..."
sleep 3

# 2. Start Streamlit Frontend
echo "Starting Frontend Interface..."
streamlit run frontend.py
