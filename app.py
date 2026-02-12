"""
PCB Defect Detection - Streamlit Web Application
=================================================

A professional, modern web application for PCB defect detection
using EfficientNet-B3 and computer vision techniques.

Features:
---------
- Modern UI with glassmorphism design
- Image upload with drag-and-drop
- Real-time defect detection
- Interactive results visualization
- Export functionality (PNG, CSV, PDF)
- Processing history

Author: PCB Defect Detection Team
Version: 1.0.0
"""

import streamlit as st
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import sys
import os
import io
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import base64
import requests

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# Import API client
try:
    from api.client import PCBDetectionClient, create_client
    API_CLIENT_AVAILABLE = True
except ImportError:
    API_CLIENT_AVAILABLE = False

# Import export utilities
try:
    from utils.export_utils import (
        export_defects_csv, export_summary_csv, export_results_json,
        export_text_report, numpy_to_bytes, create_comparison_image,
        export_all_formats, get_timestamp_filename
    )
    EXPORT_AVAILABLE = True
except ImportError:
    EXPORT_AVAILABLE = False

# Import visualization utilities
try:
    from utils.visualization import (
        create_bar_chart_svg, create_pie_chart_svg,
        create_confidence_histogram_svg, calculate_statistics,
        create_stats_card_html, DEFECT_COLORS, DEFECT_DISPLAY_NAMES,
        create_defect_heatmap, draw_enhanced_annotations
    )
    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False

# API Configuration
import logging
# Setup logging
try:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        filename=str(log_dir / 'streamlit_app.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
except Exception as e:
    print(f"Logging setup failed: {e}")

logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8000"

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PCB Defect Detection | AI-Powered Quality Control",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': """
        # PCB Defect Detection System
        
        An AI-powered system for detecting and classifying defects 
        in Printed Circuit Boards using EfficientNet-B3.
        
        **Accuracy:** 99.46%  
        **Processing Time:** < 3 seconds  
        **Supported Defects:** 6 types
        
        Built with ❤️ using Streamlit, PyTorch, and OpenCV
        """
    }
)

# ============================================================
# CUSTOM CSS STYLING
# ============================================================

def load_custom_css(dark_mode: bool = False):
    """Load custom CSS for modern UI design with theme support."""
    if dark_mode:
        load_dark_theme_css()
    else:
        load_light_theme_css()


def load_dark_theme_css():
    """Load premium dark theme CSS with extensive styling overrides."""
    st.markdown("""
    <style>
    /* ============================================
       PREMIUM DARK THEME PALETTE
       ============================================ */
    :root {
        /* Core Brand Colors - Vibrant & Premium */
        --primary-color: #6366f1;       /* Indigo-500 */
        --primary-hover: #4f46e5;       /* Indigo-600 */
        --primary-glow: rgba(99, 102, 241, 0.4);
        
        --accent-color: #ec4899;        /* Pink-500 */
        --accent-hover: #db2777;        /* Pink-600 */
        
        /* Backgrounds - Deep & Rich */
        --bg-main: #0f172a;             /* Slate-900 */
        --bg-grad-start: #0f172a;
        --bg-grad-end: #1e293b;         /* Slate-800 */
        
        --bg-card: rgba(30, 41, 59, 0.7);     /* Slate-800 with opacity */
        --bg-sidebar: #020617;          /* Slate-950 */
        --bg-input: #1e293b;            /* Slate-800 */
        
        /* Text Colors */
        --text-pure: #ffffff;
        --text-primary: #f8fafc;        /* Slate-50 */
        --text-secondary: #cbd5e1;      /* Slate-300 */
        --text-muted: #94a3b8;          /* Slate-400 */
        
        /* Borders & Effects */
        --border-subtle: rgba(148, 163, 184, 0.1);
        --border-hover: rgba(99, 102, 241, 0.5);
        
        --glass-blur: blur(12px);
        --shadow-card: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.15);
        --shadow-hover: 0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 10px 10px -5px rgba(0, 0, 0, 0.15);
    }

    /* ============================================
       GLOBAL RESETS & BACKGROUNDS
       ============================================ */
    .stApp {
        background: radial-gradient(circle at top left, var(--bg-grad-end), var(--bg-grad-start));
        color: var(--text-primary);
    }
    
    .main .block-container {
        padding-top: 2rem;
        background: transparent !important;
    }

    header, footer {
        background: transparent !important;
    }

    /* ============================================
       SIDEBAR PREMIUM STYLING
       ============================================ */
    [data-testid="stSidebar"] {
        background-color: var(--bg-sidebar);
        border-right: 1px solid var(--border-subtle);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1, 
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2, 
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        color: var(--text-pure) !important;
        font-weight: 700;
        letter-spacing: -0.025em;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p, 
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {
        color: var(--text-secondary) !important;
    }
    
    /* Sidebar separator line color fix */
    [data-testid="stSidebar"] hr {
        border-color: var(--border-subtle) !important;
    }

    /* ============================================
       TYPOGRAPHY
       ============================================ */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-pure) !important;
        font-family: 'Inter', system-ui, sans-serif;
        font-weight: 700;
    }
    
    p, label, span, div {
        color: var(--text-primary); 
    }
    
    .subtitle {
        color: var(--text-muted) !important;
        font-size: 1.1rem;
    }

    /* ============================================
       GLASS CARDS
       ============================================ */
    .glass-card {
        background: var(--bg-card);
        backdrop-filter: var(--glass-blur);
        -webkit-backdrop-filter: var(--glass-blur);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: var(--shadow-card);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        border-color: var(--border-hover);
        box-shadow: var(--shadow-hover), 0 0 15px var(--primary-glow);
    }

    /* ============================================
       BUTTONS - Modern Gradient & States
       ============================================ */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color), var(--primary-hover));
        color: white !important;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(99, 102, 241, 0.4);
        filter: brightness(1.1);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }

    /* Secondary/Ghost Buttons */
    .stDownloadButton > button {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid var(--border-subtle);
        color: var(--text-primary) !important;
    }
    
    .stDownloadButton > button:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: var(--text-primary);
        color: white !important;
    }

    /* ============================================
       INPUTS, SELECTS & TOGGLES
       ============================================ */
    /* Text Inputs */
    .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
        background-color: var(--bg-input) !important;
        color: var(--text-pure) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px;
    }
    
    /* Input formatting when focused */
    .stTextInput input:focus, .stSelectbox div[data-baseweb="select"] > div:focus-within {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 2px var(--primary-glow) !important;
    }

    /* Dropdown menu items */
    ul[data-baseweb="menu"] {
        background-color: var(--bg-sidebar) !important;
        border: 1px solid var(--border-subtle) !important;
    }
    
    li[data-baseweb="option"] {
        color: var(--text-primary) !important;
    }
    
    li[aria-selected="true"] {
        background-color: var(--bg-input) !important;
        color: var(--primary-color) !important;
    }

    /* Toggle Switch */
    .stToggle label {
        color: var(--text-primary) !important;
    }

    /* ============================================
       FILE UPLOADER - Custom Dark Style
       ============================================ */
    [data-testid="stFileUploader"] {
        background-color: rgba(30, 41, 59, 0.3);
        border: 2px dashed var(--border-subtle);
        border-radius: 16px;
        padding: 2rem;
        transition: border-color 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--primary-color);
        background-color: rgba(30, 41, 59, 0.5);
    }

    .stFileUploaderDropzone {
        background: transparent !important;
    }
    
    [data-testid="stFileUploader"] small {
        color: var(--text-muted) !important;
    }
    
    /* Upload Button inside uploader */
    [data-testid="stFileUploader"] button {
        background-color: var(--bg-input);
        color: var(--text-primary);
        border: 1px solid var(--border-subtle);
    }
    
    /* File Uploader Text Coloring */
    [data-testid="stFileUploader"] div, 
    [data-testid="stFileUploader"] span, 
    [data-testid="stFileUploader"] small {
        color: var(--text-secondary) !important;
    }
    
    /* Specific target for "Drag and drop file here" */
    section[data-testid="stFileUploaderDropzone"] > div > div > span {
        color: var(--text-primary) !important;
        font-weight: 500;
    }

    /* Remove white background from dropzone instructions */
    [data-testid="stFileUploaderDropzone"] {
        background-color: var(--bg-card) !important;
        border: 1px dashed var(--border-subtle) !important;
        border-radius: 12px;
    }
    
    /* Fix weird internal streamlit white overlay */
    section[data-testid="stFileUploaderDropzone"] > div {
        background-color: transparent !important;
        color: var(--text-primary) !important;
    }
    
    /* Question Mark Tooltip Icon */
    [data-testid="stTooltipIcon"] {
        color: var(--text-secondary) !important;
    }
    
    [data-testid="stTooltipIcon"] svg {
        fill: var(--text-secondary) !important;
    }
    
    [data-testid="stTooltipIcon"]:hover svg {
        fill: var(--primary-color) !important;
    }
    
    [data-testid="stTooltipIcon"] > div {
        background-color: transparent !important;
    }

    /* ============================================
       TABS & NAV
       ============================================ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        padding-bottom: 5px;
        border-bottom: 2px solid var(--border-subtle);
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border-radius: 8px 8px 0 0;
        color: var(--text-muted);
        border: none;
        padding: 0 20px;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--bg-input) !important;
        color: var(--primary-color) !important;
        border-bottom: 2px solid var(--primary-color) !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary);
        background-color: rgba(255, 255, 255, 0.05);
    }

    /* ============================================
       METRICS 
       ============================================ */
    [data-testid="stMetric"] {
        background-color: var(--bg-input);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-muted) !important;
    }
    
    [data-testid="stMetricValue"] {
        color: var(--text-pure) !important;
    }

    /* ============================================
       SLIDER
       ============================================ */
    .stSlider [data-baseweb="slider"] {
        /* Track */
    }

    /* ============================================
       EXPANDER - Dark Theme Fix
       ============================================ */
    [data-testid="stExpander"] {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* The Clickable Header */
    [data-testid="stExpander"] details > summary {
        background-color: transparent !important;
        color: var(--text-pure) !important;
        border-radius: 8px !important;
        transition: background-color 0.2s;
    }
    
    [data-testid="stExpander"] details > summary:hover {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: var(--primary-color) !important;
    }
    
    /* The Content Body */
    [data-testid="stExpander"] details > div {
        color: var(--text-secondary) !important;
    }
    
    /* Force SVG arrows to be white */
    [data-testid="stExpander"] details > summary svg {
        fill: var(--text-muted) !important;
        color: var(--text-muted) !important;
    }
    
    /* ============================================
       ALERTS
       ============================================ */
    .stSuccess { background-color: rgba(6, 95, 70, 0.5); border-color: #059669; color: #a7f3d0; }
    .stInfo { background-color: rgba(30, 58, 138, 0.5); border-color: #2563eb; color: #bfdbfe; }
    .stWarning { background-color: rgba(120, 53, 15, 0.5); border-color: #d97706; color: #fde68a; }
    /* ============================================
       GRID & LAYOUT FIXES
       ============================================ */
    
    /* Target the specific grid container used in Streamlit */
    [data-testid="column"] {
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }

    /* Force background on potentially white containers */
    .stMarkdown, .stText, .stJson, .stDataFrame {
         background-color: transparent !important;
    }
    
    /* Ensure metric cards don't have white background gaps */
    div[data-testid="stMetric"] {
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
    }

    /* Fix any potential white borders or outlines */
    * {
        border-color: var(--border-subtle) !important; 
    }
    
    /* Toast/Notification Fix */
    /* Tooltip Fix - Force Dark Background */
    div[role="tooltip"], div[data-baseweb="tooltip"] {
         background-color: #0f172a !important; /* Slate-900 */
         color: #f8fafc !important; /* Slate-50 */
         border: 1px solid var(--border-subtle) !important;
         border-radius: 6px !important;
         padding: 8px !important;
         font-size: 0.85rem !important;
         box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5) !important;
    }
    
    /* Metrics Top Box Container Fix - Custom HTML Version */
    .metric-card {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2) !important;
        transition: transform 0.2s ease, border-color 0.2s ease;
        min-height: 110px;
        min-width: 100px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .metric-card:hover {
        border-color: var(--primary-color) !important;
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4) !important;
    }

    .metric-card .metric-value {
        color: var(--text-pure) !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        text-shadow: 0 0 15px rgba(99, 102, 241, 0.4); 
    }
    
    .metric-card .metric-label {
        color: var(--text-secondary) !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        margin-top: 0.5rem;
    }

    /* Metrics Top Box Container Fix - Native Streamlit Version (Fallback) */
    div[data-testid="stMetric"], [data-testid="stMetric"] {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2) !important;
        transition: transform 0.2s ease, border-color 0.2s ease;
        min-height: 110px;
        min-width: 100px;
        display: block !important;
        height: 100% !important;
        position: relative;
        z-index: 1;
    }
    
    [data-testid="stMetric"]:hover {
        border-color: var(--primary-color) !important;
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4) !important;
    }
    
    /* Ensure label and value are visible on top of background */
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        position: relative;
        z-index: 2;
        background: transparent !important;
    }
    
    /* Metric Value Color Override */
    [data-testid="stMetricValue"] > div {
        color: var(--text-pure) !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        text-shadow: 0 0 15px rgba(99, 102, 241, 0.4); /* Stronger glow */
    }

    /* Metric Label Color Override */
    [data-testid="stMetricLabel"] > div > div > p {
        color: var(--text-secondary) !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }
    
    /* Global Spacing Improvements */
    .block-container {
        gap: 2rem !important;
        padding-bottom: 5rem !important;
    }
    
    /* Add spacing between elements */
    [data-testid="stVerticalBlock"] > div {
        margin-bottom: 1rem;
    }
    
    /* Help Tooltip Icon Color */
    [data-testid="stTooltipIcon"] {
        color: var(--text-muted) !important;
    }
    
    /* Upload Question Mark Tooltip specific fix */
    div[data-testid="stTooltipContent"] {
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
    }

    </style>
    """, unsafe_allow_html=True)


def load_light_theme_css():
    st.markdown("""
    <style>
    /* ============================================
       LIGHT THEME COLOR PALETTE
       ============================================
       Soft Rose:       #DDD0CC
       Light Gray:      #E2E4E4
       Sage/Teal:       #A8B5B7 (accent)
       Light Blue-Gray: #BFC8CE
       Warm Cream:      #FFEFD5
       ============================================ */
    
    /* CSS Variables */
    :root {
        --primary-color: #5a6f72;
        --primary-dark: #4a5c5f;
        --primary-light: #A8B5B7;
        --accent-color: #7a8f92;
        --accent-light: #BFC8CE;
        
        --bg-main: #F5F5F3;
        --bg-cream: #FFEFD5;
        --bg-rose: #DDD0CC;
        --bg-gray: #E2E4E4;
        --bg-card: #FFFFFF;
        --bg-sidebar: linear-gradient(180deg, #E8E5E2 0%, #DDD0CC 100%);
        
        --text-dark: #2c3e50;
        --text-primary: #34495e;
        --text-secondary: #5a6c7d;
        --text-muted: #7f8c8d;
        
        --border-light: rgba(168, 181, 183, 0.3);
        --border-medium: rgba(168, 181, 183, 0.5);
        
        --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.06);
        --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.08);
        --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.1);
        
        --success-color: #27ae60;
        --warning-color: #f39c12;
        --error-color: #e74c3c;
        --info-color: #5a6f72;
    }
    
    /* ============================================
       MAIN CONTAINER - Light Background
       ============================================ */
    .stApp {
        background: linear-gradient(135deg, #F5F5F3 0%, #E8E5E2 50%, #F5F5F3 100%);
    }
    
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* ============================================
       SIDEBAR STYLING - Warm tones
       ============================================ */
    [data-testid="stSidebar"] {
        background: var(--bg-sidebar);
        border-right: 1px solid var(--border-light);
    }
    
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--text-dark) !important;
        font-weight: 600;
    }
    
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown span,
    [data-testid="stSidebar"] label {
        color: var(--text-primary) !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: var(--text-primary);
    }
    
    /* ============================================
       SOFT CARDS (replacing glassmorphism)
       ============================================ */
    .glass-card {
        background: var(--bg-card);
        border: 1px solid var(--border-light);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-sm);
    }
    
    .glass-card:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
        border-color: var(--primary-light);
    }
    
    /* ============================================
       HEADERS & TITLES - Visible text
       ============================================ */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: var(--text-dark);
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        color: var(--text-secondary);
        text-align: center;
        font-size: 1.15rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .section-header {
        color: var(--text-dark) !important;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--primary-light);
    }
    
    /* All text should be dark */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-dark) !important;
    }
    
    p, span, label, div {
        color: var(--text-primary);
    }
    
    /* ============================================
       UPLOAD AREA - Clean design
       ============================================ */
    .upload-zone {
        background: linear-gradient(135deg, rgba(168, 181, 183, 0.15) 0%, rgba(191, 200, 206, 0.15) 100%);
        border: 2px dashed var(--primary-light);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .upload-zone:hover {
        border-color: var(--primary-color);
        background: linear-gradient(135deg, rgba(168, 181, 183, 0.25) 0%, rgba(191, 200, 206, 0.25) 100%);
        transform: scale(1.01);
    }
    
    .upload-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .upload-text {
        color: var(--text-dark);
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    .upload-hint {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* ============================================
       BUTTONS - Vibrant action button
       ============================================ */
    .stButton > button {
        background: linear-gradient(135deg, #16a085 0%, #1abc9c 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(22, 160, 133, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(22, 160, 133, 0.4);
        background: linear-gradient(135deg, #1abc9c 0%, #16a085 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Download buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        font-weight: 500;
        box-shadow: 0 3px 10px rgba(52, 152, 219, 0.3);
    }
    
    .stDownloadButton > button:hover {
        box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
        transform: translateY(-1px);
    }
    
    /* ============================================
       METRICS & STATS - Clear visibility
       ============================================ */
    .metric-container {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin: 1.5rem 0;
    }
    
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-light);
        border-radius: 14px;
        padding: 1.25rem;
        flex: 1;
        min-width: 150px;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-sm);
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-md);
        border-color: var(--primary-light);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-color);
    }
    
    .metric-label {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin-top: 0.25rem;
        font-weight: 500;
    }
    
    /* Streamlit native metrics */
    [data-testid="stMetric"] {
        background: var(--bg-card);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid var(--border-light);
        box-shadow: var(--shadow-sm);
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
    }
    
    [data-testid="stMetricValue"] {
        color: var(--primary-color) !important;
    }
    
    /* ============================================
       DEFECT CLASSES LEGEND
       ============================================ */
    .defect-legend {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
        gap: 0.75rem;
        margin: 1rem 0;
    }
    
    .defect-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0.75rem;
        background: var(--bg-card);
        border: 1px solid var(--border-light);
        border-radius: 8px;
        font-size: 0.85rem;
        color: var(--text-dark);
    }
    
    .defect-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
    }
    
    /* ============================================
       RESULTS SECTION
       ============================================ */
    .result-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
    }
    
    .defect-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: rgba(168, 181, 183, 0.2);
        border: 1px solid var(--border-medium);
        border-radius: 20px;
        color: var(--text-dark);
        font-weight: 500;
    }
    
    .confidence-bar {
        height: 8px;
        background: var(--bg-gray);
        border-radius: 4px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--success-color), var(--primary-color));
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    
    /* ============================================
       TABS - Clean flat design
       ============================================ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: var(--bg-gray);
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: var(--text-secondary);
        padding: 0.75rem 1.5rem;
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.7);
        color: var(--text-dark);
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--bg-card) !important;
        color: var(--primary-color) !important;
        font-weight: 600;
        box-shadow: var(--shadow-sm);
    }
    
    /* ============================================
       FILE UPLOADER
       ============================================ */
    [data-testid="stFileUploader"] {
        background: var(--bg-card);
        border: 2px dashed var(--border-medium);
        border-radius: 12px;
        padding: 1rem;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--primary-color);
        background: rgba(168, 181, 183, 0.1);
    }
    
    [data-testid="stFileUploader"] label {
        color: var(--text-dark) !important;
    }
    
    /* ============================================
       EXPANDER
       ============================================ */
    .streamlit-expanderHeader {
        background: var(--bg-card);
        border-radius: 8px;
        color: var(--text-dark) !important;
        border: 1px solid var(--border-light);
    }
    
    .streamlit-expanderHeader:hover {
        background: var(--bg-gray);
    }
    
    .streamlit-expanderContent {
        background: var(--bg-card);
        border: 1px solid var(--border-light);
        border-top: none;
        border-radius: 0 0 8px 8px;
    }
    
    /* ============================================
       ALERTS & MESSAGES
       ============================================ */
    .stSuccess {
        background: rgba(39, 174, 96, 0.1);
        border: 1px solid rgba(39, 174, 96, 0.3);
        border-radius: 12px;
        color: #1e7e4a !important;
    }
    
    .stInfo {
        background: rgba(90, 111, 114, 0.1);
        border: 1px solid rgba(90, 111, 114, 0.3);
        border-radius: 12px;
        color: var(--text-dark) !important;
    }
    
    .stWarning {
        background: rgba(243, 156, 18, 0.1);
        border: 1px solid rgba(243, 156, 18, 0.3);
        border-radius: 12px;
        color: #c27d0e !important;
    }
    
    .stError {
        background: rgba(231, 76, 60, 0.1);
        border: 1px solid rgba(231, 76, 60, 0.3);
        border-radius: 12px;
        color: #c0392b !important;
    }
    
    /* ============================================
       SLIDER - Coral/Orange confidence threshold
       ============================================ */
    /* Slider filled/active track */
    .stSlider [data-baseweb="slider"] > div > div:nth-child(2) {
        background: linear-gradient(90deg, #e67e22 0%, #d35400 100%) !important;
    }
    
    /* Alternative selector for filled track */
    .stSlider div[data-testid="stSlider"] > div > div > div > div:last-child {
        background: linear-gradient(90deg, #e67e22 0%, #d35400 100%) !important;
    }
    
    /* Track fill */
    .stSlider [role="slider"] ~ div {
        background: linear-gradient(90deg, #e67e22, #d35400) !important;
    }
    
    .stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
        color: var(--text-dark) !important;
        font-weight: 600;
    }
    
    /* Slider thumb/handle */
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background: #d35400 !important;
        border: 3px solid white !important;
        box-shadow: 0 2px 10px rgba(211, 84, 0, 0.5) !important;
    }
    
    /* Slider track background (unfilled) */
    .stSlider [data-baseweb="slider"] > div:first-child {
        background: #DDD0CC !important;
    }
    
    /* Sidebar slider specific styles */
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div:nth-child(2) {
        background: linear-gradient(90deg, #e67e22 0%, #d35400 100%) !important;
    }
    
    [data-testid="stSidebar"] .stSlider div[role="slider"] {
        background: #d35400 !important;
    }
    
    /* ============================================
       PROGRESS BAR
       ============================================ */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--primary-light), var(--primary-color), var(--success-color));
    }
    
    /* ============================================
       IMAGE CONTAINER
       ============================================ */
    .image-container {
        background: var(--bg-card);
        border: 1px solid var(--border-light);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: var(--shadow-sm);
    }
    
    .image-container img {
        border-radius: 8px;
        max-width: 100%;
    }
    
    .image-label {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* ============================================
       SELECTBOX & INPUTS
       ============================================ */
    .stSelectbox > div > div {
        background: var(--bg-card);
        border-color: var(--border-medium);
        color: var(--text-dark);
    }
    
    .stSelectbox label {
        color: var(--text-dark) !important;
    }
    
    .stTextInput > div > div > input {
        background: var(--bg-card);
        border-color: var(--border-medium);
        color: var(--text-dark);
    }
    
    .stTextInput label {
        color: var(--text-dark) !important;
    }
    
    /* ============================================
       RADIO BUTTONS & CHECKBOXES
       ============================================ */
    .stRadio label, .stCheckbox label {
        color: var(--text-dark) !important;
    }
    
    .stRadio > div {
        color: var(--text-dark);
    }
    
    /* ============================================
       FOOTER
       ============================================ */
    .footer {
        text-align: center;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid var(--border-light);
        color: var(--text-secondary);
        background: rgba(221, 208, 204, 0.3);
        border-radius: 16px 16px 0 0;
    }
    
    .footer a {
        color: var(--primary-color);
        text-decoration: none;
        font-weight: 500;
    }
    
    .footer a:hover {
        text-decoration: underline;
    }
    
    /* ============================================
       ANIMATIONS
       ============================================ */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .loading {
        animation: pulse 1.5s infinite;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease forwards;
    }
    
    /* ============================================
       RESPONSIVE DESIGN
       ============================================ */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }
        
        .main .block-container {
            padding: 1rem;
        }
        
        .metric-card {
            min-width: 100%;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 0.5rem 1rem;
            font-size: 0.9rem;
        }
    }
    
    @media (max-width: 480px) {
        .main-title {
            font-size: 1.6rem;
        }
        
        .subtitle {
            font-size: 1rem;
        }
        
        .metric-value {
            font-size: 1.5rem;
        }
    }
    
    /* ============================================
       SCROLLBAR STYLING
       ============================================ */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-gray);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-light);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-color);
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_metric_card(value: str, label: str, icon: str = "") -> str:
    """Create a styled metric card HTML."""
    return f"""
    <div class="metric-card">
        <div class="metric-value">{icon} {value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


def create_defect_legend() -> str:
    """Create the defect classes legend with theme-aware styling."""
    defects = [
        ("Missing Hole", "#c0392b", "🔴"),
        ("Mouse Bite", "#27ae60", "🟢"),
        ("Open Circuit", "#2980b9", "🔵"),
        ("Short", "#d4a017", "🟡"),
        ("Spur", "#8e44ad", "🟣"),
        ("Spurious Copper", "#16a085", "🔷")
    ]
    
    # Check for dark mode
    is_dark = st.session_state.get('dark_mode', False)
    
    if is_dark:
        # Dark Theme Styling (Inline to ensure it works immediately in HTML)
        bg_color = "rgba(30, 41, 59, 0.5)"
        border_color = "rgba(148, 163, 184, 0.2)"
        text_color = "#f8fafc"
        box_shadow = "0 4px 6px rgba(0, 0, 0, 0.2)"
    else:
        # Light Theme Styling
        bg_color = "#ffffff"
        border_color = "rgba(168, 181, 183, 0.3)"
        text_color = "#2c3e50"
        box_shadow = "0 1px 3px rgba(0,0,0,0.05)"
    
    html = '<div style="display: flex; flex-direction: column; gap: 0.5rem;">'
    for name, color, emoji in defects:
        html += f'''
        <div class="defect-legend-item" style="
            display: flex; 
            align-items: center; 
            gap: 0.75rem; 
            padding: 0.6rem 1rem; 
            background: {bg_color}; 
            border: 1px solid {border_color}; 
            border-radius: 10px; 
            box-shadow: {box_shadow};
            backdrop-filter: blur(5px);
            transition: transform 0.2s;
        ">
            <span style="font-size: 1rem;">{emoji}</span>
            <span style="color: {text_color}; font-size: 0.85rem; font-weight: 500;">{name}</span>
        </div>'''
    html += '</div>'
    
    return html


def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

def init_session_state():
    """Initialize session state variables."""
    if 'processing_history' not in st.session_state:
        st.session_state.processing_history = []
    
    if 'current_results' not in st.session_state:
        st.session_state.current_results = None
    
    if 'template_image' not in st.session_state:
        st.session_state.template_image = None
    
    if 'test_image' not in st.session_state:
        st.session_state.test_image = None
    
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():
    """Render the sidebar with settings and information."""
    with st.sidebar:
        # Logo and title
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 3rem;">🔍</div>
            <h2 style="color: #2c3e50; margin: 0.5rem 0;">PCB Inspector</h2>
            <p style="color: #5a6c7d; font-size: 0.9rem;">AI-Powered Quality Control</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model Information
        st.markdown("### 📊 Model Info")
        st.markdown("""
        <div class="glass-card" style="padding: 1rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="color: #5a6c7d;">Architecture</span>
                <span style="color: #2c3e50; font-weight: 600;">EfficientNet-B3</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="color: #5a6c7d;">Accuracy</span>
                <span style="color: #27ae60; font-weight: 600;">99.46%</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="color: #5a6c7d;">Classes</span>
                <span style="color: #2c3e50; font-weight: 600;">6 Types</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #5a6c7d;">Speed</span>
                <span style="color: #5a6f72; font-weight: 600;">< 3 sec</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Theme Toggle
        st.markdown("### 🎨 Theme")
        dark_mode_toggle = st.toggle("Dark Mode", value=st.session_state.dark_mode)
        
        # Update session state if toggled
        if dark_mode_toggle != st.session_state.dark_mode:
            st.session_state.dark_mode = dark_mode_toggle
            st.rerun()

        st.markdown("---")
        
        # Detection Settings
        st.markdown("### ⚙️ Detection Settings")
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence score to report a defect. Higher = fewer false positives."
        )
        
        st.markdown("---")
        
        # Display Options
        st.markdown("### 👁️ Display Options")
        
        show_contours = st.checkbox("Show Contours", value=True)
        show_bboxes = st.checkbox("Show Bounding Boxes", value=True)
        show_labels = st.checkbox("Show Labels", value=True)
        show_intermediate = st.checkbox("Show Processing Steps", value=False)
        
        st.markdown("---")
        
        # Defect Classes Legend
        st.markdown("### 🏷️ Defect Classes")
        st.markdown(create_defect_legend(), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick Help
        with st.expander("❓ Quick Help"):
            st.markdown("""
            **How to use:**
            1. Upload a defect-free template image
            2. Upload the test image to inspect
            3. Adjust confidence threshold if needed
            4. Click **Detect Defects**
            5. Review results and download reports
            
            **Tips:**
            - Use high-resolution images (min 640×640)
            - Ensure good lighting conditions
            - Template should match test PCB design
            """)
        
        return {
            'confidence_threshold': confidence_threshold,
            'show_contours': show_contours,
            'show_bboxes': show_bboxes,
            'show_labels': show_labels,
            'show_intermediate': show_intermediate
        }


# ============================================================
# MAIN CONTENT
# ============================================================

def render_header():
    """Render the main header section."""
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 2rem 0;">
        <h1 class="main-title">PCB Defect Detection</h1>
        <p class="subtitle">
            Upload PCB images to detect and classify manufacturing defects using 
            state-of-the-art deep learning
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick stats bar
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_metric_card("99.46%", "Accuracy", "🎯"), unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric_card("< 3s", "Processing", "⚡"), unsafe_allow_html=True)
    with col3:
        st.markdown(create_metric_card("6", "Defect Types", "🔍"), unsafe_allow_html=True)
    with col4:
        history_count = len(st.session_state.get('processing_history', []))
        st.markdown(create_metric_card(str(history_count), "Processed", "📊"), unsafe_allow_html=True)


def render_upload_section():
    """Render the image upload section."""
    st.markdown('<h2 class="section-header">📤 Upload Images</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #2c3e50; margin-bottom: 1rem;">🖼️ Template Image</h3>
            <p style="color: #5a6c7d; font-size: 0.9rem;">
                Upload a defect-free reference PCB image
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        template_file = st.file_uploader(
            "Upload template (defect-free)",
            type=['jpg', 'jpeg', 'png'],
            key='template_uploader',
            help="This should be a perfect PCB image without any defects"
        )
        
        if template_file:
            template_image = Image.open(template_file)
            st.session_state.template_image = template_image
            st.image(template_image, caption="✅ Template loaded", use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #2c3e50; margin-bottom: 1rem;">🔬 Test Image</h3>
            <p style="color: #5a6c7d; font-size: 0.9rem;">
                Upload the PCB image to inspect for defects
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        test_file = st.file_uploader(
            "Upload test image (to inspect)",
            type=['jpg', 'jpeg', 'png'],
            key='test_uploader',
            help="This is the image that will be analyzed for defects"
        )
        
        if test_file:
            test_image = Image.open(test_file)
            st.session_state.test_image = test_image
            st.image(test_image, caption="✅ Test image loaded", use_container_width=True)
    
    return template_file, test_file


def render_process_button(template_file, test_file, settings: Dict):
    """Render the process button and handle detection."""
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🔍 Detect Defects", use_container_width=True, type="primary"):
            if template_file and test_file:
                process_images(settings)
            else:
                missing = []
                if not template_file:
                    missing.append("template image")
                if not test_file:
                    missing.append("test image")
                st.warning(f"⚠️ Please upload: {', '.join(missing)}")


def check_api_server() -> bool:
    """Check if the FastAPI server is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if response.status_code == 200:
            data = response.json()
            return data.get('model_loaded', False)
        return False
    except Exception:
        return False


def process_images(settings: Dict):
    """Process images and run defect detection using FastAPI backend."""
    # Create progress container
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Check API server
            status_text.markdown("🔄 **Connecting to detection server...**")
            progress_bar.progress(10)
            
            if not check_api_server():
                raise ConnectionError(
                    "Detection server not available. Please start the FastAPI server:\n"
                    "`uvicorn src.api.server:app --host 0.0.0.0 --port 8000`"
                )
            
            # Step 2: Prepare images
            status_text.markdown("🖼️ **Preparing images...**")
            progress_bar.progress(25)
            
            # Convert PIL images to bytes
            template_img = st.session_state.template_image
            test_img = st.session_state.test_image
            
            template_buffer = io.BytesIO()
            template_img.save(template_buffer, format='JPEG', quality=95)
            template_bytes = template_buffer.getvalue()
            
            test_buffer = io.BytesIO()
            test_img.save(test_buffer, format='JPEG', quality=95)
            test_bytes = test_buffer.getvalue()
            
            # Step 3: Send to API
            status_text.markdown("🔍 **Running AI detection...**")
            progress_bar.progress(50)
            
            # Make API request
            files = {
                'template': ('template.jpg', template_bytes, 'image/jpeg'),
                'test': ('test.jpg', test_bytes, 'image/jpeg')
            }
            
            params = {
                'confidence_threshold': settings['confidence_threshold'],
                'include_images': True
            }
            
            response = requests.post(
                f"{API_BASE_URL}/detect",
                files=files,
                params=params,
                timeout=60
            )
            
            # Step 4: Process response
            status_text.markdown("📊 **Processing results...**")
            progress_bar.progress(80)
            
            if response.status_code != 200:
                raise Exception(f"API error: {response.json().get('detail', 'Unknown error')}")
            
            data = response.json()
            
            # Decode annotated image
            annotated_image = None
            if data.get('annotated_image_base64'):
                img_bytes = base64.b64decode(data['annotated_image_base64'])
                nparr = np.frombuffer(img_bytes, np.uint8)
                annotated_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            
            # Decode difference map
            difference_map = None
            if data.get('difference_map_base64'):
                img_bytes = base64.b64decode(data['difference_map_base64'])
                nparr = np.frombuffer(img_bytes, np.uint8)
                difference_map = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Decode mask
            mask_image = None
            if data.get('mask_base64'):
                img_bytes = base64.b64decode(data['mask_base64'])
                nparr = np.frombuffer(img_bytes, np.uint8)
                mask_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Build results
            results = {
                'success': data.get('success', False),
                'num_defects': data.get('num_defects', 0),
                'defects': data.get('defects', []),
                'processing_time': data.get('processing_time', 0.0),
                'image_size': data.get('image_size', {}),
                'timestamp': data.get('timestamp', ''),
                'summary': data.get('summary', {}),
                'annotated_image': annotated_image,
                'difference_map': difference_map,
                'mask': mask_image
            }
            
            # Step 5: Complete
            status_text.markdown("✅ **Detection complete!**")
            progress_bar.progress(100)
            time.sleep(0.3)
            
            # Clear progress
            progress_bar.empty()
            status_text.empty()
            
            # Store results
            st.session_state.current_results = results
            
            # Add to history
            st.session_state.processing_history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'num_defects': results['num_defects'],
                'processing_time': results['processing_time'],
                'defect_types': results['summary'].get('defect_types', [])
            })
            
            logger.info(f"Processing complete - Defects found: {results['num_defects']}")
            st.success(f"✅ Detection complete in {results['processing_time']:.2f} seconds! Found {results['num_defects']} defects.")
            st.rerun()
            
        except ConnectionError as e:
            progress_bar.empty()
            status_text.empty()
            logger.error(f"Connection Error: {str(e)}")
            st.error(f"❌ Connection Error: {str(e)}")
            st.info("💡 **Tip:** Make sure the FastAPI server is running on port 8000")
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            logger.error(f"Processing failed: {str(e)}", exc_info=True)
            st.error(f"❌ Error during processing: {str(e)}")


def render_results_section(settings: Dict):
    """Render the results section with enhanced visualizations."""
    results = st.session_state.get('current_results')
    
    if not results:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 3rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📊</div>
            <h3 style="color: #2c3e50;">No Results Yet</h3>
            <p style="color: #5a6c7d;">
                Upload template and test images, then click "Detect Defects" to see results
            </p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown('<h2 class="section-header">📊 Detection Results</h2>', unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_metric_card(
            str(results['num_defects']), 
            "Defects Found", 
            "🔴" if results['num_defects'] > 0 else "✅"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_metric_card(
            f"{results['processing_time']:.2f}s",
            "Processing Time",
            "⚡"
        ), unsafe_allow_html=True)
    
    with col3:
        if results['defects']:
            avg_conf = np.mean([d.get('confidence', 0) for d in results['defects']])
            st.markdown(create_metric_card(f"{avg_conf:.0%}", "Avg Confidence", "📈"), unsafe_allow_html=True)
        else:
            st.markdown(create_metric_card("N/A", "Avg Confidence", "📈"), unsafe_allow_html=True)
    
    with col4:
        unique_types = len(set(d.get('class', '') for d in results['defects']))
        st.markdown(create_metric_card(str(unique_types), "Defect Types", "🏷️"), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create subtabs for different views
    result_tabs = st.tabs(["📷 Detection View", "📊 Analytics", "📋 Defect List", "📥 Export"])
    
    # Tab 1: Detection View
    with result_tabs[0]:
        if results.get('annotated_image') is not None:
            # View mode selector
            view_mode = st.radio(
                "View Mode",
                ["Annotated", "Original vs Annotated", "Heatmap"],
                horizontal=True,
                key="view_mode"
            )
            
            if view_mode == "Annotated":
                st.image(results['annotated_image'], caption="Detection Results", use_container_width=True)
            
            elif view_mode == "Original vs Annotated":
                # Show side-by-side comparison
                if st.session_state.get('test_image'):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(st.session_state.test_image, caption="Original Test Image", use_container_width=True)
                    with col2:
                        st.image(results['annotated_image'], caption="With Detections", use_container_width=True)
                else:
                    st.image(results['annotated_image'], caption="Detection Results", use_container_width=True)
            
            elif view_mode == "Heatmap":
                if VIZ_AVAILABLE and results['defects']:
                    img_size = results.get('image_size', {})
                    width = img_size.get('width', 800)
                    height = img_size.get('height', 600)
                    heatmap = create_defect_heatmap(
                        results['defects'], 
                        (width, height),
                        (min(width, 800), min(height, 600))
                    )
                    st.image(heatmap, caption="Defect Concentration Heatmap", use_container_width=True)
                else:
                    st.info("Heatmap visualization requires defects to be detected.")
        
        # Intermediate images
        if settings.get('show_intermediate'):
            with st.expander("🔬 Processing Steps"):
                col1, col2 = st.columns(2)
                
                with col1:
                    if results.get('difference_map') is not None:
                        st.image(results['difference_map'], caption="Difference Map", use_container_width=True)
                    else:
                        st.info("Difference map not available")
                
                with col2:
                    if results.get('mask') is not None:
                        st.image(results['mask'], caption="Binary Mask", use_container_width=True)
                    else:
                        st.info("Binary mask not available")
    
    # Tab 2: Analytics
    with result_tabs[1]:
        if results['defects']:
            from collections import Counter
            class_counts = Counter(d.get('class', 'Unknown') for d in results['defects'])
            
            # Charts row
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="glass-card">
                    <h4 style="color: #2c3e50; margin-bottom: 1rem;">📊 Defects by Type</h4>
                </div>
                """, unsafe_allow_html=True)
                
                if VIZ_AVAILABLE:
                    bar_svg = create_bar_chart_svg(dict(class_counts), width=450, height=280)
                    st.markdown(bar_svg, unsafe_allow_html=True)
                else:
                    # Fallback to simple display
                    for dtype, count in sorted(class_counts.items(), key=lambda x: -x[1]):
                        st.write(f"**{dtype.replace('_', ' ')}**: {count}")
            
            with col2:
                st.markdown("""
                <div class="glass-card">
                    <h4 style="color: #2c3e50; margin-bottom: 1rem;">🥧 Distribution</h4>
                </div>
                """, unsafe_allow_html=True)
                
                if VIZ_AVAILABLE:
                    pie_svg = create_pie_chart_svg(dict(class_counts), width=320, height=320)
                    st.markdown(pie_svg, unsafe_allow_html=True)
                else:
                    total = sum(class_counts.values())
                    for dtype, count in sorted(class_counts.items(), key=lambda x: -x[1]):
                        pct = (count / total * 100) if total > 0 else 0
                        st.write(f"**{dtype.replace('_', ' ')}**: {pct:.1f}%")
            
            # Confidence histogram
            st.markdown("""
            <div class="glass-card" style="margin-top: 1rem;">
                <h4 style="color: #2c3e50; margin-bottom: 1rem;">📈 Confidence Distribution</h4>
            </div>
            """, unsafe_allow_html=True)
            
            confidences = [d.get('confidence', 0) for d in results['defects']]
            
            if VIZ_AVAILABLE:
                hist_svg = create_confidence_histogram_svg(confidences, width=800, height=200)
                st.markdown(hist_svg, unsafe_allow_html=True)
            
            # Statistics summary
            with st.expander("📊 Detailed Statistics"):
                if VIZ_AVAILABLE:
                    stats = calculate_statistics(results)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Confidence Stats**")
                        st.write(f"- Min: {stats['confidence']['min']:.1%}")
                        st.write(f"- Max: {stats['confidence']['max']:.1%}")
                        st.write(f"- Mean: {stats['confidence']['mean']:.1%}")
                        st.write(f"- Std Dev: {stats['confidence']['std']:.3f}")
                    
                    with col2:
                        st.markdown("**Area Stats**")
                        st.write(f"- Min: {stats['area']['min']} px²")
                        st.write(f"- Max: {stats['area']['max']} px²")
                        st.write(f"- Mean: {stats['area']['mean']:.0f} px²")
                        st.write(f"- Total: {stats['area']['total']} px²")
                    
                    with col3:
                        st.markdown("**Spatial Distribution**")
                        st.write(f"- Center X: {stats['spatial']['x_mean']:.0f}")
                        st.write(f"- Center Y: {stats['spatial']['y_mean']:.0f}")
                else:
                    st.write("Install visualization utilities for detailed statistics.")
        else:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 2rem; background: rgba(39, 174, 96, 0.1); border: 1px solid rgba(39, 174, 96, 0.3);">
                <div style="font-size: 3rem; margin-bottom: 1rem;">✅</div>
                <h3 style="color: #27ae60;">No Defects to Analyze</h3>
                <p style="color: #5a6c7d;">The PCB appears to be defect-free.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Tab 3: Defect List
    with result_tabs[2]:
        if results['defects']:
            # Filters
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                min_conf = st.slider("Minimum Confidence", 0.0, 1.0, 0.0, 0.05, key="filter_conf")
            
            with col2:
                from collections import Counter
                class_counts = Counter(d.get('class', 'Unknown') for d in results['defects'])
                all_types = ['All'] + list(class_counts.keys())
                filter_type = st.selectbox("Filter by Type", all_types, key="filter_type")
            
            with col3:
                sort_order = st.selectbox("Sort By", ["Index", "Confidence", "Area"], key="sort_order")
            
            # Filter defects
            filtered = results['defects']
            if min_conf > 0:
                filtered = [d for d in filtered if d.get('confidence', 0) >= min_conf]
            if filter_type != 'All':
                filtered = [d for d in filtered if d.get('class') == filter_type]
            
            # Sort defects
            if sort_order == "Confidence":
                filtered = sorted(filtered, key=lambda x: -x.get('confidence', 0))
            elif sort_order == "Area":
                filtered = sorted(filtered, key=lambda x: -x.get('area', 0))
            
            st.markdown(f"**Showing {len(filtered)} of {len(results['defects'])} defects**")
            
            # Display as table
            for defect in filtered[:50]:
                class_name = defect.get('class', 'Unknown')
                display_name = class_name.replace('_', ' ')
                confidence = defect.get('confidence', 0)
                bbox = defect.get('bbox', {})
                area = defect.get('area', 0)
                
                # Color dot based on class
                color_rgb = DEFECT_COLORS.get(class_name, (156, 163, 175)) if VIZ_AVAILABLE else (100, 100, 100)
                color_hex = f'#{color_rgb[0]:02x}{color_rgb[1]:02x}{color_rgb[2]:02x}'
                
                st.markdown(f"""
                <div style="
                    display: flex; 
                    justify-content: space-between; 
                    align-items: center;
                    padding: 0.75rem 1rem; 
                    margin: 0.25rem 0;
                    background: #ffffff;
                    border-radius: 8px;
                    border-left: 4px solid {color_hex};
                    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
                    border: 1px solid rgba(168, 181, 183, 0.3);
                ">
                    <div style="display: flex; align-items: center; gap: 1rem;">
                        <span style="color: #5a6c7d; font-weight: bold;">#{defect.get('index', 0)}</span>
                        <span style="color: {color_hex}; font-weight: 600;">{display_name}</span>
                    </div>
                    <div style="display: flex; gap: 2rem; color: #5a6c7d; font-size: 0.9rem;">
                        <span>Conf: <strong style="color: #5a6f72;">{confidence:.0%}</strong></span>
                        <span>Area: {area} px²</span>
                        <span>Pos: ({bbox.get('x', 0)}, {bbox.get('y', 0)})</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            if len(filtered) > 50:
                st.info(f"Showing first 50 of {len(filtered)} defects. Export for full list.")
        else:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 2rem; background: rgba(39, 174, 96, 0.1); border: 1px solid rgba(39, 174, 96, 0.3);">
                <div style="font-size: 3rem; margin-bottom: 1rem;">✅</div>
                <h3 style="color: #27ae60;">No Defects Found</h3>
            </div>
            """, unsafe_allow_html=True)
    
    # Tab 4: Export
    with result_tabs[3]:
        render_export_section(results, settings)


def render_export_section(results: Dict, settings: Dict):
    """Render the export section with download options."""
    exports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")
    os.makedirs(exports_dir, exist_ok=True)

    st.markdown(f"""
    <div class="glass-card">
        <h4 style="color: #2c3e50; margin-bottom: 1rem;">📥 Export Detection Results</h4>
        <p style="color: #5a6c7d; font-size: 0.9rem;">
            Download detection results in various formats for reporting and analysis.
            <br><strong>✅ Files are automatically saved to:</strong> <code style="color: #e67e22; background: #fdf2e9; padding: 2px 5px; border-radius: 4px;">{exports_dir}</code>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if not EXPORT_AVAILABLE:
        st.warning("Export utilities not available. Please check installation.")
        return
    
    # Get timestamp for filenames
    # Get timestamp for filenames
    from datetime import datetime
    # timestamp already imported globally or available? No, imported inside function usually.
    # Actually datetime is imported at top level too (line 31).
    # But let's check what I wrote.
    # I wrote: from datetime import datetime\n    import os\n    timestamp = ...
    
    # Just removing import os:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create exports directory if it doesn't exist
    exports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")
    os.makedirs(exports_dir, exist_ok=True)
    
    # Export format selection
    st.markdown("### Choose Export Format")
    
    col1, col2, col3 = st.columns(3)
    
    # Column 1: Images
    with col1:
        st.markdown("""
        <div style="background: rgba(168, 181, 183, 0.2); padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid rgba(168, 181, 183, 0.3);">
            <h5 style="color: #2c3e50; margin: 0;">🖼️ Images</h5>
        </div>
        """, unsafe_allow_html=True)
        
        # Annotated image
        if results.get('annotated_image') is not None:
            try:
                img_array = np.array(results['annotated_image'])
                img_bytes = numpy_to_bytes(img_array, 'png')
                
                # Save to local exports folder
                ann_filename = f"pcb_annotated_{timestamp}.png"
                ann_filepath = os.path.join(exports_dir, ann_filename)
                with open(ann_filepath, 'wb') as f:
                    f.write(img_bytes)
                
                st.download_button(
                    label="📷 Annotated Image (PNG)",
                    data=img_bytes,
                    file_name=ann_filename,
                    mime="image/png",
                    key="dl_annotated",
                    use_container_width=True
                )
                st.caption(f"✅ Saved: {ann_filename}")
            except Exception as e:
                st.error(f"Error preparing image: {e}")
        
        # Comparison image
        if results.get('annotated_image') is not None and st.session_state.get('test_image'):
            try:
                orig_array = np.array(st.session_state.test_image)
                ann_array = np.array(results['annotated_image'])
                comparison = create_comparison_image(orig_array, ann_array)
                comp_bytes = numpy_to_bytes(comparison, 'png')
                
                # Save to local exports folder
                comp_filename = f"pcb_comparison_{timestamp}.png"
                comp_filepath = os.path.join(exports_dir, comp_filename)
                with open(comp_filepath, 'wb') as f:
                    f.write(comp_bytes)
                
                st.download_button(
                    label="↔️ Comparison Image (PNG)",
                    data=comp_bytes,
                    file_name=comp_filename,
                    mime="image/png",
                    key="dl_comparison",
                    use_container_width=True
                )
                st.caption(f"✅ Saved: {comp_filename}")
            except Exception as e:
                st.error(f"Error creating comparison: {e}")
    
    # Column 2: Data Exports
    with col2:
        st.markdown("""
        <div style="background: rgba(168, 181, 183, 0.2); padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid rgba(168, 181, 183, 0.3);">
            <h5 style="color: #2c3e50; margin: 0;">📊 Data</h5>
        </div>
        """, unsafe_allow_html=True)
        
        # CSV Export (detailed)
        csv_data = export_defects_csv(
            results.get('defects', []),
            {'timestamp': timestamp, 'processing_time': results.get('processing_time', 0)}
        )
        
        # Save CSV to local exports folder
        csv_filename = f"pcb_defects_{timestamp}.csv"
        csv_filepath = os.path.join(exports_dir, csv_filename)
        with open(csv_filepath, 'w') as f:
            f.write(csv_data)
        
        st.download_button(
            label="📋 Defects CSV",
            data=csv_data,
            file_name=csv_filename,
            mime="text/csv",
            key="dl_csv",
            use_container_width=True
        )
        st.caption(f"✅ Saved: {csv_filename}")
        
        # CSV Summary
        summary_csv = export_summary_csv(results.get('defects', []))
        
        # Save summary CSV to local exports folder
        summary_filename = f"pcb_summary_{timestamp}.csv"
        summary_filepath = os.path.join(exports_dir, summary_filename)
        with open(summary_filepath, 'w') as f:
            f.write(summary_csv)
        
        st.download_button(
            label="📈 Summary CSV",
            data=summary_csv,
            file_name=summary_filename,
            mime="text/csv",
            key="dl_summary",
            use_container_width=True
        )
        st.caption(f"✅ Saved: {summary_filename}")
        
        # JSON Export
        json_data = export_results_json(results)
        
        # Save JSON to local exports folder
        json_filename = f"pcb_detection_{timestamp}.json"
        json_filepath = os.path.join(exports_dir, json_filename)
        with open(json_filepath, 'w') as f:
            f.write(json_data)
        
        st.download_button(
            label="🔧 Full JSON Data",
            data=json_data,
            file_name=json_filename,
            mime="application/json",
            key="dl_json",
            use_container_width=True
        )
        st.caption(f"✅ Saved: {json_filename}")
    
    # Column 3: Reports
    with col3:
        st.markdown("""
        <div style="background: rgba(168, 181, 183, 0.2); padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid rgba(168, 181, 183, 0.3);">
            <h5 style="color: #2c3e50; margin: 0;">📝 Reports</h5>
        </div>
        """, unsafe_allow_html=True)
        
        # Text Report
        text_report = export_text_report(results)
        
        # Save text report to local exports folder
        report_filename = f"pcb_report_{timestamp}.txt"
        report_filepath = os.path.join(exports_dir, report_filename)
        with open(report_filepath, 'w') as f:
            f.write(text_report)
        
        st.download_button(
            label="📄 Text Report",
            data=text_report,
            file_name=report_filename,
            mime="text/plain",
            key="dl_text",
            use_container_width=True
        )
        st.caption(f"✅ Saved: {report_filename}")
        
        # Quick stats display
        st.markdown("---")
        st.markdown("**Quick Stats**")
        st.write(f"📊 {results.get('num_defects', 0)} defects detected")
        st.write(f"⏱️ {results.get('processing_time', 0):.2f}s processing time")
        
        if results.get('defects'):
            from collections import Counter
            types = Counter(d.get('class', 'Unknown') for d in results['defects'])
            st.write(f"🏷️ {len(types)} defect types")
    
    # Export all button
    st.markdown("---")
    
    with st.expander("📦 Export All Formats"):
        st.info("Click the button below to save all export formats to the outputs directory.")
        
        if st.button("💾 Save All to Disk", key="export_all"):
            try:
                output_dir = PROJECT_ROOT / 'outputs' / 'exports' / timestamp
                
                annotated_img = np.array(results['annotated_image']) if results.get('annotated_image') else None
                original_img = np.array(st.session_state.test_image) if st.session_state.get('test_image') else None
                
                exports = export_all_formats(
                    results,
                    annotated_image=annotated_img,
                    original_image=original_img,
                    output_dir=output_dir
                )
                
                st.success(f"✅ Exported {len(exports)} files to: `{output_dir}`")
                
                # List exported files
                for key, export in exports.items():
                    st.write(f"  • {export['filename']}")
                    
            except Exception as e:
                st.error(f"❌ Export failed: {e}")


def render_history_section():
    """Render the processing history section."""
    st.markdown('<h2 class="section-header">📜 Processing History</h2>', unsafe_allow_html=True)
    
    history = st.session_state.get('processing_history', [])
    
    if not history:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 2rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📝</div>
            <p style="color: #5a6c7d;">No processing history yet</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Display history in reverse order (most recent first)
    for i, entry in enumerate(reversed(history)):
        idx = len(history) - i
        defect_count = entry['num_defects']
        status_icon = "🔴" if defect_count > 0 else "✅"
        
        with st.expander(f"{status_icon} Run #{idx} - {entry['timestamp']} ({defect_count} defects)"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Defects Found", entry['num_defects'])
            with col2:
                st.metric("Processing Time", f"{entry['processing_time']:.2f}s")
            
            # Show defect types if available
            defect_types = entry.get('defect_types', [])
            if defect_types:
                st.markdown("**Defect Types:**")
                for dtype in defect_types:
                    st.markdown(f"- {dtype.replace('_', ' ')}")
    
    # Clear history button
    if st.button("🗑️ Clear History", key="clear_history"):
        st.session_state.processing_history = []
        st.rerun()


def render_footer():
    """Render the footer section."""
    st.markdown("""
    <div class="footer">
        <p style="color: #2c3e50;">
            <strong>PCB Defect Detection System</strong> | 
            Powered by EfficientNet-B3 | 
            99.46% Accuracy
        </p>
        <p style="font-size: 0.85rem; margin-top: 0.5rem; color: #5a6c7d;">
            Built with ❤️ using 
            <span style="color: #e74c3c; font-weight: 500;">Streamlit</span> • 
            <span style="color: #16a085; font-weight: 500;">FastAPI</span> • 
            <span style="color: #c0392b; font-weight: 500;">PyTorch</span> • 
            <span style="color: #8e44ad; font-weight: 500;">OpenCV</span>
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    """Main application entry point."""
    # Initialize session state first to get theme preference
    init_session_state()
    
    # Load custom CSS with theme
    load_custom_css(st.session_state.dark_mode)
    
    # Render sidebar and get settings
    settings = render_sidebar()
    
    # Render main header
    render_header()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Detect", "📊 Results", "📜 History"])
    
    with tab1:
        # Upload section
        template_file, test_file = render_upload_section()
        
        # Process button
        render_process_button(template_file, test_file, settings)
    
    with tab2:
        # Results section
        render_results_section(settings)
    
    with tab3:
        # History section
        render_history_section()
    
    # Render footer
    render_footer()


if __name__ == "__main__":
    main()
