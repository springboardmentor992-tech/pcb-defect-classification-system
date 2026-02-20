import streamlit as st
import requests
import base64
import io
from PIL import Image

# --- CONFIGURATION ---
API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="PCB Defect Detector", layout="wide")

st.title("🔍 PCB Defect Detection & Classification")
st.markdown("Upload a **Defect Image** and its corresponding **Template** to analyze manufacturing flaws.")

# --- SIDEBAR: CONTROLS ---
with st.sidebar:
    st.header("Upload Images")
    defect_file = st.file_uploader("Upload Defect Image", type=["jpg", "png", "jpeg"])
    template_file = st.file_uploader("Upload Template Image", type=["jpg", "png", "jpeg"])
    
    st.divider()
    return_images = st.checkbox("Return Annotated Images", value=True)
    analyze_btn = st.button("🚀 Run Analysis", use_container_width=True)

# --- MAIN UI LAYOUT ---
col1, col2 = st.columns(2)

if defect_file and template_file:
    with col1:
        st.subheader("Input: Defect Image")
        st.image(defect_file, use_container_width=True)
    with col2:
        st.subheader("Input: Template Image")
        st.image(template_file, use_container_width=True)

    if analyze_btn:
        with st.spinner("Analyzing PCB..."):
            # Prepare files for the API request
            files = {
                "defect_image": (defect_file.name, defect_file.getvalue(), defect_file.type),
                "template_image": (template_file.name, template_file.getvalue(), template_file.type)
            }
            data = {"return_images": str(return_images).lower()}

            try:
                response = requests.post(API_URL, files=files, data=data)
                response.raise_for_status()
                result = response.json()

                st.divider()
                
                # --- DISPLAY RESULTS ---
                if result["success"]:
                    res_col1, res_col2 = st.columns([2, 1])

                    with res_col1:
                        st.subheader("Detection Result")
                        if "annotated_image_base64" in result and result["annotated_image_base64"]:
                            img_bytes = base64.b64decode(result["annotated_image_base64"])
                            st.image(img_bytes, caption="Annotated Defects", use_container_width=True)
                        
                        # Show the threshold map for technical debugging
                        with st.expander("View Subtraction Threshold Map"):
                            thresh_bytes = base64.b64decode(result["threshold_image_base64"])
                            st.image(thresh_bytes, caption="Binary Difference Map")

                    with res_col2:
                        st.subheader("Defect Summary")
                        st.metric("Total Defects", result["num_defects"])
                        st.metric("Processing Time", f"{result['processing_time']}s")
                        
                        if result["defects"]:
                            # Display defects in a clean table
                            st.dataframe(
                                result["defects"], 
                                column_config={
                                    "id": "ID",
                                    "class_name": "Defect Type",
                                    "confidence": st.column_config.ProgressColumn(
                                        "Confidence", format="%.2f", min_value=0, max_value=1
                                    ),
                                    "bbox": "Bounding Box"
                                },
                                hide_index=True
                            )
                        else:
                            st.success("No defects detected!")
                
            except Exception as e:
                st.error(f"Error connecting to API: {e}")
else:
    st.info("Please upload both the defect and template images to begin.")