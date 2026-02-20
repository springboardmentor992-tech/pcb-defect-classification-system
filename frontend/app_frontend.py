import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="PCB Vision AI", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔍 PCB Defect Detection System")
st.write("Upload a **Template (Golden)** image and a **Test** image to identify manufacturing flaws.")

col1, col2 = st.columns(2)

with st.sidebar:
    st.header("Upload Center")
    template_file = st.file_uploader("Upload Template image", type=['jpg', 'jpeg', 'png'])
    test_file = st.file_uploader("Upload Test Board", type=['jpg', 'jpeg', 'png'])
    inspect_btn = st.button("🚀 Run Inspection", use_container_width=True)

if template_file and test_file:
    col1.image(template_file, caption="Template Image", use_container_width=True)
    col2.image(test_file, caption="Test Image", use_container_width=True)

    if inspect_btn:
        with st.spinner("Analyzing circuitry..."):

            files = {
                "template": template_file.getvalue(),
                "test": test_file.getvalue()
            }

            try:
                response = requests.post("http://localhost:8000/inspect", files=files)
                data = response.json()

                if data["status"] == "success":
                    detections = data["detections"]

                    img = Image.open(test_file).convert("RGB")
                    draw = ImageDraw.Draw(img)

                    # ✅ Load Bigger Font
                    try:
                        font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 40)
                    except:
                        font = ImageFont.load_default()

                    total_defects = len(detections)

                    # 🔴 Draw Red Boxes + Big Labels
                    for det in detections:
                        x, y, w, h = det['box']

                        draw.rectangle([x, y, x+w, y+h], outline="red", width=10)

                        draw.text(
                            (x, y-50),
                            det['label'].upper(),
                            fill="yellow",
                            font=font
                        )

                    # 🔥 Total Defect Count at Top
                    # draw.text(
                    #     (30, 30),
                    #     f"TOTAL DEFECTS: {total_defects}",
                    #     fill="red",
                    #     font=font
                    # )

                    st.subheader("Defects Detected successfully")
                    st.image(img, use_container_width=True)

                else:
                    st.error("Inspection failed.")

            except Exception as e:
                st.error(f"Connection Error: Ensure the FastAPI backend is running. {e}")