import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw

# Custom CSS for Blue, White, and Black Contrast
st.markdown("""
<style>
    body {
        background: linear-gradient(to right, #0066ff, #ffffff, #000000);
        color: #ffffff;
        font-family: Arial, sans-serif;
    }
    .main { 
        background: #ffffff; 
        color: #000000; 
        padding: 20px;
    }
    h1 {
        color: #ffffff;
        text-align: center;
    }
    .upload-btn {
        background-color: #0066ff !important;
        color: white !important;
    }
    .sidebar .sidebar-content {
        background: #000000 !important;
        color: #ffffff;
    }
    .footer {
        background-color: #000000;
        color: #ffffff;
        padding: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ✅ Correct way to load YOLO model
@st.cache_resource
def load_model():
    model_path = r"E:\PROJECT\acne_yolo project\acne\Acne-Detector\acne-computer-vision_model.pt"
    return YOLO(model_path)  # 🔥 Use ultralytics.YOLO instead of torch.hub.load

model = load_model()

# Main App
st.title("🧴 Acne Detection AI")
st.subheader("Upload a face image to detect acne spots")

# Sidebar
with st.sidebar:
    st.header("Settings")
    confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)

# Image Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], key="uploader")

if uploaded_file is not None:
    # Read Image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    # Run Detection ✅
    results = model.predict(img_array)
    
    # Draw bounding boxes
    annotated_img = img_array.copy()
    draw = ImageDraw.Draw(image)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    # Display Results ✅
    st.image(image, caption="Detection Results", use_column_width=True)

    # Show Detection Details
    st.subheader("Detection Statistics")
    num_acne = len(results[0].boxes)
    st.write(f"Detected Acne Spots: {num_acne}")

# Footer with "Data Scientist"
st.markdown("""
<div class="footer">
    <p>Made with ❤️ by MOHID_KHAN, Data Scientist</p>
    <p>Email: Mohidadil24@gmail.com |Github:https://github.com/mohidadil| Reviews: Excellent work on AI-based acne detection models</p>
</div>
""", unsafe_allow_html=True)
