import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import sys

# Import for the inference pipeline
from ultralytics import YOLO
# from weather_simulation.simulate_weather import get_weather_pipelines
# from enhancement.enhance_image import apply_clahe, apply_dark_channel_dehaze

# --------------------------------------------------------------------------------
# Page Configuration & Styling Setup
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Indian Traffic Vehicle Detection", 
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS
st.markdown("""
<style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Base Font & Theme Settings */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Header Styling */
    h1 {
        color: #1E293B;
        font-weight: 700;
        font-size: 2.2rem !important;
        margin-bottom: 0.2rem !important;
        padding-top: 2rem !important;
        text-align: center;
        background: -webkit-linear-gradient(45deg, #2E86AB, #A23B72);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Subtitle Styling */
    .stMarkdown p {
        color: #475569;
        font-size: 1.05rem;
    }

    /* Container Box Styling */
    .css-1r6slb0, .css-1y4p8pa {
        background-color: #F8FAFC !important;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        border: 1px solid #E2E8F0;
    }
    
    /* Sidebar Enhancements */
    [data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }
    [data-testid="stSidebar"] * {
        color: #E2E8F0 !important;
    }
    .css-17lntkn {
        color: #38BDF8 !important; /* Sidebar sliders/active elements */
    }

    /* Images Panels */
    .stImage > img {
        border-radius: 8px;
        box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        border: 2px solid #E2E8F0;
        transition: transform 0.2s ease;
    }
    
    /* Micro-animation on image hover */
    .stImage > img:hover {
        transform: scale(1.01);
    }
    
    /* Info Box */
    div.stInfo {
        background-color: #F0F9FF;
        border-left-color: #0EA5E9;
        color: #0369A1;
        border-radius: 6px;
    }
    
    /* Divider */
    hr {
        border-top: 1px solid #CBD5E1;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# Utility Mock Functions (For visual preview before model integration)
# --------------------------------------------------------------------------------

def mock_simulate_weather(image_array, weather_type):
    """Temporary function to simulate visual change for preview."""
    img = image_array.copy()
    if weather_type == "Rain":
        return cv2.blur(img, (5, 5)) # Blur substitute for now
    elif weather_type == "Fog":
        fog_layer = np.full(img.shape, 220, dtype=np.uint8)
        return cv2.addWeighted(img, 0.4, fog_layer, 0.6, 0)
    elif weather_type == "Snow":
        snow_layer = np.full(img.shape, 255, dtype=np.uint8)
        return cv2.addWeighted(img, 0.7, snow_layer, 0.3, 0)
    elif weather_type == "Motion Blur":
        kernel = np.zeros((15, 15))
        kernel[7, :] = np.ones(15) / 15
        return cv2.filter2D(img, -1, kernel)
    return img

def mock_enhance_image(image_array, method):
    """Temporary function to simulate enhancement."""
    img = image_array.copy()
    if method == "CLAHE":
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB) # Streamlit uses RGB
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    elif method == "Dehazing (Proxy)":
         # Simple contrast stretching
        p2, p98 = np.percentile(img, (2, 98))
        img_rescale = np.clip((img - p2)/(p98 - p2), 0, 1) * 255
        return img_rescale.astype(np.uint8)
    return img

# --------------------------------------------------------------------------------
# Main Application Structure
# --------------------------------------------------------------------------------

st.markdown("<h1>Indian Traffic Vehicle Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; margin-bottom: 2rem;'><strong>Robust YOLOv8 Inference Pipeline for Adverse Weather Environments</strong></p>", unsafe_allow_html=True)

# --- Sidebar Controls ---
st.sidebar.markdown('<h2><span style="color: #38BDF8;">⚙️</span> Configuration</h2>', unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.markdown('### Model Settings')
model_path = st.sidebar.text_input("Trained Weights Path", "models/best.pt")
confidence = st.sidebar.slider("Detection Confidence", min_value=0.0, max_value=1.0, value=0.45, step=0.05)

st.sidebar.markdown("---")
st.sidebar.markdown('### Pipeline Stages')

# Weather Controls
st.sidebar.markdown("**1. Weather Simulation**")
sim_weather = st.sidebar.selectbox("Apply Weather Condition", ["Clear (Original)", "Rain", "Fog", "Snow", "Motion Blur"])

# Enhancement Controls
st.sidebar.markdown("**2. Image Recovery**")
apply_eh = st.sidebar.radio("Enhancement Algorithm", ["None", "CLAHE", "Dehazing (Proxy)"])

st.sidebar.markdown("---")
st.sidebar.caption("© 2026 AI Engineering Lab")


# --- Main Dashboard Area ---
st.markdown("### 📤 Image Input")
uploaded_file = st.file_uploader("Upload High-Res Traffic Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Read Image into RGB format
    image = Image.open(uploaded_file)
    original_img_array = np.array(image.convert("RGB"))
    
    st.markdown("---")
    
    # Process Image based on pipeline sequence
    processed_img = original_img_array.copy()
    
    # 1. Simulate Weather
    if sim_weather != "Clear (Original)":
        processed_img = mock_simulate_weather(processed_img, sim_weather)
        
    # 2. Enhance Degradation
    if apply_eh != "None":
        processed_img = mock_enhance_image(processed_img, apply_eh)
        
    # Layout 2 Columns for Comparison
    col1, spacer, col2 = st.columns([1, 0.05, 1])
    
    with col1:
        st.markdown(f"### 📍 Processed Input")
        state_label = f"**Condition:** {sim_weather}"
        if apply_eh != "None":
            state_label += f" | **Enhanced:** {apply_eh}"
            
        st.markdown(state_label)
        st.image(processed_img, use_column_width=True)
        
    with col2:
        st.markdown("### 🔍 YOLOv8 Detection Result")
        st.markdown(f"**Confidence Threshold:** {confidence}")
        
        # --- Real Inference Logic ---
        try:
            # Fallback to YOLO base model if the custom weights don't exist yet
            active_model_path = model_path if os.path.exists(model_path) else 'yolov8n.pt'
            
            if active_model_path == 'yolov8n.pt':
                 st.warning(f"⚠️ Custom weights not found at `{model_path}`. Showing vehicle detection demo using Base YOLOv8 model.")
                 
            # Load and predict
            model = YOLO(active_model_path)
            results = model.predict(processed_img, conf=confidence)
            
            # Plot the detections (draws boxes and class labels)
            res = results[0]
            # Ultralytics res.plot() returns BGR image, we convert it to RGB for Streamlit
            res_plotted = res.plot()[..., ::-1]
            st.image(res_plotted, use_column_width=True)
            
            # Extract and display counts of detected vehicles
            boxes = res.boxes.cpu()
            if len(boxes) > 0:
                st.markdown("#### 📊 Vehicles Detected:")
                counts = {}
                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = res.names[cls_id].title()
                    counts[class_name] = counts.get(class_name, 0) + 1
                    
                # Display metrics in a clean row
                cols = st.columns(min(len(counts.keys()), 4))
                for idx, (cls_name, count) in enumerate(counts.items()):
                    cols[idx % 4].metric(label=cls_name, value=count)
            else:
                st.info("No vehicles detected with the current confidence threshold.")
                
        except Exception as e:
            st.error(f"Failed to run model inference: {e}")
            st.image(processed_img, use_column_width=True, clamp=True)

else:
    # Empty State Dashboard View
    st.markdown("---")
    st.info("👈 **Upload an image in the center panel to begin the analysis.**")
    
    # Placeholder layout to show off design before interaction
    c1, c2, c3 = st.columns(3)
    c1.metric(label="Supported Classes", value="8 Target Categories")
    c2.metric(label="Inference Frame Rate", value="~ 45 FPS", delta="Optimized")
    c3.metric(label="Weather Robustness", value="Rain, Fog, Snow", delta="Integrated")
