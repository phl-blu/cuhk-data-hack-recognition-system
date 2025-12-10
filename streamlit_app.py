import os
import time
import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image
from feature_extractor import FeatureExtractor

# Page setup
st.set_page_config(page_title="Waste Classification", page_icon="♻️", layout="wide")
st.markdown("<h1 style='text-align:center;color:#2E7D32'>♻️ Material Stream Identification</h1>", unsafe_allow_html=True)
st.markdown("---")

# Load Model + Feature Extractor
@st.cache_resource
def load_model(model_type="svm"):
    if model_type == "svm":
        model_file = "svm_model.pkl"
    else:
        model_file = "knn_model.pkl"
    
    if not os.path.exists(model_file):
        st.error(f"❌ {model_file} not found! Train the model first.")
        return None, None, None
    
    model = joblib.load(model_file)
    extractor = FeatureExtractor()
    labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    return model, extractor, labels

# Sidebar: Model selection
st.sidebar.title("⚙️ Model Configuration")
model_type = st.sidebar.radio("Choose Classification Model:", ["SVM", "KNN"], horizontal=True)
model_selected = model_type.lower()

model, extractor, class_names = load_model(model_selected)

color_map = {
    'cardboard': '#8D6E63', 'glass': '#42A5F5', 'metal': '#78909C',
    'paper': '#FFF59D', 'plastic': '#EF5350', 'trash': '#757575'
}

# Helper: classify an image
def classify(image_np):
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    features = extractor.extract_features(image_bgr)

    pred_idx = model.predict([features])[0]
    pred_name = class_names[pred_idx]

    scores = model.decision_function([features])[0]
    probs = np.exp(scores - scores.max())
    probs /= probs.sum()
    return pred_name, probs, pred_idx

# Tabs
tab_upload, tab_cam = st.tabs(["📤 Upload Image", "📷 Webcam"])

# ------------------ Upload Tab ------------------
with tab_upload:
    file = st.file_uploader("Upload an image", ["jpg", "png", "jpeg", "bmp","jfif","webp"])
    if file:
        img = Image.open(file)
        img_np = np.array(img)

        col1, col2 = st.columns(2)
        col1.image(img, use_container_width=True)

        pred_name, probs, pred_idx = classify(img_np)
        confidence = probs[pred_idx] * 100

        color = color_map[pred_name]
        col2.markdown(f"""
            <div style="padding:1rem;background:{color};color:white;font-size:2rem;font-weight:bold;border-radius:10px;">
                {pred_name.upper()} ({confidence:.2f}%)
            </div>
        """, unsafe_allow_html=True)

        st.bar_chart({name: [p * 100] for name, p in zip(class_names, probs)})

# ------------------ Webcam Tab ------------------
with tab_cam:
    st.info("Click the button below to capture an image from your webcam.")
    snapshot = st.camera_input("Take a picture")

    if snapshot:
        img = Image.open(snapshot)
        img_np = np.array(img)

        pred_name, probs, pred_idx = classify(img_np)
        confidence = probs[pred_idx] * 100

        color = color_map[pred_name]
        st.markdown(f"""
            <div style="padding:1rem;background:{color};color:white;font-size:2rem;font-weight:bold;border-radius:10px;">
                {pred_name.upper()} ({confidence:.2f}%)
            </div>
        """, unsafe_allow_html=True)

        st.bar_chart({name: [p * 100] for name, p in zip(class_names, probs)})

        # Save image
        if st.button("💾 Save Image"):
            os.makedirs("captured_images", exist_ok=True)
            filename = f"captured_images/{pred_name}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            img.save(filename)
            st.success(f"Saved: {filename}")

st.markdown("---")