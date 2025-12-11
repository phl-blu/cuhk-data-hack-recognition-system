import os
import time
import joblib
import numpy as np
import streamlit as st
from PIL import Image
from feature_extractor import FeatureExtractor

st.set_page_config(page_title="Waste Classification", page_icon="♻️", layout="wide")
st.markdown("<h1 style='text-align:center;color:#2E7D32'>♻️ Material Stream Identification</h1>", unsafe_allow_html=True)
st.markdown("---")

@st.cache_resource
def load_model(model_type="svm"):
    if model_type == "svm":
        model_file = "svm_model.pkl"
    else:
        model_file = "knn_model.pkl"

    if not os.path.exists(model_file):
        st.error(f"❌ {model_file} not found! Train the model first.")
        return None, None, None, None

    scaler_file = "feature_scaler.pkl"
    if not os.path.exists(scaler_file):
        st.error(f"❌ {scaler_file} not found!")
        return None, None, None, None

    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    extractor = FeatureExtractor()
    labels = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash', 'unknown']

    return model, extractor, labels, scaler

def test_image_streamlit(img_np, model, extractor, scaler, class_names, model_type):
    try:
        features = extractor.extract_features(img_np)
        features = np.array(features)
        scaled = scaler.transform(features.reshape(1, -1))
    except Exception as e:
        return None, None, None

    try:
        pred_idx = model.predict(scaled)[0]
        pred_name = class_names[pred_idx]
    except Exception as e:
        return None, None, None

    try:
        if model_type == "svm":
            scores = model.decision_function(scaled)[0]
            probs = np.exp(scores - scores.max())
            probs /= probs.sum()
        else:  # KNN uses predict_proba
            probs = model.predict_proba(scaled)[0]

    except Exception as e:
        return None, None, None
    return pred_name, probs, pred_idx

st.sidebar.title("⚙️ Model Configuration")
model_type = st.sidebar.radio("Choose Classification Model:", ["SVM", "KNN"], horizontal=True)
model_selected = model_type.lower()

model, extractor, class_names, scaler = load_model(model_selected)

color_map = {
    'glass': '#4CAF50',
    'paper': '#2196F3',
    'cardboard': '#FF9800',
    'plastic': '#9C27B0',
    'metal': '#F44336',
    'trash': '#607D8B',
    'unknown': '#795548'
}

tab_upload, tab_cam = st.tabs(["📤 Upload Image", "📷 Webcam"])

with tab_upload:
    file = st.file_uploader("Upload an image", ["jpg", "png", "jpeg", "bmp", "jfif", "webp"])

    if file:
        img = Image.open(file)
        img_np = np.array(img)

        col1, col2 = st.columns(2)

        # Show image
        col1.image(img, caption="Uploaded Image", use_container_width=True)

        # Run full test
        pred_name, probs, pred_idx = test_image_streamlit(
            img_np, model, extractor, scaler, class_names, model_selected
        )

        if pred_name is not None:
            confidence = probs[pred_idx] * 100

            color = color_map[pred_name]
            col2.markdown(f"""
                <div style="padding:1rem;background:{color};color:white;
                font-size:2rem;font-weight:bold;border-radius:10px;">
                    {pred_name.upper()} ({confidence:.2f}%)
                </div>
            """, unsafe_allow_html=True)

            st.bar_chart({name: [p * 100] for name, p in zip(class_names, probs)})


with tab_cam:

    st.info("Click button to capture an image from your webcam.")
    snapshot = st.camera_input("Take a picture")

    if snapshot:
        img = Image.open(snapshot)
        img_np = np.array(img)

        # show captured image
        st.image(img, use_container_width=True)
       
        pred_name, probs, pred_idx = test_image_streamlit(
            img_np, model, extractor, scaler, class_names, model_selected
        )

        if pred_name is not None:
            confidence = probs[pred_idx] * 100

            color = color_map[pred_name]
            st.markdown(f"""
                <div style="padding:1rem;background:{color};color:white;
                font-size:2rem;font-weight:bold;border-radius:10px;text-align:center;">
                    {pred_name.upper()} ({confidence:.2f}%)
                </div>
            """, unsafe_allow_html=True)

            st.bar_chart({name: [p * 100] for name, p in zip(class_names, probs)})

        # save captured image
        if st.button("💾 Save Image", key="save_cam"):
            os.makedirs("captured_images", exist_ok=True)
            filename = f"captured_images/{pred_name}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            img.save(filename)
            st.success(f"Saved: {filename}")

st.markdown("---")