import os
import time
import joblib
import numpy as np
import streamlit as st
from PIL import Image
from feature_extractor import FeatureExtractor
import cv2

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
    labels = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']

    return model, extractor, labels, scaler

def test_image_streamlit(img_np, model, extractor, scaler, class_names, model_type, threshold=0.61):
    try:
        features = extractor.extract_features(img_np)
        features = np.array(features)
        scaled = scaler.transform(features.reshape(1, -1))
    except Exception as e:
        return None, None, None

    try:
        # Prefer calibrated probabilities when available (matches svm_test.py)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(scaled)[0]
        else:
            # Fallback: derive probabilities from decision function
            if hasattr(model, "decision_function"):
                scores = model.decision_function(scaled)[0]
                # Ensure scores is 1D
                scores = np.array(scores)
                # Softmax over scores as an approximation
                probs = np.exp(scores - scores.max())
                probs = probs / probs.sum()
            else:
                # Last resort: one-hot on predicted class
                pred_idx = int(model.predict(scaled)[0])
                probs = np.zeros(len(class_names), dtype=float)
                probs[pred_idx] = 1.0

        max_prob = probs.max()
        pred_idx = probs.argmax()

        # Apply threshold for unknown detection
        if max_prob < threshold:
            pred_name = "unknown"
        else:
            pred_name = class_names[pred_idx]

    except Exception as e:
        return None, None, None
    
    return pred_name, probs, pred_idx


def decode_image_to_rgb(uploaded_file):
    """Decode an UploadedFile to a 3-channel RGB numpy array using OpenCV, matching svm_test path."""
    try:
        file_bytes = np.frombuffer(uploaded_file.getvalue(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        # Handle channels: BGRA/BGR -> RGB
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # grayscale -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img
    except Exception:
        return None

st.sidebar.title("⚙️ Model Configuration")
model_type = st.sidebar.radio("Choose Classification Model:", ["SVM", "KNN"], horizontal=True)
model_selected = model_type.lower()
unknown_threshold = st.sidebar.slider("Unknown Detection Threshold", 0.0, 1.0, 0.61, 0.01)

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
        img_np = decode_image_to_rgb(file)
        if img_np is None:
            st.error("Could not decode the uploaded image.")
        else:
            col1, col2 = st.columns(2)

            # Show image decoded via OpenCV (aligned with svm_test)
            col1.image(img_np, caption="Uploaded Image", use_container_width=True)

            # Run full test
            pred_name, probs, pred_idx = test_image_streamlit(
                img_np, model, extractor, scaler, class_names, model_selected, unknown_threshold
            )

            if pred_name is not None:
                confidence = probs[pred_idx] * 100 if pred_name != "unknown" else probs.max() * 100

                color = color_map[pred_name]
                col2.markdown(f"""
                    <div style="padding:1rem;background:{color};color:white;
                    font-size:2rem;font-weight:bold;border-radius:10px;">
                        {pred_name.upper()} ({confidence:.2f}%)
                    </div>
                """, unsafe_allow_html=True)

                st.bar_chart({name: [p * 100] for name, p in zip(class_names, probs)})

                with st.expander("Prediction details"):
                    st.write({
                        "threshold": unknown_threshold,
                        "max_prob": float(probs.max()),
                        "pred_idx": int(pred_idx),
                        "pred_name": pred_name,
                    })
                    sorted_probs = sorted(zip(class_names, probs), key=lambda x: x[1], reverse=True)
                    for cname, p in sorted_probs:
                        st.write(f"{cname}: {p*100:.2f}%")


with tab_cam:

    st.info("Click button to capture an image from your webcam.")
    snapshot = st.camera_input("Take a picture")

    if snapshot:
        img_np = decode_image_to_rgb(snapshot)
        if img_np is None:
            st.error("Could not decode the captured image.")
        else:
            # show captured image
            st.image(img_np, use_container_width=True)
           
            pred_name, probs, pred_idx = test_image_streamlit(
                img_np, model, extractor, scaler, class_names, model_selected, unknown_threshold
            )

            if pred_name is not None:
                confidence = probs[pred_idx] * 100 if pred_name != "unknown" else probs.max() * 100

                color = color_map[pred_name]
                st.markdown(f"""
                    <div style="padding:1rem;background:{color};color:white;
                    font-size:2rem;font-weight:bold;border-radius:10px;text-align:center;">
                        {pred_name.upper()} ({confidence:.2f}%)
                    </div>
                """, unsafe_allow_html=True)

                st.bar_chart({name: [p * 100] for name, p in zip(class_names, probs)})

                with st.expander("Prediction details"):
                    st.write({
                        "threshold": unknown_threshold,
                        "max_prob": float(probs.max()),
                        "pred_idx": int(pred_idx),
                        "pred_name": pred_name,
                    })
                    sorted_probs = sorted(zip(class_names, probs), key=lambda x: x[1], reverse=True)
                    for cname, p in sorted_probs:
                        st.write(f"{cname}: {p*100:.2f}%")

        # save captured image
        if st.button("💾 Save Image", key="save_cam") and "img_np" in locals() and img_np is not None:
            os.makedirs("captured_images", exist_ok=True)
            filename = f"captured_images/{pred_name}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            Image.fromarray(img_np).save(filename)
            st.success(f"Saved: {filename}")

st.markdown("---")