import streamlit as st
import numpy as np
import cv2
import joblib
import os
from skimage.feature import local_binary_pattern

# --- Load model and utilities ---
@st.cache_resource
def load_model():
    model = joblib.load("best_model_lightgbm.pkl")
    le = joblib.load("label_encoder.pkl")
    return model, le

model, le = load_model()

IMG_SIZE = (224, 224)
HSV_BINS = 32
RGB_BINS = 32
LBP_POINTS = 24
LBP_RADIUS = 3
LBP_BINS = LBP_POINTS + 2

def extract_features(img):
    img = cv2.resize(img, IMG_SIZE)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    features = []

    # HSV statistics
    for i in range(3):
        chan = img_hsv[:, :, i].ravel()
        features.extend([np.mean(chan), np.std(chan),
                         np.percentile(chan, 25), np.percentile(chan, 50),
                         np.percentile(chan, 75), np.percentile(chan, 90)])

    # Histograms
    for hist, bins in [(cv2.calcHist([img_hsv], [0], None, [HSV_BINS], [0, 180]), HSV_BINS),
                       (cv2.calcHist([img_hsv], [1], None, [HSV_BINS], [0, 256]), HSV_BINS),
                       (cv2.calcHist([img_hsv], [2], None, [HSV_BINS], [0, 256]), HSV_BINS)]:
        features.extend(cv2.normalize(hist, hist).flatten())

    for chan in range(3):
        hist = cv2.calcHist([img], [chan], None, [RGB_BINS], [0, 256])
        features.extend(cv2.normalize(hist, hist).flatten())

    # LBP
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=LBP_BINS, range=(0, LBP_BINS), density=True)
    features.extend(lbp_hist)

    return np.array(features).reshape(1, -1)

# --- Streamlit UI ---
st.set_page_config(page_title="Fruit Ripeness Detector", layout="centered")
st.title("🍎🍌 Fruit Ripeness Detector")
st.markdown("**99.14% Accurate • Classical ML Only • No Deep Learning Needed**")

uploaded_file = st.file_uploader("Upload a fruit photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_rgb, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.write("### Analyzing...")
        with st.spinner("Extracting features & predicting..."):
            features = extract_features(img)
            pred = model.predict(features)[0]
            prob = model.predict_proba(features)[0]
            confidence = prob[pred] * 100
            label = le.inverse_transform([pred])[0]

            if "ripe" in label.lower():
                st.success(f"**{label.upper()}**")
                st.balloons()
            elif "unripe" in label.lower():
                st.warning(f"**{label.upper()}**")
            else:
                st.error(f"**{label.upper()}**")

            st.metric("Confidence", f"{confidence:.1f}%")

    st.markdown("---")
    st.caption("Built with LightGBM + Hand-crafted Features | Accuracy: 99.14% on test set")