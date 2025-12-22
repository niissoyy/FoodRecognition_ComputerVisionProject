import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
from skimage.feature import hog, local_binary_pattern

model = joblib.load("food_prediction.pkl")

calorie_map = {
    "ayam_goreng": 260,
    "ayam_pop": 210,
    "daging_rendang": 468,
    "dendeng_batokok": 320,
    "gulai_tunjang": 380,
    "telur_balado": 220,
    "telur_dadar": 190
}

IMG_SIZE = (128, 128)

def extract_feature(img_rgb):
    img = cv2.resize(img_rgb, IMG_SIZE)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    hog_feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(32, 32),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    color_hist = cv2.calcHist(
        [img_bgr],
        [0, 1, 2],
        None,
        [8, 8, 8],
        [0, 256, 0, 256, 0, 256]
    )
    cv2.normalize(color_hist, color_hist)
    color_hist = color_hist.flatten()

    radius = 2
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2)
    )
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    features = np.hstack([
        hog_feat * 0.45,
        color_hist * 0.30,
        lbp_hist * 0.25
    ])

    return features

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

.stApp {
    background-color: #F7EDE2;
}

.stApp, .stApp * {
    font-family: 'Poppins', sans-serif !important;
}

.title {
    font-size: 46px;
    font-weight: 700;
    color: #9E2A2B;
}

.subtitle {
    font-size: 18px;
    color: #C94C4C;
    margin-bottom: 30px;
}

.result-card {
    background-color: #FFF8F1;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    margin-top: 25px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;">
    <div class="title">Padang Food Detection</div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload gambar makanan Padang",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=350)

    img_np = np.array(image)
    features = extract_feature(img_np)

    pred = model.predict([features])[0]
    prob = np.max(model.predict_proba([features]))

    calorie = calorie_map.get(pred, "Tidak tersedia")

    st.markdown(f"""
    <div class="result-card">
        <h3>🍽️ Hasil Prediksi</h3>
        <p><b>Makanan:</b> {pred.replace("_", " ").title()}</p>
        <p><b>Confidence:</b> {prob:.2f}</p>
        <p><b>Estimasi Kalori:</b> {calorie} kcal / porsi</p>
    </div>
    """, unsafe_allow_html=True)
