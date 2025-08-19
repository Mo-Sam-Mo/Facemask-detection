import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load pretrained face mask detection model
# You need a trained model file "mask_detector.model"
@st.cache_resource
def load_mask_model():
    return load_model("myModel.keras")

model = load_mask_model()

st.title("😷 Face Mask Detection App")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(rgb, caption="Uploaded Image", use_column_width=True)

    # Preprocess image (assuming MobileNetV2 input size 224x224)
    face = cv2.resize(rgb, (128, 128))
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0) / 255.0

    # Prediction
    pred = model.predict(face)[0]

    label = "Mask" if pred > 0.5 else "No Mask"


    st.subheader(f"Prediction: {label}")
