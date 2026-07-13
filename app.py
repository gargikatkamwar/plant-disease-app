import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.keras")

model = load_model()

with open("class_indices.json") as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}

st.title("Plant Disease Detection")
st.write("Upload a leaf image to detect disease")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_resized = image.resize((64, 64))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = idx_to_class[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: **{predicted_class}**")
    st.write(f"Confidence: {confidence:.2f}%")
