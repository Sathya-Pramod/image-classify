import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# CIFAR-10 class labels
CLASS_NAMES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# Cache model so it loads only once
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.keras")
    return model

model = load_model()

# Image preprocessing
def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (32, 32))
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image


# Streamlit UI
st.title("CIFAR-10 Image Classifier")
st.write("Upload an image and the model will predict its category.")

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)

    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: **{CLASS_NAMES[predicted_class]}**")
    st.write(f"Confidence: **{confidence:.2f}%**")

    st.subheader("Prediction Probabilities")

    for i, name in enumerate(CLASS_NAMES):
        st.progress(float(prediction[0][i]))
        st.write(f"{name}: {prediction[0][i]*100:.2f}%")
