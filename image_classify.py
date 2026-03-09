import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the trained CIFAR-10 model
model = tf.keras.models.load_model("model.keras")

# CIFAR-10 class labels
CLASS_NAMES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# Function to preprocess the image
def preprocess_image(image):
    image = np.array(image)  # Convert PIL image to NumPy array
    image = cv2.resize(image, (32, 32))  # Resize to 32x32 (CIFAR-10 size)
    image = image.astype("float32") / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("CIFAR-10 Image Classifier")
st.write("Upload an image, and the model will predict its category.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)  # Get highest probability class
    confidence = np.max(prediction) * 100  # Confidence score

    # Display prediction results
    st.write(f"### Prediction: {CLASS_NAMES[predicted_class]}")
    st.write(f"**Confidence: {confidence:.2f}%**")

    # Show confidence scores for all classes
    st.subheader("Prediction Probabilities:")
    for i, class_name in enumerate(CLASS_NAMES):
        st.write(f"{class_name}: {prediction[0][i]*100:.2f}%")