import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json

# Load the model
MODEL_PATH = "plant_disease_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels
CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___healthy", "Cherry_(including_sour)___Powdery_mildew",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy", "Corn_(maize)___Northern_Leaf_Blight", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___healthy", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___healthy", "Potato___Late_blight", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___healthy", "Strawberry___Leaf_scorch",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___healthy", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_mosaic_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]

# Function to preprocess image
def preprocess_image(image):
    img = image.convert("RGB").resize((224, 224))
    img_array = np.array(img).astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)

# Streamlit UI
st.title("🌿 Plant Disease Detection")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict when button is clicked
    if st.button("Predict"):
        img_array = preprocess_image(image)
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions)
        confidence = float(np.max(predictions))

        # Display result
        st.success(f"Prediction: {CLASS_NAMES[class_index]}")
        st.write(f"Confidence: {confidence:.2%}")

# Handle API Request (React Native will use this)
params = st.experimental_get_query_params()
if "api" in params:
    if uploaded_file is not None:
        response = {
            "prediction": CLASS_NAMES[class_index],
            "confidence": confidence
        }
        st.json(response)
