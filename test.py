import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io

# **Set page config FIRST**
st.set_page_config(page_title="🌿 Plant Disease Detector", layout="centered")

# Load Model
MODEL_PATH = "plant_disease_model.h5"

@st.cache_resource
def load_trained_model():
    """Load and return the trained plant disease model."""
    try:
        model = load_model(MODEL_PATH, custom_objects={'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D})
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_trained_model()

# Class Labels
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]

# Preprocess Image
def preprocess_image(image):
    """Convert image to RGB, resize, normalize, and prepare for prediction."""
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit UI
st.title("🌿 Plant Disease Detection System")
st.markdown("Upload a plant leaf image to detect possible diseases.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict button
    if st.button("🔍 Detect Disease"):
        st.info("Processing image...")

        if model:
            try:
                # Preprocess and predict
                img_array = preprocess_image(image)
                predictions = model.predict(img_array)
                class_index = np.argmax(predictions)
                confidence = float(np.max(predictions))

                # Display results
                st.success(f"✅ Prediction: {CLASS_NAMES[class_index]}")
                st.write(f"📊 Confidence: {confidence * 100:.2f}%")
            
            except Exception as e:
                st.error(f"⚠️ Error during prediction: {str(e)}")
        else:
            st.error("⚠️ Model is not loaded properly.")

st.markdown("---")
st.markdown("🔗 Local Model is being used for predictions.")
