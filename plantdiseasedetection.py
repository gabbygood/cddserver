import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io
from PIL import Image
import os

# Clear previous session to avoid conflicts
tf.keras.backend.clear_session()

# Model Path
MODEL_PATH = "plant_disease_model.h5"

# Load Model with Error Handling
@st.cache_resource  # Caching to improve performance
def load_plant_disease_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file '{MODEL_PATH}' not found. Please check the path.")
        return None
    try:
        model = load_model(MODEL_PATH, compile=False)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Load the model
model = load_plant_disease_model()

# Class Names for Disease Prediction
CLASS_NAMES = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']

# Function to preprocess the uploaded image
def preprocess_image(image_data):
    img = Image.open(io.BytesIO(image_data)).convert("RGB")  # Ensure RGB format
    img = img.resize((224, 224))  # Resize to model's expected size
    img_array = np.array(img).astype("float32") / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Function to predict the plant disease
def predict(img_array):
    if model is None:
        return None, None
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions)
    return CLASS_NAMES[class_index], confidence

# Streamlit UI
st.title("🌱 Plant Disease Detection Web App")
st.markdown("Upload an image of a plant leaf to detect possible diseases.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image_data = uploaded_file.read()
    st.image(image_data, caption="🖼️ Uploaded Image", use_column_width=True)

    img_array = preprocess_image(image_data)

    # Only predict if the model was loaded successfully
    if model is not None:
        class_name, confidence = predict(img_array)
        st.subheader(f"✅ Prediction: {class_name}")
        st.write(f"📊 Confidence: {confidence:.2%}")

        # Show additional messages if the plant is diseased
        if "Healthy" not in class_name:
            st.warning("⚠️ The plant may have a disease. Consider taking proper care and treatment.")

    else:
        st.error("❌ Model not loaded correctly. Please check for errors.")
