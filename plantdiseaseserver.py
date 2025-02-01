from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# Initialize FastAPI app
app = FastAPI()

# Load Model
MODEL_PATH = "plant_disease_model.h5"

try:
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Class Names
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
def preprocess_image(image_data):
    img = Image.open(io.BytesIO(image_data)).convert("RGB")
    img = img.resize((224, 224))  # Resize to model input size
    img_array = np.array(img).astype("float32") / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)

# Prediction Endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded properly."})

    image_data = await file.read()
    img_array = preprocess_image(image_data)

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = float(np.max(predictions))

    response = {
        "prediction": CLASS_NAMES[class_index],
        "confidence": confidence
    }
    
    return JSONResponse(content=response)

# Root Endpoint with H1 HTML Response
@app.get("/")
async def root():
    html_content = """
    <html>
        <head>
            <title>API Status</title>
        </head>
        <body style="text-align: center; padding: 50px;">
            <h1 style="color: green;">✅ API is running!</h1>
            <p>Use the <code>/predict</code> endpoint to classify plant diseases.</p>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)
