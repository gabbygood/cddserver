from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# Initialize FastAPI app
app = FastAPI()

# Load Model
MODEL_PATH = "plant_classification.h5"

try:
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Class Names
CLASS_NAMES = ['aloevera', 'banana', 'bilimbi', 'cantaloupe', 'cassava', 'coconut', 'corn', 'cucumber', 'curcuma', 'eggplant', 'galangal', 'ginger', 'guava', 'kale', 'longbeans', 'mango', 'melon', 'orange', 'paddy', 'papaya', 'peper chili', 'pineapple', 'pomelo', 'shallot', 'soybeans', 'spinach', 'sweet potatoes', 'tobacco', 'waterapple', 'watermelon']


def preprocess_image(image_data):
    img = Image.open(io.BytesIO(image_data)).convert("RGB")
    img = img.resize((224, 224))  
    img_array = np.array(img).astype("float32") / 255.0 
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

# Root Endpoint
@app.get("/")
async def root():
    return {"message": "Plant Disease Detection API is running!"}
