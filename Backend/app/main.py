# AgroscanAI/backend/app/main.py

from fastapi import FastAPI, UploadFile, File # <--- NEW: Import UploadFile and File
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any # <--- NEW: Import Dict, Any for type hints
import io # <--- NEW: For handling image bytes
from PIL import Image # <--- NEW: For basic image processing 

app = FastAPI(
    title="Agroscan AI Backend",
    description="API for detecting tea plant diseases using AI/ML.",
    version="0.1.0",
)

# --- CORS (Cross-Origin Resource Sharing) Configuration ---
origins = [
    "http://localhost:3000",  # Default React dev server port
    "http://localhost:5173",  # Default Vite dev server port
    # Add your Netlify deployed frontend URL here later when you deploy
    # "https://your-netlify-app.netlify.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    """
    Root endpoint for the Agroscan AI API.
    """
    return {"message": "Welcome to Agroscan AI Backend! Go to /docs for API documentation."}

@app.get("/health")
async def health_check():
    """
    Health check endpoint to ensure the API is running.
    """
    return {"status": "ok", "message": "API is healthy!"}


# --- ML Model Placeholder Function ---
# This function will simulate your ML model's prediction.
# Later, you will replace this with actual model loading and inference.
async def predict_disease_placeholder(image_bytes: bytes) -> Dict[str, Any]:
    """
    Placeholder function to simulate disease prediction from an image.
    In a real scenario, this would load an ML model and predict.
    """
    try:
        # Optional: Basic check to see if it's a valid image (using Pillow)
        image = Image.open(io.BytesIO(image_bytes))
        image_format = image.format
        image_size = image.size
        # You could also resize, convert to numpy array here for a real model

        # Simulate a prediction based on some logic (e.g., file size, or just a dummy)
        # For demonstration, we'll just return a fixed dummy prediction
        if image_size[0] > 500: # Just an example condition
            predicted_disease = "Tea Blight (Simulated)"
            confidence = 0.85
            suggestions = "Apply fungicide and improve air circulation."
        else:
            predicted_disease = "Healthy Tea Leaf (Simulated)"
            confidence = 0.99
            suggestions = "Continue good agricultural practices."

        return {
            "success": True,
            "prediction": predicted_disease,
            "confidence": confidence,
            "suggestions": suggestions,
            "debug_info": {
                "image_format": image_format,
                "image_size": f"{image_size[0]}x{image_size[1]}",
                "received_bytes": len(image_bytes)
            }
        }
    except Image.UnidentifiedImageError:
        return {
            "success": False,
            "error": "Invalid image file. Could not identify image format.",
            "debug_info": {"received_bytes": len(image_bytes)}
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"An unexpected error occurred during prediction placeholder: {str(e)}",
            "debug_info": {"received_bytes": len(image_bytes)}
        }


# --- Image Upload and Prediction Endpoint ---
@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    """
    Receives an image file, passes it to the ML model placeholder for prediction,
    and returns the prediction result.
    """
    try:
        # Read the uploaded file's content as bytes
        image_bytes = await file.read()

        # Pass the bytes to our placeholder ML model function
        prediction_result = await predict_disease_placeholder(image_bytes)

        # Return the prediction result
        return prediction_result

    except Exception as e:
        # Handle potential errors during file processing or prediction
        return {
            "success": False,
            "error": f"Failed to process image or make prediction: {str(e)}"
        }