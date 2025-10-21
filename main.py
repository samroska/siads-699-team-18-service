import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
from typing import Optional
import logging
import skin_lesion_classifier as Processor
from skin_lesion_classifier import SkinLesionClassifier
from image_converter import ImageConverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Image Prediction API",
    description="A FastAPI backend that processes PNG, JPG, JPEG, HEIC, HEIF, and MPO images through a machine learning model",
    version="1.0.0"
)

# Initialize ImageConverter
image_converter = ImageConverter()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://capstoneteam18.netlify.app", "https://capstoneteam18.netlify.com", "https://team-18-test.netlify.app"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173"],
#     allow_credentials=True,
#     allow_methods=["GET", "POST", "OPTIONS"],
#     allow_headers=["*"],
# )

@app.get("/")
async def root():
    return {"message": "ML Image Prediction API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ml-image-api"}

async def process_image_with_model(file: UploadFile, endpoint_name: str):
    """
    Helper function to process images with the default model.
    Args:
        file: Uploaded file
        endpoint_name: Name of the endpoint for logging purposes
    """
    try:
        logger.info(f"Received file upload on {endpoint_name}: filename={file.filename}, content_type={file.content_type}")
        if not file or not file.filename:
            logger.error("No file uploaded")
            return JSONResponse(
                status_code=422,
                content={"error": "No file uploaded. Please provide a PNG, JPG, JPEG, HEIC, HEIF, or MPO image file."},
                headers={
                    "Content-Type": "application/json"
                }
            )
        file_content = await file.read()
        logger.info(f"File content read successfully, size: {len(file_content)} bytes")
        if not file_content:
            logger.error("Uploaded file is empty")
            return JSONResponse(
                status_code=422,
                content={"error": "Uploaded file is empty. Please provide a valid image file."},
                headers={
                    "Content-Type": "application/json"
                }
            )
        # Validate and convert image using ImageConverter
        try:
            img, original_format, was_converted = image_converter.process_image(file_content)
            logger.info(f"Successfully processed image: original_format={original_format}, final_format=PNG, size={img.size}, mode={img.mode}, converted={was_converted}")
        except Exception as e:
            logger.error(f"Image validation/conversion failed: {e}")
            supported_formats_msg = image_converter.get_supported_formats_message()
            return JSONResponse(
                status_code=422,
                content={
                    "error": f"Invalid or unsupported image file: {str(e)}", 
                    "supported_formats": supported_formats_msg
                },
                headers={
                    "Content-Type": "application/json"
                }
            )

        try:
            logger.info(f"Starting ML model inference...")
            predictions = SkinLesionClassifier.predict(img)
            logger.info(f"ML model inference completed. Predictions: {predictions}")
            if not isinstance(predictions, dict):
                logger.error(f"Invalid predictions format: {type(predictions)}")
                return JSONResponse(
                    status_code=500,
                    content={"error": "ML model returned invalid prediction format"},
                    headers={
                        "Content-Type": "application/json"
                    }
                )
            try:
                top_class = max(predictions, key=predictions.get)
                confidence = predictions[top_class]
                logger.info(f"Top prediction: {top_class} with confidence: {confidence}")
            except Exception as e:
                logger.error(f"Error processing predictions: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"error": "Error processing model predictions"},
                    headers={
                        "Content-Type": "application/json"
                    }
                )
            logger.info(f"Prediction completed successfully for {file.filename}")
            return JSONResponse(
                status_code=200,
                content={
                    "message": "Image processed successfully",
                    "filename": file.filename,
                    "endpoint": endpoint_name,
                    "image_info": image_converter.get_image_info(img, original_format, len(file_content), was_converted),
                    "predictions": {
                        "top_prediction": {
                            "class": top_class,
                            "confidence": confidence
                        },
                        "all_probabilities": predictions
                    }
                },
                headers={
                    "Content-Type": "application/json"
                }
            )
        except Exception as e:
            logger.error(f"Error during ML inference: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Error processing image through ML model: {str(e)}"},
                headers={
                    "Content-Type": "application/json"
                }
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing image on {endpoint_name}: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"},
            headers={
                "Content-Type": "application/json"
            }
        )

@app.post("/predict")
async def predict_image(file: UploadFile = File(..., description="PNG, JPG, JPEG, HEIC, HEIF, or MPO image file to process")):
    """
    Process image using the default ML model.
    """
    return await process_image_with_model(file, '/predict')

@app.post("/user")
async def predict_image_user(file: UploadFile = File(..., description="PNG, JPG, JPEG, HEIC, HEIF, or MPO image file for user model prediction")):
    """
    Process image using the default ML model (user endpoint).
    """
    return await process_image_with_model(file, '/user')