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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Image Prediction API",
    description="A FastAPI backend that processes PNG images through a machine learning model",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
        "https://bespoke-medovik-0b9d2c.netlify.app",
 
    ],  # Explicit origins for production
    allow_credentials=False,  # Set to False when using wildcard origins
    allow_methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Additional CORS middleware for extra coverage
@app.middleware("http")
async def cors_handler(request: Request, call_next):
    # Get the origin from the request
    origin = request.headers.get("origin")
    
    if request.method == "OPTIONS":
        # Handle preflight requests
        response = JSONResponse(content={"message": "OK"})
    else:
        response = await call_next(request)
    
    # Always add CORS headers
    response.headers["Access-Control-Allow-Origin"] = origin or "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept, Origin, X-Requested-With"
    response.headers["Access-Control-Max-Age"] = "86400"
    response.headers["Access-Control-Expose-Headers"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "false"
    
    # Add Vary header for proper caching
    response.headers["Vary"] = "Origin"
    
    return response

@app.get("/")
async def root():
    return {"message": "ML Image Prediction API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ml-image-api"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(..., description="PNG, JPG, or JPEG image file to process")):
    """
    Process a PNG, JPG, or JPEG image through the skin lesion classification model.
    Automatically converts JPG/JPEG to PNG format for processing.
    
    Args:
        file: PNG, JPG, or JPEG image file to be classified
        
    Returns:
        JSONResponse with prediction results
    """
    try:
        logger.info(f"Received file upload: filename={file.filename}, content_type={file.content_type}")
        
        # Validate file upload
        if not file or not file.filename:
            logger.error("No file uploaded")
            return JSONResponse(
                status_code=422,
                content={"error": "No file uploaded. Please provide a PNG, JPG, or JPEG image file."},
                headers={
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                }
            )
        
        # Read file content
        file_content = await file.read()
        logger.info(f"File content read successfully, size: {len(file_content)} bytes")
        
        # Validate file content is not empty
        if not file_content:
            logger.error("Uploaded file is empty")
            return JSONResponse(
                status_code=422,
                content={"error": "Uploaded file is empty. Please provide a valid image file."},
                headers={
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                }
            )
        
        # Validate and convert image format
        try:
            img = Image.open(io.BytesIO(file_content))
            img.verify()  # Verify the image is valid
            
            # Re-open the image after verify (verify() can corrupt the image object)
            img = Image.open(io.BytesIO(file_content))
            original_format = img.format
            
            # Check if the image format is supported
            if img.format not in ['PNG', 'JPEG']:
                logger.error(f"Unsupported image format: {img.format}")
                return JSONResponse(
                    status_code=422,
                    content={
                        "error": f"Unsupported image format: {img.format}. Only PNG, JPG, and JPEG are supported.",
                        "received_format": img.format,
                        "supported_formats": ["PNG", "JPEG", "JPG"]
                    },
                    headers={
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*"
                    }
                )
            
            # Convert to PNG if it's JPEG/JPG
            if img.format == 'JPEG':
                logger.info(f"Converting {img.format} to PNG for processing")
                
                # Ensure RGB mode (JPEG might be in different modes)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Convert to PNG in memory
                png_buffer = io.BytesIO()
                img.save(png_buffer, format='PNG')
                png_buffer.seek(0)
                
                # Create new PNG image object
                img = Image.open(png_buffer)
                logger.info(f"Successfully converted {original_format} to PNG")
            
            logger.info(f"Valid image validated: format={original_format}, final_format=PNG, size={img.size}")
            
        except Exception as e:
            logger.error(f"Invalid image file: {e}")
            return JSONResponse(
                status_code=422,
                content={"error": "Invalid image file. Please upload a valid PNG, JPG, or JPEG image."},
                headers={
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                }
            )
        
        # Process image through ML model
        try:
            # Use the inference function from ml_model
            logger.info("Starting ML model inference...")
            predictions = SkinLesionClassifier.predict(img)
            
            # Validate predictions format
            if not isinstance(predictions, dict):
                logger.error(f"Invalid predictions format: {type(predictions)}")
                return JSONResponse(
                    status_code=500,
                    content={"error": "ML model returned invalid prediction format"},
                    headers={
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*"
                    }
                )
            
            # Get the top prediction
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
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*"
                    }
                )
            
            logger.info(f"Prediction completed successfully for {file.filename}")
            
            return JSONResponse(
                status_code=200,
                content={
                    "message": "Image processed successfully",
                    "filename": file.filename,
                    "image_info": {
                        "original_format": original_format,
                        "processed_format": "PNG",
                        "size": img.size,
                        "mode": img.mode,
                        "file_size_bytes": len(file_content),
                        "converted": original_format != "PNG"
                    },
                    "predictions": {
                        "top_prediction": {
                            "class": top_class,
                            "confidence": confidence
                        },
                        "all_probabilities": predictions
                    }
                },
                headers={
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type, Authorization"
                }
            )
            
        except Exception as e:
            logger.error(f"Error during ML inference: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Error processing image through ML model: {str(e)}"},
                headers={
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing image: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"},
            headers={
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            }
        )