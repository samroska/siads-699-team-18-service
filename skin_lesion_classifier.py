import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import logging
from typing import Dict, Union, Optional
import os
import zipfile
import tempfile
import shutil
import glob

logger = logging.getLogger(__name__)

# Module-level variables for static class
_model: Optional[tf.keras.Model] = None
_model_loaded: bool = False
_temp_dir: Optional[str] = None

class SkinLesionClassifier:
    """
    A static class for skin lesion classification using a pre-trained Keras model.
    
    This class provides static methods for preprocessing images and making predictions
    on skin lesion images. The model is loaded once and shared across all calls.
    """
    
    # Class constants
    CLASS_NAMES = ['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK']
    INPUT_SIZE = (64, 64)
    DEFAULT_MODEL_PATH = 'PAD-UFES-20.zip'
    
    @staticmethod
    def _reassemble_split_files(base_path: str) -> str:
        """Reassemble split files if they exist."""
        # Check if split files exist
        split_pattern = f"{base_path}.part*"
        split_files = sorted(glob.glob(split_pattern))
        
        if not split_files:
            return base_path  # No split files found, return original path
        
        logger.info(f"Found {len(split_files)} split files: {split_files}")
        
        # Create temporary file for reassembled content
        temp_fd, temp_path = tempfile.mkstemp(suffix='.zip')
        
        try:
            with os.fdopen(temp_fd, 'wb') as temp_file:
                for split_file in split_files:
                    logger.info(f"Reading split file: {split_file}")
                    with open(split_file, 'rb') as sf:
                        temp_file.write(sf.read())
            
            logger.info(f"Successfully reassembled {len(split_files)} files into {temp_path}")
            return temp_path
            
        except Exception as e:
            # Clean up temp file if something went wrong
            try:
                os.unlink(temp_path)
            except:
                pass
            raise e
    @staticmethod
    def _extract_model_if_zipped(original_model_path: str) -> str:
        """Extract model from zip file if needed, handling split files."""
        global _temp_dir
        
        original_path = original_model_path
        
        # Check if we need to reassemble split files
        if not os.path.exists(original_path):
            # Try to find and reassemble split files
            reassembled_path = SkinLesionClassifier._reassemble_split_files(original_path)
            if reassembled_path != original_path:
                original_path = reassembled_path
        
        # Check if the file is a zip file
        if original_path.endswith('.zip') or original_model_path.endswith('.zip'):
            try:
                logger.info(f"Extracting model from zip: {original_path}")
                
                # Create a temporary directory
                if not _temp_dir:
                    _temp_dir = tempfile.mkdtemp()
                
                # Extract the zip file
                with zipfile.ZipFile(original_path, 'r') as zip_ref:
                    zip_ref.extractall(_temp_dir)
                
                # Find the model file in the extracted content
                for root, dirs, files in os.walk(_temp_dir):
                    for file in files:
                        if file.endswith('.keras') or file.endswith('.h5'):
                            model_path = os.path.join(root, file)
                            logger.info(f"Found model file: {model_path}")
                            return model_path
                
                raise FileNotFoundError("No .keras or .h5 model file found in the zip archive")
                
            except Exception as e:
                logger.error(f"Error extracting model from zip: {e}")
                raise
        
        return original_path
    
    @staticmethod
    def _ensure_model_loaded(model_path: str = None):
        """Ensure the model is loaded. Load it if not already loaded."""
        global _model, _model_loaded
        
        if _model_loaded and _model is not None:
            return
        
        if model_path is None:
            model_path = SkinLesionClassifier.DEFAULT_MODEL_PATH
        
        try:
            # Check if we need to extract from zip
            actual_model_path = SkinLesionClassifier._extract_model_if_zipped(model_path)
            
            if not os.path.exists(actual_model_path):
                raise FileNotFoundError(f"Model file not found: {actual_model_path}")
            
            _model = tf.keras.models.load_model(actual_model_path)
            _model_loaded = True
            logger.info(f"Model loaded successfully from {actual_model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Clean up temp directory if it was created
            SkinLesionClassifier._cleanup_temp_files()
            raise
    
    @staticmethod
    def _cleanup_temp_files():
        """Clean up temporary files if they exist."""
        global _temp_dir
        if _temp_dir and os.path.exists(_temp_dir):
            try:
                shutil.rmtree(_temp_dir)
                logger.info(f"Cleaned up temporary directory: {_temp_dir}")
                _temp_dir = None
            except Exception as e:
                logger.warning(f"Could not clean up temporary directory: {e}")
    
    @staticmethod
    def preprocess_image(image: Union[Image.Image, str]) -> np.ndarray:
        """
        Preprocess an image for model prediction.
        
        Args:
            image: PIL Image object or path to image file
            
        Returns:
            np.ndarray: Preprocessed image array ready for prediction
        """
        try:
            # Handle both PIL Image objects and file paths
            if isinstance(image, str):
                image_rgb = Image.open(image).convert('RGB')
            elif isinstance(image, Image.Image):
                image_rgb = image.convert('RGB')
            else:
                raise ValueError("Image must be a PIL Image object or file path")
            
            # Convert to array and resize
            image_array = img_to_array(image_rgb)
            resized_image = tf.image.resize(image_array, SkinLesionClassifier.INPUT_SIZE)
            
            # Reshape and normalize
            processed_array = img_to_array(resized_image).reshape(1, 64, 64, 3)
            processed_array = processed_array / 255.0
            
            return processed_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    @staticmethod
    def predict(image: Union[Image.Image, str], model_path: str = None) -> Dict[str, float]:
        """
        Make a prediction on a skin lesion image.
        
        Args:
            image: PIL Image object or path to image file
            model_path: Optional path to model file (uses default if not provided)
            
        Returns:
            Dict[str, float]: Dictionary with class names as keys and probabilities as values
        """
        try:
            # Ensure model is loaded
            SkinLesionClassifier._ensure_model_loaded(model_path)
            
            if _model is None:
                raise RuntimeError("Model failed to load.")
            
            # Preprocess the image
            processed_image = SkinLesionClassifier.preprocess_image(image)
            
            # Make prediction
            prediction = _model.predict(processed_image, verbose=0)
            
            # Create results dictionary
            results = {}
            for i, class_name in enumerate(SkinLesionClassifier.CLASS_NAMES):
                results[class_name] = float(round(prediction[0][i], 3))
            
            logger.info(f"Prediction completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    @staticmethod
    def get_top_prediction(image: Union[Image.Image, str], model_path: str = None) -> tuple:
        """
        Get the top prediction class and confidence.
        
        Args:
            image: PIL Image object or path to image file
            model_path: Optional path to model file (uses default if not provided)
            
        Returns:
            tuple: (class_name, confidence) of the top prediction
        """
        predictions = SkinLesionClassifier.predict(image, model_path)
        top_class = max(predictions, key=predictions.get)
        confidence = predictions[top_class]
        
        return top_class, confidence
    
    @staticmethod
    def print_predictions(image: Union[Image.Image, str], model_path: str = None):
        """
        Print predictions in a formatted way (for debugging/testing).
        
        Args:
            image: PIL Image object or path to image file
            model_path: Optional path to model file (uses default if not provided)
        """
        predictions = SkinLesionClassifier.predict(image, model_path)
        
        print('\nProbabilities:')
        for class_name, probability in predictions.items():
            print(f'{class_name}: {probability}')
    
    @staticmethod
    def get_prediction_summary(image: Union[Image.Image, str], model_path: str = None) -> Dict:
        """
        Get a complete prediction summary including top prediction and all probabilities.
        
        Args:
            image: PIL Image object or path to image file
            model_path: Optional path to model file (uses default if not provided)
            
        Returns:
            Dict: Complete prediction summary
        """
        predictions = SkinLesionClassifier.predict(image, model_path)
        top_class, confidence = SkinLesionClassifier.get_top_prediction(image, model_path)
        
        return {
            'top_prediction': {
                'class': top_class,
                'confidence': confidence
            },
            'all_predictions': predictions,
            'model_info': {
                'classes': SkinLesionClassifier.CLASS_NAMES,
                'input_size': SkinLesionClassifier.INPUT_SIZE
            }
        }
    
    @staticmethod
    def cleanup():
        """Manually clean up resources and temporary files."""
        global _model, _model_loaded
        SkinLesionClassifier._cleanup_temp_files()
        _model = None
        _model_loaded = False
        logger.info("Static classifier resources cleaned up")


# For backward compatibility - function interface
def load_model(model_path: str = None):
    """Load the model (for backward compatibility)."""
    SkinLesionClassifier._ensure_model_loaded(model_path)


def inference_function(image: Union[Image.Image, str]) -> Dict[str, float]:
    """
    Legacy inference function that uses the static classifier.
    
    Args:
        image: PIL Image object or path to image file
        
    Returns:
        Dict[str, float]: Prediction results
    """
    return SkinLesionClassifier.predict(image)