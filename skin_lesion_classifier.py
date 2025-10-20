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

# Dictionary to store multiple models and their loaded state
_models: Dict[str, Optional[tf.keras.Model]] = {}
_models_loaded: Dict[str, bool] = {}
_temp_dirs: Dict[str, Optional[str]] = {}

class SkinLesionClassifier:

    CLASS_NAMES = ['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK']
    INPUT_SIZE = (224, 224)
    DEFAULT_MODEL_ZIP = 'PAD-UFES-20.keras.zip'
    
    @staticmethod
    def _extract_model_from_zip(zip_path: str) -> str:
        """Extract model from PAD-UFES-20.keras.zip and return the .keras file path."""
        global _temp_dirs
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Model zip file not found: {zip_path}")

        if 'default' not in _temp_dirs or not _temp_dirs['default']:
            _temp_dirs['default'] = tempfile.mkdtemp(suffix='_model')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(_temp_dirs['default'])

        for root, dirs, files in os.walk(_temp_dirs['default']):
            for file in files:
                if file.endswith('.keras'):
                    model_path = os.path.join(root, file)
                    logger.info(f"Found model file: {model_path}")
                    return model_path
        raise FileNotFoundError("No .keras model file found in the zip archive")
    @staticmethod
    def _ensure_model_loaded():
        """Ensure the model is loaded from PAD-UFES-20.keras.zip."""
        global _models, _models_loaded
        if 'default' in _models_loaded and _models_loaded['default'] and _models.get('default') is not None:
            return
        zip_path = SkinLesionClassifier.DEFAULT_MODEL_ZIP
        try:
            model_path = SkinLesionClassifier._extract_model_from_zip(zip_path)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            _models['default'] = tf.keras.models.load_model(model_path)
            _models_loaded['default'] = True
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            SkinLesionClassifier._cleanup_temp_files()
            raise
    
    @staticmethod
    
    @staticmethod
    def _cleanup_temp_files():
        global _temp_dirs
        for temp_dir in _temp_dirs.values():
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    logger.info(f"Cleaned up temporary directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Could not clean up temporary directory: {e}")
        _temp_dirs.clear()
    
    @staticmethod
    def preprocess_image(image: Union[Image.Image, str]) -> np.ndarray:
 
        try:
            if isinstance(image, str):
                image_rgb = Image.open(image).convert('RGB')
            elif isinstance(image, Image.Image):
                image_rgb = image.convert('RGB')
            else:
                raise ValueError("Image must be a PIL Image object or file path")

            image_array = img_to_array(image_rgb)
            resized_image = tf.image.resize(image_array, SkinLesionClassifier.INPUT_SIZE)

            processed_array = img_to_array(resized_image).reshape(1, 224, 224, 3)
            processed_array = processed_array / 255.0

            return processed_array
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
            
    @staticmethod
    def predict(image: Union[Image.Image, str]) -> Dict[str, float]:
        """
        Make prediction using the default model.
        """
        try:
            SkinLesionClassifier._ensure_model_loaded()
            if 'default' not in _models or _models['default'] is None:
                raise RuntimeError("Model failed to load.")
            processed_image = SkinLesionClassifier.preprocess_image(image)
            prediction = _models['default'].predict(processed_image, verbose=0)
            results = {}
            for i, class_name in enumerate(SkinLesionClassifier.CLASS_NAMES):
                results[class_name] = float(round(prediction[0][i], 3))
            logger.info(f"Prediction completed: {results}")
            return results
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    @staticmethod
    def get_top_prediction(image: Union[Image.Image, str]) -> tuple:
        predictions = SkinLesionClassifier.predict(image)
        top_class = max(predictions, key=predictions.get)
        confidence = predictions[top_class]
        return top_class, confidence
    
    @staticmethod
    def print_predictions(image: Union[Image.Image, str]):
        predictions = SkinLesionClassifier.predict(image)
        print('\nProbabilities:')
        for class_name, probability in predictions.items():
            print(f'{class_name}: {probability}')
    
    @staticmethod
    def get_prediction_summary(image: Union[Image.Image, str]) -> Dict:
        predictions = SkinLesionClassifier.predict(image)
        top_class, confidence = SkinLesionClassifier.get_top_prediction(image)
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
        global _models, _models_loaded
        SkinLesionClassifier._cleanup_temp_files()
        _models.clear()
        _models_loaded.clear()
        logger.info("Static classifier resources cleaned up")

# Backward compatibility functions
def load_model():
    """Load the model (for backward compatibility)."""
    SkinLesionClassifier._ensure_model_loaded()

def inference_function(image: Union[Image.Image, str]) -> Dict[str, float]:
    """Legacy inference function that uses the static classifier."""
    return SkinLesionClassifier.predict(image)