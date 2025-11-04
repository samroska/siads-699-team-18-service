import io
import logging
from PIL import Image
from typing import Tuple, Union

logger = logging.getLogger(__name__)

class ImageConverter:
 
    # Supported image formats
    SUPPORTED_FORMATS = ['PNG', 'JPEG', 'MPO']
    
    def __init__(self):
 
        self.heic_supported = self._setup_heic_support()
        if self.heic_supported:
            self.SUPPORTED_FORMATS.extend(['HEIF', 'HEIC'])
    
    def _setup_heic_support(self) -> bool:
 
        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
            logger.info("HEIC/HEIF support enabled for iPhone images")
            return True
        except ImportError:
            logger.warning("pillow-heif not available. HEIC/HEIF images from iPhones will not be supported.")
            return False
    
    def validate_format(self, img: Image.Image) -> bool:
 
        return img.format in self.SUPPORTED_FORMATS
    
    def get_supported_formats_message(self) -> str:
   
        base_formats = "PNG, JPEG"
        if self.heic_supported:
            return f"{base_formats}, HEIC/HEIF (iPhone photos)"
        return base_formats
    
    def convert_mpo_to_png(self, img: Image.Image) -> Image.Image:
   
        try:
 
            if hasattr(img, 'n_frames') and img.n_frames > 1:
                # MPO files can have multiple frames, get the first one
                img.seek(0)  # Go to first frame
                logger.info(f"MPO file detected with {img.n_frames} frames, using first frame")            
 
            img = self._ensure_rgb_mode(img)
            
            png_buffer = io.BytesIO()
            img.save(png_buffer, format='PNG')
            png_buffer.seek(0)
 
            return Image.open(png_buffer)
            
        except Exception as e:
            logger.error(f"Error converting MPO to PNG: {e}")
            raise
    
    def _ensure_rgb_mode(self, img: Image.Image) -> Image.Image:
   
        if img.mode == 'RGB':
            return img
        
        if img.mode in ['RGBA', 'LA', 'P']:
 
            background = Image.new('RGB', img.size, (255, 255, 255))
            
            if img.mode == 'P':
                img = img.convert('RGBA')
            
            if img.mode in ['RGBA', 'LA']:
                mask = img.split()[-1] if img.mode == 'RGBA' else None
                background.paste(img, mask=mask)
                return background
            
        return img.convert('RGB')
    
    def convert_to_png(self, img: Image.Image, original_format: str) -> Tuple[Image.Image, bool]:
        """
        Convert any image format to PNG format.
        Always converts to PNG to ensure consistent processing.
        
        Args:
            img: PIL Image object
            original_format: Original format of the image
            
        Returns:
            Tuple of (converted PNG image, was_converted boolean)
        """
        logger.info(f"Converting {original_format} to PNG for consistent processing")
        
        try:
            if img.format == 'MPO':
                converted_img = self.convert_mpo_to_png(img)
            else:
                # Handle all formats (PNG, JPEG, HEIC, etc.) - convert everything to PNG
                rgb_img = self._ensure_rgb_mode(img)
                
                # Convert to PNG in memory
                png_buffer = io.BytesIO()
                rgb_img.save(png_buffer, format='PNG')
                png_buffer.seek(0)
                converted_img = Image.open(png_buffer)
            
            # Track if conversion was needed (original format != PNG)
            was_converted = original_format != 'PNG'
            logger.info(f"Successfully processed {original_format} to PNG (conversion needed: {was_converted})")
            return converted_img, was_converted
            
        except Exception as e:
            logger.error(f"Error converting {original_format} to PNG: {e}")
            raise
    
    def process_image(self, file_content: bytes) -> Tuple[Image.Image, str, bool]:
 
        try:
            # Open and verify image
            img = Image.open(io.BytesIO(file_content))
            img.verify()  # Verify the image is valid
            
            # Re-open after verify (verify can close the image)
            img = Image.open(io.BytesIO(file_content))
            original_format = img.format
            
            logger.info(f"Detected image format: {original_format}")
            
            # Validate format is supported
            if not self.validate_format(img):
                error_message = f"Unsupported image format: {original_format}. Supported formats: {self.get_supported_formats_message()}"
                logger.error(error_message)
                raise ValueError(error_message)
            
            # Convert to PNG if needed
            processed_img, was_converted = self.convert_to_png(img, original_format)
            
            logger.info(f"Image processed successfully: format={original_format}, final_format=PNG, size={processed_img.size}, mode={processed_img.mode}")
            
            return processed_img, original_format, was_converted
            
        except ValueError:
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise Exception(f"Invalid image file: {str(e)}")
    
    def get_image_info(self, img: Image.Image, original_format: str, file_size: int, was_converted: bool) -> dict:
        """
        Get comprehensive information about the processed image.
        All images are now processed to PNG format for consistency.
        """
        return {
            "original_format": original_format,
            "processed_format": "PNG",  # Always PNG now
            "size": img.size,
            "mode": img.mode,
            "file_size_bytes": file_size,
            "converted": was_converted,
            "source_type": {
                "MPO": "iPhone MPO",
                "HEIC": "iPhone HEIC", 
                "HEIF": "iPhone HEIF"
            }.get(original_format, original_format),
            "processing_note": "All images are standardized to PNG format for consistent ML processing"
        }