import cv2
import numpy as np
from PIL import Image

class InputAgent:
    """
    Handles preprocessing and standardizing inputs from both 
    image uploads and webcam frames.
    """
    def __init__(self):
        pass

    def process_image(self, image):
        """
        Takes an input image (which could be a numpy array from Gradio webcam
        or a PIL Image from file upload) and ensures it's returned as an RGB RGB numpy array.
        """
        if isinstance(image, Image.Image):
            # Convert PIL image to numpy array
            image = np.array(image.convert('RGB'))
        elif isinstance(image, np.ndarray):
            # Gradio often passes webcam frames as numpy arrays (RGB)
            pass
        else:
            raise ValueError("Unsupported image format. Must be PIL Image or numpy array.")
            
        return image
