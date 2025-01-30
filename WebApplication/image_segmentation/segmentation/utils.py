# segmentation/utils.py

import onnxruntime
import numpy as np
from PIL import Image
import io

# Load ONNX model
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "UNet_Based_Model.onnx")
def load_model(model_path=MODEL_PATH):
    return onnxruntime.InferenceSession(model_path)

# Preprocess the image (resize to 128x128 and normalize)
def preprocess_image(image: Image.Image):
    image = image.resize((128, 128))  # Resize to 128x128
    image_array = np.array(image).astype(np.float32) / 255.0  # Convert to numpy array and normalize
    image_array = image_array.transpose(2, 0, 1)  # Change shape to (C, H, W)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Run inference on the image and return the mask
def run_inference(model, image_array):
    input_name = model.get_inputs()[0].name  # Get input name
    output_name = model.get_outputs()[0].name  # Get output name
    mask = model.run([output_name], {input_name: image_array})[0]
    return mask

# Post-process the mask to resize it back to original size and convert to image
def postprocess_mask(mask, original_size):
    mask = np.squeeze(mask)  # Remove single-dimensional entries from the shape
    print(mask)
    if mask.ndim == 3 and mask.shape[0] == 2:
        # Compare the two channels and create a binary mask
        output_mask = np.where(mask[1] > mask[0], 255, 0).astype(np.uint8)
    else:
        raise ValueError("Unsupported mask shape: {}".format(mask.shape))
    print(output_mask)
    mask_resized = Image.fromarray(output_mask)  # Convert to PIL image
    mask_resized = mask_resized.resize(original_size, Image.NEAREST)  # Resize to original image size
    return mask_resized
