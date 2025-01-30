from django.shortcuts import render

# Create your views here.
# segmentation/views.py

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from .utils import load_model, preprocess_image, run_inference, postprocess_mask
from PIL import Image
import numpy as np
import os

# Load the ONNX model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "UNet_Based_Model.onnx")
model = load_model(MODEL_PATH)
@csrf_exempt
def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        image_file = request.FILES['image']
        
        # Save the uploaded image temporarily
        image_path = default_storage.save('temp_image.png', image_file)
        image = Image.open(image_path)
        original_size = image.size
        # Preprocess the image
        image_array = preprocess_image(image)
        
        # Run the model inference
        mask = run_inference(model, image_array)
        
        # Post-process the mask (resize to original size)
        mask_image = postprocess_mask(mask, original_size)
        
        # Save the mask to a temporary file
        mask_path = 'temp_mask.png'
        mask_image.save(mask_path, 'PNG')
        
        # Read the mask and send it as response
        with open(mask_path, 'rb') as mask_file:
            mask_data = mask_file.read()
        
        # Clean up temporary files
        os.remove(image_path)
        os.remove(mask_path)
        
        # Return the mask image as a downloadable response
        response = JsonResponse({'message': 'Mask generated successfully'})
        response['Content-Disposition'] = 'attachment; filename="mask.png"'
        response['Content-Type'] = 'image/png'
        response.content = mask_data
        return response

    return JsonResponse({'error': 'No image uploaded'}, status=400)
