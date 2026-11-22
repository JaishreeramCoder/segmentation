from django.core.files.storage import default_storage
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET

from PIL import Image
import os

from .utils import (
    load_model,
    match_reference_histogram,
    preprocess_image,
    run_inference,
    postprocess_mask,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "UNet_Based_Model.onnx")

REFERENCE_IMAGE_PATH = os.path.abspath(os.path.join(
    BASE_DIR,
    "..", "..", "..",
    "predictions",
    "UNet_based_model_predictions",
    "0",
    "original_img.png",
))

model = load_model(MODEL_PATH)

@require_GET
def health_check(request):
    return JsonResponse({"status": "ok"})
    
@csrf_exempt
def upload_image(request):
    if request.method == "POST" and request.FILES.get("image"):
        image_file = request.FILES["image"]

        image_path = default_storage.save("temp_image.png", image_file)
        image_path = default_storage.path(image_path)

        try:
            with Image.open(image_path) as uploaded_image:
                uploaded_image.load()
                original_size = uploaded_image.size

                # Apply histogram matching before model preprocessing
                corrected_image = match_reference_histogram(
                    uploaded_image,
                    REFERENCE_IMAGE_PATH,
                )

                image_array = preprocess_image(corrected_image)

            mask = run_inference(model, image_array)
            mask_image = postprocess_mask(mask, original_size)

            mask_path = "temp_mask.png"
            mask_image.save(mask_path, "PNG")

            with open(mask_path, "rb") as mask_file:
                mask_data = mask_file.read()

            response = JsonResponse({"message": "Mask generated successfully"})
            response["Content-Disposition"] = 'attachment; filename="mask.png"'
            response["Content-Type"] = "image/png"
            response.content = mask_data
            return response

        finally:
            if os.path.exists(image_path):
                os.remove(image_path)

            if os.path.exists("temp_mask.png"):
                os.remove("temp_mask.png")

    return JsonResponse({"error": "No image uploaded"}, status=400)
