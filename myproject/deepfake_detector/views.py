# deepfake_detector/views.py

from django.shortcuts import render
from django.http import HttpResponse
from .forms import ImageUploadForm
from .model_loader import model
import numpy as np
from PIL import Image
from django.middleware.csrf import get_token
import io
import cv2

def preprocess_image(image_file, img_size=128):
    """
    Preprocesses the uploaded image file and prepares it for model prediction.
    
    Args:
        image_file: A file-like object of the uploaded image.
        img_size: Desired image size for the model.
        
    Returns:
        Preprocessed image as a numpy array.
    """
    # Convert image file to OpenCV format
    img = np.array(Image.open(io.BytesIO(image_file.read())))
    
    # Resize the image to the target size
    img = cv2.resize(img, (img_size, img_size))
    
    # Convert image to RGB (OpenCV loads in BGR format)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize the image
    img = img / 255.0
    
    # Add batch dimension (1, IMG_SIZE, IMG_SIZE, 3)
    img = np.expand_dims(img, axis=0)
    
    return img

def predict_image(image_file):
    """
    Makes a prediction using the preprocessed image.
    
    Args:
        image_file: A file-like object of the uploaded image.
        
    Returns:
        Prediction result from the model.
    """
    # Preprocess the image
    image = preprocess_image(image_file)
    
    # Make prediction
    prediction = model.predict(image)
    return prediction

def upload_image(request):
    """
    Handles image upload and prediction.
    
    Args:
        request: The Django request object.
        
    Returns:
        HttpResponse with the result of the prediction or the upload form.
    """
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = request.FILES['image']
            prediction = predict_image(image)
            result = 'Deepfake' if prediction[0] > 0.01 else 'Real'
            return HttpResponse(f'The image is: {result} with prediction of {prediction[0]}')
    else:
        form = ImageUploadForm()
    
    csrf_token = get_token(request)

    # HTML form for image upload
    html = f"""
    <html>
        <head>
            <title>Upload Image</title>
        </head>
        <body>
            <h1>Upload Image</h1>
            <form method="post" enctype="multipart/form-data">
                <input type="hidden" name="csrfmiddlewaretoken" value="{csrf_token}" />
                <label for="image">Select image:</label>
                <input type="file" name="image" id="image" required />
                <input type="submit" value="Upload" />
            </form>
        </body>
    </html>
    """
    return HttpResponse(html)
