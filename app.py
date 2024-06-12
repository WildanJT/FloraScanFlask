from flask import Flask, request, jsonify
from waitress import serve
import socket

import numpy as np
import requests
from PIL import Image
import io

import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# URL of the model file
model_url = "https://storage.googleapis.com/florascan-model-bucket/keras-model/model.h5"

# Download the model file
response = requests.get(model_url)
with open("model.h5", "wb") as f:
    f.write(response.content)

# Load the model from the local file
model = load_model("model.h5")

# Preprocess image function
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((256, 256))  # Resize image to match model input shape
    img = np.array(img) / 255.0  # Normalize pixel values
    return img

# Getting IP address
def get_ip_address():
    # Get the hostname of the machine
    hostname = socket.gethostname()
    # Get the IP address corresponding to the hostname
    ip_address = socket.gethostbyname(hostname)
    return ip_address

@app.route('/')
def home():
    return "Flask server is running"

class_labels = ['Bacterial', 'Fungal', 'Hama', 'Healthy', 'Virus']

@app.route('/predict', methods=['POST'])
def predict():
    # Check if request contains image file
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image = request.files['image']

    try:
        # Read image bytes
        image_bytes = image.read()
        
        # Preprocess image
        processed_image = preprocess_image(image_bytes)
        
        # Predict using the loaded model
        predictions = model.predict(np.expand_dims(processed_image, axis=0))

        # Map predictions to class labels
        prediction_index = np.argmax(predictions)
        predicted_class = class_labels[prediction_index]
        confidence_score = float(predictions[0][prediction_index])

        # Return the predicted class as JSON
        return jsonify({
            'predicted_class': predicted_class,
            'confidence_score': confidence_score
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    ip_address = get_ip_address()
    print("Flask server is running")
    app.run(host='0.0.0.0', port=5000, debug=True)
