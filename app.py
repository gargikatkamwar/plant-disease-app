import streamlit as st
st.write("Hello, this is a test")
import json
import numpy as np
from PIL import Image
import io
import os

# Ensure the model and class_indices are available in the current directory.
# For actual deployment, place these files alongside app.py.

# Try to import Flask. This will fail if Flask is not installed, but it's mainly for the app.py structure.
try:
    from flask import Flask, request, jsonify
except ImportError:
    print("Flask not found. Install with: pip install Flask")
    # Define a dummy Flask class to allow the rest of the code to parse without error
    class Flask:
        def __init__(self, name):
            pass
        def route(self, rule, methods=None):
            def decorator(f):
                return f
            return decorator
    class request: # Dummy request object
        files = {}
    def jsonify(data): # Dummy jsonify function
        return str(data)

import tensorflow as tf

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('plant_disease_model.keras')

# Load class indices
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Create a reverse mapping for predictions
idx_to_class = {v: k for k, v in class_indices.items()}

# Define target image size (must match what the model was trained on)
IMG_SIZE = (64, 64)

def preprocess_image(image_bytes):
    # Open image from bytes
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    # Resize image
    image = image.resize(IMG_SIZE)
    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0
    # Expand dimensions to create a batch of 1
    return np.expand_dims(image_array, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        image_bytes = file.read()
        preprocessed_image = preprocess_image(image_bytes)

        # Make prediction
        predictions = model.predict(preprocessed_image)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        predicted_class_name = idx_to_class[predicted_class_idx]
        confidence = float(np.max(predictions))

        return jsonify({
            'prediction': predicted_class_name,
            'confidence': confidence
        })

if __name__ == '__main__':
    # In a real deployment, you might configure host='0.0.0.0'
    # For Colab demonstration, running directly won't expose a web server.
    # This part is typically run from a terminal.
    print("To run this Flask app, save it as app.py and execute 'python app.py' from your terminal.")
    print("Make sure 'plant_disease_model.h5' and 'class_indices.json' are in the same directory.")
    # app.run(debug=True) # Uncomment for local testing outside Colab
