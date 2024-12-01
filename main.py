from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import pickle
import os

#hello
# Initialize Flask app
app = Flask(__name__)
CORS(app)  # This enables CORS for all routes, allowing requests from any domain

# Load the trained MLP model
model_filename = "mlp_model.pkl"
with open(model_filename, 'rb') as f:
    model = pickle.load(f)

weights_1 = model['weights_1']
biases_1 = model['biases_1']
weights_2 = model['weights_2']
biases_2 = model['biases_2']
weights_3 = model['weights_3']
biases_3 = model['biases_3']

# Leaky ReLU activation function
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# Softmax activation function
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Forward pass of the neural network
def forward_pass(X, weights_1, biases_1, weights_2, biases_2, weights_3, biases_3):
    hidden_layer_1 = leaky_relu(np.dot(X, weights_1) + biases_1)
    hidden_layer_2 = leaky_relu(np.dot(hidden_layer_1, weights_2) + biases_2)
    output_layer = softmax(np.dot(hidden_layer_2, weights_3) + biases_3)
    return hidden_layer_1, hidden_layer_2, output_layer

# Function to process the image
def process_image(image_file):
    # Read the image
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError("Invalid image")

    # If the image has an alpha channel (transparency), remove it
    if img.shape[2] == 4:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img_rgb[img[:, :, 3] == 0] = [255, 255, 255]  # Replace transparent pixels with white
    else:
        img_rgb = img

    # Convert to grayscale
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    # Thresholding to ensure digits are black and background is white
    _, img_bin = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Resize to 28x28 while maintaining aspect ratio
    processed_img = resize_with_aspect_ratio(img_bin, (28, 28))

    # Flatten the image to a 1D array (used for prediction)
    processed_img_for_prediction = processed_img.flatten().astype(np.float32)
    
    # Normalize the image
    processed_img_for_prediction = processed_img_for_prediction / 255.0
    
    return processed_img, processed_img_for_prediction

# Function to resize image with aspect ratio
def resize_with_aspect_ratio(image, target_size=(28, 28)):
    h, w = image.shape
    scale = min(target_size[0] / h, target_size[1] / w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized_image = cv2.resize(image, (new_w, new_h))
    canvas = np.full(target_size, 255, dtype=np.uint8)

    y_offset = (target_size[0] - new_h) // 2
    x_offset = (target_size[1] - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image
    
    return canvas

# API route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if an image file is part of the request
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Process the image and get the processed image for prediction
        processed_img, processed_img_for_prediction = process_image(file)

        # Reshape the image and make prediction using MLP
        processed_img_for_prediction = processed_img_for_prediction.reshape(1, -1)  # Reshape to match the input shape
        hidden_layer_1, hidden_layer_2, output_layer = forward_pass(
            processed_img_for_prediction, weights_1, biases_1, weights_2, biases_2, weights_3, biases_3
        )

        predicted_digit = int(np.argmax(output_layer, axis=1)[0])

        # Convert the processed image to a base64 encoded string (2D image, not the flattened 1D array)
        _, buffer = cv2.imencode('.png', processed_img)  # Use the original processed image here
        img_str = base64.b64encode(buffer).decode('utf-8')

        return jsonify({"prediction": predicted_digit, "processed_image": img_str})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Main function to run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
