from flask import Flask, request, jsonify
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the trained k-NN model
model_filename = "knn_model.xml"
knn = cv2.ml.KNearest_load(model_filename)

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

    # Flatten the image to a 1D array
    processed_img = processed_img.flatten().astype(np.float32)
    
    # Normalize the image
    processed_img = processed_img / 255.0
    
    return processed_img

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

        # Process the image
        processed_img = process_image(file)

        # Reshape and make prediction
        processed_img = processed_img.reshape(1, -1)  # Reshape to match the input shape
        ret, results, neighbors, dist = knn.findNearest(processed_img, k=5)

        predicted_digit = int(results[0][0])

        return jsonify({"prediction": predicted_digit})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Main function to run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
from flask_cors import CORS
import cv2
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the trained k-NN model
model_filename = "knn_model.xml"
knn = cv2.ml.KNearest_load(model_filename)

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

    # Flatten the image to a 1D array
    processed_img = processed_img.flatten().astype(np.float32)
    
    # Normalize the image
    processed_img = processed_img / 255.0
    
    return processed_img

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

        # Process the image
        processed_img = process_image(file)

        # Reshape and make prediction
        processed_img = processed_img.reshape(1, -1)  # Reshape to match the input shape
        ret, results, neighbors, dist = knn.findNearest(processed_img, k=5)

        predicted_digit = int(results[0][0])

        return jsonify({"prediction": predicted_digit})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Main function to run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
