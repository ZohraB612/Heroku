import requests
import json
import numpy as np
from PIL import Image

# Load your input image
image_path = 'uploads/file.jpg'  # Replace with the path to the uploaded image
image = Image.open(image_path)  # Open the uploaded image
image = image.convert('L')  # Convert to grayscale
image = image.resize((28, 28))  # Resize to 28x28 pixels

# Convert the image to a numpy array
data = np.array(image) / 255.0  # Normalize pixel values to range 0-1
data = data.reshape(1, 28, 28)  # Reshape to match the model's input shape

# Define the URL for the predict endpoint
url = 'http://localhost:5000/predict'  # Replace with your actual URL

# Create the request payload as JSON
payload = {'data': data.tolist()}
headers = {'Content-Type': 'application/json'}

# Send the POST request
response = requests.post(url, json=payload, headers=headers)

# Get the prediction result
prediction = response.json()

# Print the prediction
print('Prediction:', prediction)