from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from PIL import Image

def process_image(image_path):
    # Open the image file
    image = Image.open(image_path)

    # Resize the image to 28x28
    resized_image = image.resize((28, 28))

    # Convert the image to grayscale<
    grayscale_image = resized_image.convert('L') 

    # new comment

    # Normalize the pixel values
    normalized_image = np.array(grayscale_image) / 255.0

    # Reshape the image to match the model's input shape
    processed_data = normalized_image.reshape(1, 28, 28)

    return processed_data

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('mnist_model.h5')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded file
    uploaded_file = request.files['file']

    # Save the file
    file_path = 'uploads/file.jpg'
    uploaded_file.save(file_path)

    # Process the image file
    processed_data = process_image(file_path)

    # Make the prediction
    prediction = model.predict(processed_data)

    # Get the predicted class as an integer
    predicted_class = np.argmax(prediction).item()

    # Render the prediction result page with the preprocessed image URL and predicted class
    return render_template('prediction.html', image_url=file_path, predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)