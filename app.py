from flask import Flask, request, render_template
import pickle
import numpy as np

import boto3
import os
from flask import Flask

AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
BUCKET_NAME = os.getenv('BUCKET_NAME')
OBJECT_NAME = 'model.pkl'  # The name of the file in S3
LOCAL_FILE_NAME = 'model.pkl'  # The local file name

s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY,
                  aws_secret_access_key=AWS_SECRET_KEY)

s3.download_file(BUCKET_NAME, OBJECT_NAME, LOCAL_FILE_NAME)

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    MedInc = float(request.form['MedInc'])
    HouseAge = float(request.form['HouseAge'])
    AveRooms = float(request.form['AveRooms'])
    AveBedrms = float(request.form['AveBedrms'])
    Population = float(request.form['Population'])
    AveOccup = float(request.form['AveOccup'])
    Latitude = float(request.form['Latitude'])
    Longitude = float(request.form['Longitude'])

    # Create a numpy array with the input values
    features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])

    # Make the prediction
    predicted_price = model.predict(features)[0]

    return render_template('prediction.html', predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
