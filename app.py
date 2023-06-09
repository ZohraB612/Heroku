from flask import Flask, request, render_template
import pickle
import numpy as np

import boto3
import joblib
from io import BytesIO
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Read the AWS credentials from environment variables
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

# Replace with your bucket name
bucket_name = 'week5-dg'

s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
obj = s3.get_object(Bucket=bucket_name, Key='model.pkl')
model = joblib.load(BytesIO(obj['Body'].read()))

app = Flask(__name__)

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
