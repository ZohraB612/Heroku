from flask import Flask, request, render_template
import pickle
import numpy as np

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
