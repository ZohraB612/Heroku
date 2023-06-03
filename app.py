from flask import Flask, request, jsonify, render_template
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
    # Get the data from the POST request
    sepal_length = request.form['sepal_length']
    sepal_width = request.form['sepal_width']
    petal_length = request.form['petal_length']
    petal_width = request.form['petal_width']

    # Make prediction
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])

    # Take the first value of prediction
    output = prediction[0]

    return render_template('prediction.html', 
                           sepal_length=sepal_length,
                           sepal_width=sepal_width,
                           petal_length=petal_length,
                           petal_width=petal_width,
                           predicted_class=output)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
