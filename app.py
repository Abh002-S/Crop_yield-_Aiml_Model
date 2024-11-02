# app.py

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("crop_yield_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    rainfall = float(request.form['rainfall'])
    tonnes = float(request.form['tonnes'])
    temp = float(request.form['temp'])

    # Scale the input data
    input_data = np.array([[rainfall, tonnes, temp]])
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)

    # Return the result
    return jsonify({'predicted_yield': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
