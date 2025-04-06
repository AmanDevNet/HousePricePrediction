import numpy as np
import joblib
import tensorflow as tf
from flask import Flask, render_template, request

app = Flask(__name__)

# Load trained models
lr_model = joblib.load('lr_model.pkl')
rf_model = joblib.load('rf_model.pkl')
nn_model = tf.keras.models.load_model('nn_model.keras')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values (keys must match the HTML 'name' fields)
        med_inc = float(request.form['MedInc']) / 10000  # to scale
        house_age = float(request.form['HouseAge'])
        avg_rooms = float(request.form['AveRooms'])

        # Prepare and scale input
        input_data = np.array([[med_inc, house_age, avg_rooms]])
        input_scaled = scaler.transform(input_data)

        # Make predictions
        lr_pred = lr_model.predict(input_scaled)[0] * 1000
        rf_pred = rf_model.predict(input_scaled)[0] * 1000
        nn_pred = nn_model.predict(input_scaled)[0][0] * 1000

        return render_template('index.html',
                               lr_prediction=f"{lr_pred:,.2f}",
                               rf_prediction=f"{rf_pred:,.2f}",
                               nn_prediction=f"{nn_pred:,.2f}")
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
