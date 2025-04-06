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
        # Get input values
        med_inc = float(request.form['MedInc']) / 10000  # Match your scaling
        house_age = float(request.form['HouseAge'])
        ave_rooms = float(request.form['AveRooms'])
        model_choice = request.form.get('model', 'nn')  # Default to Neural Network (best MSE)

        # Prepare and scale input
        input_data = np.array([[med_inc, house_age, ave_rooms]])
        input_scaled = scaler.transform(input_data)

        # Make predictions
        lr_pred = lr_model.predict(input_scaled)[0] * 1000  # $K
        rf_pred = rf_model.predict(input_scaled)[0] * 1000  # $K
        nn_pred = nn_model.predict(input_scaled, verbose=0)[0][0] * 1000  # $K

        # Format predictions
        predictions = {
            'lr': f"{lr_pred:,.2f}",
            'rf': f"{rf_pred:,.2f}",
            'nn': f"{nn_pred:,.2f}"
        }
        selected_model = {'lr': 'Linear Regression', 'rf': 'Random Forest', 'nn': 'Neural Network'}[model_choice]

        return render_template('index.html', 
                              predictions=predictions, 
                              selected_model=selected_model,
                              med_inc=request.form['MedInc'],  # Pass raw input for display
                              house_age=house_age,
                              ave_rooms=ave_rooms)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
