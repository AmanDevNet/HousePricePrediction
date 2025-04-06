# HousePricePrediction

House Price Prediction using ML & Deep Learning
This project is a Machine Learning + Deep Learning based web application built with Flask that predicts house prices using the California Housing Dataset. It includes:

Linear Regression

Random Forest

Neural Network (TensorFlow)

The user can input key house features like median income, house age, and average number of rooms — and get predictions from all three models.

🚀 Features
🔢 Trained with real-world California housing data

📈 Predict house price using:

Linear Regression

Random Forest

Neural Network (Keras)

🌐 Web interface using Flask

📊 Scaled inputs for accurate prediction

🖥️ Tech Stack
Layer	Technology
Language	Python
Web Framework	Flask
ML Models	scikit-learn, TensorFlow
Frontend	HTML, CSS
Dataset	California Housing (sklearn)
🛠️ Installation
Clone the repository

git clone https://github.com/your-username/house-price-predictor.git
cd house-price-predictor
Create virtual environment (recommended)


python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies


pip install -r requirements.txt
Train the models

python housePrice.py
Run the Flask app

python app.py
Open your browser and go to:
http://127.0.0.1:5000/

📁 File Structure

├── app.py                 # Flask web app
├── housePrice.py          # ML/DL model training
├── scaler.pkl             # StandardScaler for inputs
├── lr_model.pkl           # Trained Linear Regression model
├── rf_model.pkl           # Trained Random Forest model
├── nn_model.keras         # Trained Neural Network model
├── templates/
│   └── index.html         # Frontend UI
└── README.md              # Project documentation
✨ Sample Prediction Input
Feature	Example Value
MedInc	8.3252
HouseAge	41.0
AveRooms	6.984
