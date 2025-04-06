import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)[['MedInc', 'HouseAge', 'AveRooms']].values
y = housing.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
joblib.dump(lr_model, 'lr_model.pkl')

# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
joblib.dump(rf_model, 'rf_model.pkl')

# Train Neural Network
nn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
nn_model.compile(optimizer='adam', loss='mse')
nn_model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=1)
nn_model.save('nn_model.keras')

# Save scaler
joblib.dump(scaler, 'scaler.pkl')

# Evaluate
lr_pred = lr_model.predict(X_test_scaled)
rf_pred = rf_model.predict(X_test_scaled)
nn_pred = nn_model.predict(X_test_scaled).flatten()

print("Linear Regression MSE:", mean_squared_error(y_test, lr_pred))
print("Random Forest MSE:", mean_squared_error(y_test, rf_pred))
print("Neural Network MSE:", mean_squared_error(y_test, nn_pred))
print("Neural Network RMSE:", np.sqrt(mean_squared_error(y_test, nn_pred)))