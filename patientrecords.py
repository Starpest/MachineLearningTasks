import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Data Collection
# Assume you have a CSV file named "network_data.csv" with columns: Timestamp, Latency, PacketLoss, NetworkUtilization, Stability

# Step 2: Data Preprocessing
data = pd.read_csv("network_data.csv")
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data.sort_values(by='Timestamp', inplace=True)

# Step 3: Data Splitting
X = data[['Latency', 'PacketLoss', 'NetworkUtilization']]
y = data['Stability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Training
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Step 5: Model Evaluation
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

train_mse = mean_squared_error(y_train, train_predictions)
test_mse = mean_squared_error(y_test, test_predictions)

train_r2 = r2_score(y_train, train_predictions)
test_r2 = r2_score(y_test, test_predictions)

print("Training MSE:", train_mse)
print("Testing MSE:", test_mse)
print("Training R-squared:", train_r2)
print("Testing R-squared:", test_r2)

# Step 6: Predictions (Optional)
# You can use the trained model to make predictions for new data, if available.

# Example prediction for new data:
new_data = pd.DataFrame([[100, 0.5, 70]], columns=['Latency', 'PacketLoss', 'NetworkUtilization'])
prediction = model.predict(new_data)
print("Predicted Stability:", prediction)
