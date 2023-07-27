import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Step 1: Data Collection
# Assume you have a CSV file named "sales_data.csv" with columns: Date, Sales

# Step 2: Data Preprocessing
data = pd.read_csv("sales_data.csv")
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values(by='Date', inplace=True)

# Feature Engineering (Extract month and day-of-week as new features)
data['Month'] = data['Date'].dt.month
data['DayOfWeek'] = data['Date'].dt.dayofweek

# Step 3: Data Splitting
X = data[['Month', 'DayOfWeek']]
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Model Evaluation
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Train R-squared: {train_score:.2f}")
print(f"Test R-squared: {test_score:.2f}")

# Step 6: Predictions
predictions = model.predict(X_test)

# Step 7: Visualization
plt.figure(figsize=(10, 6))
plt.scatter(data['Date'], data['Sales'], label='Actual Sales', color='b')
plt.plot(data['Date'].iloc[len(X_train):], predictions, label='Predicted Sales', color='r')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Sales Forecasting using Linear Regression')
plt.legend()
plt.show()
