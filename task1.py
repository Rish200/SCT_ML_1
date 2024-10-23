import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd


df = pd.read_csv('C:/Users/Rishav Roshan/OneDrive/Desktop/ML Project/train.csv')


print(df.head())


# Select features
features = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]

# Target variable
target = df['SalePrice']

# Check for missing values
print(features.isnull().sum())

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Calculate the R² score
r2 = r2_score(y_test, y_pred)
print(f'R² Score: {r2}')

# Plot actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs Predicted House Prices')
plt.show()

import joblib
joblib.dump(model, 'house_price_model.pkl')

# Load test data
test_df = pd.read_csv('test.csv')

# Prepare the test features
test_features = test_df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]

# Predict house prices
test_predictions = model.predict(test_features)

# Save predictions to a CSV file
output = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': test_predictions})
output.to_csv('house_price_predictions.csv', index=False)
