# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Step 2: Generate a Sample Dataset (for demonstration purposes)
np.random.seed(42)

# Generate a synthetic dataset with 1000 entries
data = pd.DataFrame({
    'Ad_Budget': np.random.randint(1000, 10000, 1000),  # Ad budget between 1000 and 10000
    'Platform': np.random.choice(['TV', 'Facebook', 'Instagram'], 1000),  # Randomly assign platforms
    'Season': np.random.choice(['Winter', 'Spring', 'Summer', 'Fall'], 1000),  # Random seasons
    'Sales': np.random.randint(100, 1000, 1000)  # Random sales numbers
})

# Step 3: Data Exploration
print(data.head())
print(data.info())

# Step 4: Handle Categorical Variables (Label Encoding)
label_encoder = LabelEncoder()

# Encode the categorical columns ('Platform' and 'Season')
data['Platform'] = label_encoder.fit_transform(data['Platform'])
data['Season'] = label_encoder.fit_transform(data['Season'])

# Step 5: Feature Selection (X) and Target Variable (y)
X = data.drop('Sales', axis=1)  # Features (Ad_Budget, Platform, Season)
y = data['Sales']  # Target (Sales)

# Step 6: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train a Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Step 8: Train a Random Forest Regressor Model (for comparison)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 9: Predictions
y_pred_linear = linear_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# Step 10: Evaluate the Models
# Linear Regression Metrics
mae_linear = mean_absolute_error(y_test, y_pred_linear)
mse_linear = mean_squared_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mse_linear)
r2_linear = r2_score(y_test, y_pred_linear)

# Random Forest Metrics
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Step 11: Print Evaluation Metrics
print("\nLinear Regression Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae_linear}")
print(f"Mean Squared Error (MSE): {mse_linear}")
print(f"Root Mean Squared Error (RMSE): {rmse_linear}")
print(f"R-Squared (R2): {r2_linear}")

print("\nRandom Forest Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae_rf}")
print(f"Mean Squared Error (MSE): {mse_rf}")
print(f"Root Mean Squared Error (RMSE): {rmse_rf}")
print(f"R-Squared (R2): {r2_rf}")

# Step 12: Visualize Actual vs Predicted Sales (Linear Regression)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_linear)
plt.title('Actual vs Predicted Sales (Linear Regression)')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.show()

# Step 13: Visualize Actual vs Predicted Sales (Random Forest)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf)
plt.title('Actual vs Predicted Sales (Random Forest)')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.show()

# Step 14: Save the Model (Optional)
import joblib
joblib.dump(linear_model, 'sales_prediction_linear_model.pkl')  # Save Linear Regression model
joblib.dump(rf_model, 'sales_prediction_rf_model.pkl')  # Save Random Forest model
