import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json

# Load category correlations data
file_path_correlations = 'C:/Users/ompat/Documents/School/MEng/Fall_23/ML2/Project-Combine/filtered_category_correlations.json'
with open(file_path_correlations, 'r', encoding='utf-8') as file:
    category_correlations = json.load(file)

# Load hotel data
file_path_hotels = 'C:/Users/ompat/Documents/School/MEng/Fall_23/ML2/Project-Combine/hu_hotel_dropped.json'  # Updated file path
hotels_data = pd.read_json(file_path_hotels)

# Prepare the DataFrame
hotel_data = pd.DataFrame()

# Prepare a dictionary to hold the new data
new_data = {}

# Add utility counts to new_data
for category, data in category_correlations.items():
    new_data[category] = data['utilities_count']

# Add ratings to new_data
#new_data['rating'] = hotels_data['Total Score']

# Add room Capacity to new_data
#new_data['Total Rooms'] = hotels_data['Unit Capacity']

# Convert new_data to a DataFrame
new_data_df = pd.DataFrame(new_data)

# Now combine this with the existing hotel_data using pd.concat
hotel_data = pd.concat([hotel_data, new_data_df], axis=1)


# Target variable
y = np.log(hotels_data['Total Room Receipts'] / 1000)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(hotel_data, y, test_size=0.2, random_state=42)
print(X_train.head())
print(X_train.shape[1])
# Hyperparameter tuning for RandomForestRegressor
param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Use the best estimator
best_model = grid_search.best_estimator_

# Model evaluation
predictions = best_model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R² Score: {r2}")
print(f"Best Model Parameters: {grid_search.best_params_}")
print(f"Mean Squared Error: {mse}")

# Feature importances
feature_importances = best_model.feature_importances_
importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

print(importance_df)
