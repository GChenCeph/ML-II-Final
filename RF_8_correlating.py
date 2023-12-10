import json
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Load JSON and CSV data
def load_json_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

def load_csv_to_matrix(filename):
    df = pd.read_csv(filename)
    return df.iloc[:, 1:].values.tolist()  # Exclude the first column (hotel identifiers)

# Function to count utilities within a radius for a specific category
def count_utilities_within_radius(dist_matrix, threshold, category, utils_df):
    category_indices = utils_df[utils_df['Category Name'] == category].index.tolist()
    return [sum(dist_matrix[hotel_idx][util_idx] <= threshold for util_idx in category_indices) for hotel_idx in range(len(dist_matrix))]

# Load data
city_prefix = 'hu'
hotels_data = load_json_data(f'{city_prefix}_hotel_dropped.json')
utils_filtered = load_json_data(f'{city_prefix}_utils_filtered.json')
distance_matrix = load_csv_to_matrix(f'{city_prefix}_distance_matrix.csv')
utils_df = pd.DataFrame(utils_filtered)

# Define a distance threshold (e.g., 1.6 km)
distance_threshold = 1.6

# Calculate correlations for each category and include summed utilities count
category_correlations = {}
unique_categories = utils_df['Category Name'].unique()
for category in unique_categories:
    utilities_count = count_utilities_within_radius(distance_matrix, distance_threshold, category, utils_df)
    total_scores = [hotel.get('Total Score', 0) for hotel in hotels_data]
    revenue = [hotel.get('Total Room Receipts', 0) for hotel in hotels_data]

    # Handle nan values and calculate correlations
    correlation_score, p_value_score = pearsonr(utilities_count, total_scores) if len(utilities_count) > 1 else (0, 1)
    correlation_revenue, p_value_revenue = pearsonr(utilities_count, revenue) if len(utilities_count) > 1 else (0, 1)

    # Add data to the dictionary
    category_correlations[category] = {
        'utilities_count': utilities_count,
        'score_correlation': 0 if np.isnan(correlation_score) else correlation_score,
        'score_p_value': 0 if np.isnan(p_value_score) else p_value_score,
        'revenue_correlation': 0 if np.isnan(correlation_revenue) else correlation_revenue,
        'revenue_p_value': 0 if np.isnan(p_value_revenue) else p_value_revenue
    }

# Filter out categories with zero correlation in revenue or score
filtered_correlations = {cat: val for cat, val in category_correlations.items() if val['revenue_correlation'] != 0
                         and val['score_correlation'] != 0}

# Save the filtered correlations back to a json file
with open('filtered_category_correlations.json', 'w', encoding='utf-8') as file:
    json.dump(filtered_correlations, file, indent=4)

# If needed, print the output
for category, correlations in filtered_correlations.items():
    print(f"Category: {category}")
    print(f"Utilities Count: {correlations['utilities_count']}")
    print(f"Score Correlation: {correlations['score_correlation']}, p-value: {correlations['score_p_value']}")
    print(f"Revenue Correlation: {correlations['revenue_correlation']}, p-value: {correlations['revenue_p_value']}")
