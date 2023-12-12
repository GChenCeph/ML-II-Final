import json
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

#####################################################################
def load_json_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

def load_csv_to_matrix(filename):
    df = pd.read_csv(filename)
    return df.iloc[:, 1:].values.tolist()

# Function to count utilities within a radius for a specific category
def count_utilities_within_radius(dist_matrix, threshold, category, utils_df):
    category_indices = utils_df[utils_df['Category Name'] == category].index.tolist()
    return [sum(dist_matrix[hotel_idx][util_idx] <= threshold for util_idx in category_indices) for hotel_idx in range(len(dist_matrix))]
#####################################################################

# Load data
city_prefix = 'hu'  # City prefix for file naming
hotels_data = load_json_data(f'{city_prefix}_hotel_revenue.json')  # Loading hotel data from a JSON file
utils_filtered = load_json_data(f'{city_prefix}_utils_filtered.json')  # Loading filtered utilities data from a JSON file
distance_matrix = load_csv_to_matrix(f'{city_prefix}_distance_matrix.csv')  # Loading distance matrix data from a CSV file
utils_df = pd.DataFrame(utils_filtered)  # Converting the list of utilities data into a DataFrame

# Setting the best distance threshold
distance_threshold = 1.69

category_correlations = {}  # Dictionary to store correlation results
unique_categories = utils_df['Category Name'].unique()  # Getting unique categories from utility data

# Iterating over each unique category
for category in unique_categories:

    # Counting utilities within the specified threshold for each hotel
    utilities_count = count_utilities_within_radius(distance_matrix, distance_threshold, category, utils_df)
    # Extracting total scores for each hotel
    total_scores = [hotel.get('Total Score', 0) for hotel in hotels_data]
    # Extracting total room receipts (revenue) for each hotel
    revenue = [hotel.get('Total Room Receipts', 0) for hotel in hotels_data]

    # Calculating Pearson correlation and p-value between utilities count and total scores
    correlation_score, p_value_score = pearsonr(utilities_count, total_scores) if len(utilities_count) > 1 else (0, 1)
    # Calculating Pearson correlation and p-value between utilities count and revenue
    correlation_revenue, p_value_revenue = pearsonr(utilities_count, revenue) if len(utilities_count) > 1 else (0, 1)

    # Handling NaN values in correlation scores and p-values
    correlation_score = 0 if np.isnan(correlation_score) else correlation_score
    p_value_score = 0 if np.isnan(p_value_score) else p_value_score
    correlation_revenue = 0 if np.isnan(correlation_revenue) else correlation_revenue
    p_value_revenue = 0 if np.isnan(p_value_revenue) else p_value_revenue

    # Storing the correlation results in the dictionary
    category_correlations[category] = {
        'score_correlation': correlation_score,
        'score_p_value': p_value_score,
        'revenue_correlation': correlation_revenue,
        'revenue_p_value': p_value_revenue
    }

# Writing the correlation results to a JSON file
with open('pearson_cate_corr.json', 'w', encoding='utf-8') as file:
    json.dump(category_correlations, file, indent=4)