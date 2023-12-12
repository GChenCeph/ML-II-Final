import json
import pandas as pd

# Load JSON data
def load_json_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

# Load hotel revenue, utility data, and correlations
hotels_data = load_json_data('hu_hotel_revenue.json')
spearman_correlations = load_json_data('spearman_cate_corr.json')
pearson_correlations = load_json_data('pearson_cate_corr.json')
revenue = [hotel.get('Total Room Receipts', 0) / 1000 for hotel in hotels_data]
#print(revenue[:10])

# Load distance matrix (rows as hotels, columns as utilities)
distance_matrix = pd.read_csv('hu_distance_matrix_cate.csv', index_col=0)

# Load distance vector (assuming a CSV with hotel index and distance)
distance_vector = pd.read_csv('hu_distance_vector.csv', index_col=0)

# Calculate the simple average distance for each hotel
avg_distance = distance_matrix.mean(axis=1)

# Adjust the calculate_weighted_average function to explicitly map correlations
def calculate_weighted_average(dist_matrix, correlations):
    # Initialize all weights to zero
    weights = pd.Series(0, index=dist_matrix.columns)

    # Assign weights based on correlation values
    for category, info in correlations.items():
        if category in dist_matrix.columns:
            weights[category] = info['revenue_correlation']

    # Calculate the weighted average
    weighted_matrix = dist_matrix.mul(weights, axis=1)
    weighted_avg = weighted_matrix.sum(axis=1) / weights.sum()

    #print("Weighted averages:", weighted_avg.head())
    return weighted_avg

# Calculate Pearson and Spearman weighted averages
pearson_weighted_avg = calculate_weighted_average(distance_matrix, pearson_correlations)
spearman_weighted_avg = calculate_weighted_average(distance_matrix, spearman_correlations)

# Incorporate these values and the original data into hotel data
for hotel in hotels_data:
    index = hotel['Index']
    hotel['avg_distance'] = avg_distance.get(index, 0)
    hotel['pearson_weighted_avg_distance'] = pearson_weighted_avg.get(index, 0)
    hotel['spearman_weighted_avg_distance'] = spearman_weighted_avg.get(index, 0)
    hotel['distance_vector'] = distance_vector.at[index, 'Distance'] if index in distance_vector.index else 0
    hotel['revenue'] = revenue[index]

# Save the updated hotel data to a new JSON file
with open('hu_hotel_weighted.json', 'w', encoding='utf-8') as outfile:
    json.dump(hotels_data, outfile, indent=4)
