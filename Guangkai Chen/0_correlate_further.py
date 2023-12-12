import csv
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Helper function to load JSON data
def load_json_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

# Load the distance matrix from a CSV file, excluding the first column (hotel identifiers)
def load_csv_to_matrix(filename):
    df = pd.read_csv(filename)
    return df.iloc[:, 1:].values.tolist()  # Exclude the first column (hotel identifiers)

# Count the number of utilities within a specified radius for each hotel
def count_utilities_within_radius(dist_matrix, threshold):
    return [sum(dist <= threshold for dist in row) for row in dist_matrix]

def save_to_csv(thresholds, values, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Writing headers
        headers = ['Distance_km', 'Value']
        csvwriter.writerow(headers)
        
        # Writing values for each distance threshold
        for threshold, value in zip(thresholds, values):
            csvwriter.writerow([threshold, value])

# List of city prefixes
city_prefixes = ['hu']

#distance_threshold = 1.0
p = []
pearson = []

# Process each city's data
distance_thresholds = np.arange(1, 65)  # Create an array from 1 to 3 with a step of 0.01
for distance_threshold in distance_thresholds:
    hotels_data = load_json_data('hu_hotel_revenue.json')
    distance_matrix = load_csv_to_matrix('hu_distance_matrix.csv')

    utilities_count_within_radius = count_utilities_within_radius(distance_matrix, distance_threshold)

    total_scores = [hotel.get('Total Score', 0) for hotel in hotels_data]

    correlation, p_value = pearsonr(utilities_count_within_radius, total_scores)

    pearson.append(correlation)
    p.append(p_value)

save_to_csv(distance_thresholds, p, 'p.csv')
save_to_csv(distance_thresholds, pearson, 'pearson.csv')