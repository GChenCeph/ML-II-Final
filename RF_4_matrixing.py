import json
import csv
from math import radians, cos, sin, asin, sqrt

# Haversine function to calculate the great circle distance between two points
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles.
    return c * r

# Load data from a JSON file
def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

# Create a distance matrix from the hotel and utility data
def create_distance_matrix(hotels_data, utils_data):
    distance_matrix = []
    for hotel in hotels_data:
        hotel_row = []
        for util in utils_data:
            distance = haversine(hotel['Longitude'], hotel['Latitude'], util['Longitude'], util['Latitude'])
            hotel_row.append(distance)
        distance_matrix.append(hotel_row)
    return distance_matrix

# Function to save the distance matrix to a CSV file
def save_distance_matrix_to_csv(distance_matrix, hotels_data, utils_data, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Writing headers
        headers = ['Hotel'] + [f'Util_{util["Index"]}' for util in utils_data]
        csvwriter.writerow(headers)
        
        # Writing distance rows for each hotel
        for i, row in enumerate(distance_matrix):
            csvwriter.writerow([f'Hotel_{hotels_data[i]["Index"]}'] + row)

# List of city prefixes
city_prefixes = ['hu']


# Process each city's hotel and utilities data
for city_prefix in city_prefixes:
    hotels_data = load_data(f'{city_prefix}_hotel_filtered.json')
    utils_data = load_data(f'{city_prefix}_utils_filtered.json')
    
    distance_matrix = create_distance_matrix(hotels_data, utils_data)
    
    # Save the distance matrix to a CSV file using the custom function
    save_distance_matrix_to_csv(distance_matrix, hotels_data, utils_data, f'{city_prefix}_distance_matrix.csv')
