# 29.7604° N, 95.3698° W, center of Houston
import json
import csv
from math import radians, cos, sin, asin, sqrt

# Haversine function to calculate the great circle distance between two points
def haversine(lon1, lat1, lon2, lat2):
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r

# Load data from a JSON file
def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

# Create a distance matrix from the hotel data
def cal_distance_matrix(city_long, city_lat, hotels_data):
    distance_vector = []
    for hotel in hotels_data:
        hotel_long = hotel['Longitude']
        hotel_lat = hotel['Latitude']
        distance = haversine(city_long, city_lat, hotel_long, hotel_lat)
        distance_vector.append(distance)
    return distance_vector

# Function to save the distance vector to a CSV file
def save_distance_vector_to_csv(distance_vector, hotels_data, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Writing headers
        headers = ['Hotel', 'Distance']
        csvwriter.writerow(headers)
        
        # Writing distance rows for each hotel
        for i, distance in enumerate(distance_vector):
            csvwriter.writerow([hotels_data[i]["Index"], distance])

# Process each city's hotel data
hotels_data = load_data('hu_hotel_revenue.json')
# Coordinates for Houston city center
houston_long = -95.3698
houston_lat = 29.7604
distance_vector = cal_distance_matrix(houston_long, houston_lat, hotels_data)
    
# Save the distance vector to a CSV file using the custom function
save_distance_vector_to_csv(distance_vector, hotels_data, 'hu_distance_vector.csv')
