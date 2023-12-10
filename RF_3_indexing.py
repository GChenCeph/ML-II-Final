import json

# Helper function to load JSON data
def load_json_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

# Helper function to save JSON data
def save_json_data(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

# Function to filter out entries without valid coordinates
def filter_by_coordinates(data):
    return [item for item in data if item.get('Longitude') is not None and item.get('Latitude') is not None]

# Function to further filter hotels by 'Total Score'
def filter_hotels_by_score(hotels_data):
    return [hotel for hotel in hotels_data if isinstance(hotel.get('Total Score'), (int, float))]

# Function to add an index to each item in a list of dictionaries
def add_indexes_to_data(data):
    for index, entry in enumerate(data):
        entry['Index'] = index
    return data

# Process and save filtered and indexed data
def process_and_save_filtered_data(city_prefix):
    hotels_data = load_json_data('hu_hotel_merged.json')
    utils_data = load_json_data(f'{city_prefix}_utils_post.json')

    filtered_hotels = filter_hotels_by_score(filter_by_coordinates(hotels_data))
    filtered_utils = filter_by_coordinates(utils_data)

    indexed_hotels = add_indexes_to_data(filtered_hotels)
    indexed_utils = add_indexes_to_data(filtered_utils)

    save_json_data(indexed_hotels, f'{city_prefix}_hotel_filtered.json')
    save_json_data(indexed_utils, f'{city_prefix}_utils_filtered.json')

# List of city prefixes
city_prefixes = ['hu']

process_and_save_filtered_data(city_prefixes[0])
