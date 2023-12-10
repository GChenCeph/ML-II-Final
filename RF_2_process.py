import json
import pandas as pd
import chardet
def extract_full_reviews(entry):

    # Extract and modify the address
    full_address = entry.get('address', '')
    # Assuming the street address is separated by a comma
    street_address = full_address.split(',')[0] if full_address else ''

    extracted_entry = {
        'Address': street_address,
        'Category Name': entry.get('categoryName', ''),
        'Cid': entry.get('cid', ''),
        'City': entry.get('city', ''),
        'Latitude': entry.get('location', {}).get('lat', None),
        'Longitude': entry.get('location', {}).get('lng', None),
        'Place ID': entry.get('placeId', ''),
        'Rank': entry.get('rank', ''),
        'Reviews Distribution': entry.get('reviewsDistribution', {}),
        'Street': entry.get('street', ''),
        'Title': entry.get('title', ''),
        'Total Score': entry.get('totalScore', '')
    }
    return extracted_entry



def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read(10000)  # Read first 10000 bytes to guess the encoding
        encoding = chardet.detect(raw_data)['encoding']
        print(encoding)  # Print the detected encoding
        return encoding

def merge_json_csv(json_file_name, csv_file_name, output_file_name):
    try:
        json_encoding = detect_encoding(f'{json_file_name}.json')


        with open(f'{json_file_name}.json', 'r', encoding=json_encoding) as file:
            data = json.load(file)
        json_df = pd.DataFrame([extract_full_reviews(entry) for entry in data])
        csv_df = pd.read_csv(f'{csv_file_name}.csv')
        print(json_df.columns)  # Print column names of JSON DataFrame
        print(csv_df.columns)  # Print column names of CSV DataFrame

        # Print sample data from both DataFrames for inspection
        print("JSON DataFrame Sample:")
        print(json_df.head())
        print("CSV DataFrame Sample:")
        print(csv_df.head())

        # Standardize 'Address' field format
        json_df['Address'] = json_df['Address'].str.strip().str.lower()
        csv_df['Address'] = csv_df['Address'].str.strip().str.lower()

        # Print sample 'Address' data from both DataFrames for inspection
        print("JSON 'Address' Column Sample:")
        print(json_df['Address'].head())
        print("CSV 'Address' Column Sample:")
        print(csv_df['Address'].head())

        merged_data = pd.merge(json_df, csv_df, on='Address', how='inner')
        print(f"Number of rows in merged DataFrame: {len(merged_data)}")
        merged_data.to_json(f'{output_file_name}.json', orient='records', indent=4)


        print(f'Merged data saved to {output_file_name}.json successfully.')
    except Exception as e:
        print(f'An error occurred during merging: {e}')

def process_json_file(file_name):
    input_path = f'{file_name}.json'
    output_path = f'{file_name}_post.json'

    try:
        with open(input_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        cleaned_data = [extract_full_reviews(entry) for entry in data]

        # Save the processed data to a new JSON file
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(cleaned_data, file, ensure_ascii=False, indent=4)

        print(f'Processed and saved data to {output_path} successfully.')

    except FileNotFoundError as e:
        print(f'File {input_path} not found.')
    except json.JSONDecodeError as e:
        print(f'Error decoding JSON from file {input_path}.')
    except Exception as e:
        print(f'An error occurred: {e}')

if __name__ == '__main__':
    file_names = ['hu_hotel', 'hu_utils']
    csv_name = 'Filtered_Houston_Hotels'


    # Merge JSON and CSV for the first file
    merge_json_csv(file_names[0], csv_name, 'hu_hotel_merged')

    # Process and save the original JSON files and the merged file
    file_names = ['hu_hotel_merged', 'hu_utils']
    for file_name in file_names:
        process_json_file(file_name)
