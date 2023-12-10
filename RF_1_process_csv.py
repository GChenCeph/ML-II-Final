import pandas as pd
import chardet

# List of column names to be used when reading the CSV file
column_names = [
    "Taxpayer Number", "Taxpayer Name", "Taxpayer Address", "Taxpayer City",
    "Taxpayer State", "Taxpayer Zip", "Taxpayer County", "Taxpayer Phone",
    "Location Number", "Address", "Location City", "Location State",
    "Location Zip", "Location County", "Location Phone", "Unit Capacity",
    "Responsibility Begin Date (YYYYMMDD)", "Responsibility End Date (YYYYMMDD)",
    "Obligation End Date (YYYYMMDD)", "Filer Type", "Total Room Receipts", "Taxable Receipts"
]


def detect_encoding(file_path):
    # Detect the file encoding
    with open(file_path, 'rb') as file:
        # Read a large chunk of the file to accurately guess the encoding
        raw_data = file.read(10000000)
        return chardet.detect(raw_data)['encoding']


def filter_houston_hotels(csv_path, output_path):
    # Detect the encoding of the CSV file
    encoding = detect_encoding(csv_path)

    # Read the CSV file with the detected encoding and predefined column names
    df = pd.read_csv(csv_path, encoding='ISO-8859-1', on_bad_lines='skip', header=None, names=column_names,
                     low_memory=False)

    # Filter the DataFrame to include only hotels located in Houston
    houston_hotels = df[df['Location City'].str.strip().str.lower() == 'houston']

    # Drop columns that are not necessary for the analysis
    columns_to_drop = ['Taxpayer Number', 'Taxpayer State', 'Taxpayer Zip', 'Taxpayer County', 'Taxpayer Phone',
                       'Taxpayer Name', 'Taxpayer City', 'Taxpayer Address', 'Location State', 'Location Phone',
                       'Responsibility Begin Date (YYYYMMDD)', 'Responsibility End Date (YYYYMMDD)',
                       'Obligation End Date (YYYYMMDD)']
    houston_hotels = houston_hotels.drop(columns_to_drop, axis=1)

    # Group the data by Address and Unit Capacity, and sum the Total Room Receipts
    grouped_df = houston_hotels.groupby(['Address', 'Unit Capacity']).agg({'Total Room Receipts': 'sum'}).reset_index()

    # Save the filtered and grouped data to a new CSV file
    grouped_df.to_csv(output_path, index=False)


if __name__ == '__main__':
    # Define the paths for the input and output files
    csv_file_path = 'C:/Users/ompat/Documents/School/MEng/Fall_23/ML2/Project-Combine/HOTEL2022 (2).csv'
    output_file_path = 'C:/Users/ompat/Documents/School/MEng/Fall_23/ML2/Project-Combine/Filtered_Houston_Hotels.csv'

    # Call the function to filter and process Houston hotels data
    filter_houston_hotels(csv_file_path, output_file_path)
