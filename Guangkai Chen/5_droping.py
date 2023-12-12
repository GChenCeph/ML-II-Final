# Use this after incor.py
import json
import numpy as np
import pandas as pd


#####################################################################
def load_json_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
        print("before drop: ", len(data))
        filtered_data = [hotel for hotel in data if hotel.get('revenue', 0) >= threshold]
        print("after drop: ", len(filtered_data))
        return filtered_data
#####################################################################

threshold = 100
hotels_data = load_json_data('hu_hotel_weighted.json')
revenue = np.array([hotel['revenue'] for hotel in hotels_data])
log_revenue = np.log(revenue)

with open('hu_hotel_dropped.json', 'w', encoding='utf-8') as outfile:
    json.dump(hotels_data, outfile, indent=4)