import json
import pickle
import os
import numpy as np

current_dir = os.path.dirname(__file__)
columns = os.path.join(current_dir, "artifacts", "columns.json")
model = os.path.join(current_dir, "artifacts","real_estate_price_prediction_bnglr.pickle")

print(columns)
print(model)

def get_estimated_price(location, sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >=0:
        x[loc_index] = 1
    return round(__model.predict([x])[0],2)

def get_location_names():
    return __locations

def load_saved_artifacts():
    print("loading saved artifacts.....start")
    global __data_columns
    global __locations
    global __model
    with open(columns,'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]
    
    with open(model, 'rb') as f:
        __model = pickle.load(f)
    
    print("Loading artifacts is complete")

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar', 1000,3,3))