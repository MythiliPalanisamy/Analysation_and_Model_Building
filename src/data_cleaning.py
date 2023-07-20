import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
    

def clean(trail):

    # removing outliers in 'price'
    trail = trail[trail['Price'] < 7.1e5]

    # cleaning 'postal code'
    trail = trail[trail['postal code'].str.len() <= 4] 

    # cleaning 'number of frontages'
    trail = trail.drop(trail[trail['Number of frontages']==14].index) # after romving,min = 6

    # cleaning 'toilets'
    trail = trail.drop(trail[trail['Toilets'] > 9].index)

    # since bathrooms and shower rooms are same adding it and removing >10 and (-1) values considering it as outliers 
    trail['Bathrooms']=trail['Bathrooms']+trail['Shower rooms'] 
    trail = trail.drop(trail[trail['Bathrooms'] > 10].index)
    trail = trail.drop(trail[trail['Bathrooms'] == -1].index)

    # replacing catagorical names in 'type of property'
    trail['Type of property'] = trail['Type of property'].replace('new-real-estate-project-apartments', 'apartment')
    trail['Type of property'] = trail['Type of property'].replace('new-real-estate-project-houses', 'house')
    trail['Type of property'] = trail['Type of property'].replace('apartment-block', 'apartment')

    return trail


def run_data_cleaning(path):
    trail = pd.read_csv(path) #'../data/final.csv'
    clean_trail = clean(trail)
    final_trail = clean_trail.drop(columns=['Energy class', 'Primary energy consumption','Heating type','Address', 'Location',  'immo code','Construction year','Terrace','Shower rooms', 'Office' ,'postal code'], axis=1)
    return final_trail
