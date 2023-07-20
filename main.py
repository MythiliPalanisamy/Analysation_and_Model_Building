from src.data_cleaning import run_data_cleaning
from src.split_format import run_split_format
from src.models import run_models

# getting regression values from main.py

path = 'data/final.csv'

def run(path):
    cleaned_data = run_data_cleaning(path)
    x_train,x_test, y_train, y_test = run_split_format(cleaned_data)
    run_models(x_train,x_test, y_train, y_test)

run(path)

