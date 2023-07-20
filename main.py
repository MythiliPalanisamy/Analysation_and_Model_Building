"""import sys
import os
import pandas as pd
"""
# the absolute path of the "parts" directory
#parts_dir = os.path.abspath(r"C:\Users\Karthick Palanivel\Documents\GitHub\analysing-data\src\model building\parts.py")

# Adding the "parts" directory to sys.path
#sys.path.append(parts_dir)

from src.data_cleaning import run_data_cleaning
from src.split_format import run_split_format
from src.models import run_models

path = 'data/final.csv'

def run(path):
    cleaned_data = run_data_cleaning(path)
    x_train,x_test, y_train, y_test = run_split_format(cleaned_data)
    run_models(x_train,x_test, y_train, y_test)

run(path)

