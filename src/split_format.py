import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def splitting_and_scaling(data):

    # initialising x and y
    x = data.drop([  'Price', 'Kitchen type_0', 'Building condition_0'], axis=1)
    y = data['Price'] 

    #splitting data
    x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # normalisation or scaling - train and test
    minmax_scaler = MinMaxScaler()

    x_train[['Terrace surface', 'Surface of the plot', 'Living room surface']] = minmax_scaler.fit_transform(x_train[['Terrace surface', 'Surface of the plot', 'Living room surface']])
    x_test[['Terrace surface', 'Surface of the plot', 'Living room surface']]= minmax_scaler.transform(x_test[['Terrace surface', 'Surface of the plot', 'Living room surface']])

    return x_train,y_train,x_test,y_test

def run_split_format(data): # trail

    # getting dummies 
    trail_with_dummies = pd.get_dummies(data, columns=[ 'Type of property', 'Building condition', 'Kitchen type',  'province'], dtype='int')
    x_train,x_test, y_train, y_test = splitting_and_scaling(trail_with_dummies)
    return x_train,x_test, y_train, y_test

    
    