
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

def LinearReg(train_x,train_y, test_x,test_y):

    print('working in Linear Regression')
    # initialising 'linear regression' and creating a model
    linear = LinearRegression()
    linear.fit(train_x,train_y)

    # score of train
    train_score = linear.score(train_x,train_y)
    print("Train Score:", train_score)

    # score of test
    test_score = linear.score(test_x,test_y)
    print("Test Score:", test_score)

    print('\n')
    print('-----------------------------')
    print('\n')


def DecisionTreeReg(train_x,train_y, test_x,test_y):

    print('working in Decision Tree Regression')

    #max_depth=7, min_samples_leaf=4, min_samples_split=10
    DT = DecisionTreeRegressor(max_depth=10,  min_samples_leaf=10, min_samples_split=10)
    DT.fit(train_x,train_y)

    # score of train
    train_score = DT.score(train_x,train_y)
    print("Train Score:", train_score)

    # score of test
    test_score = DT.score(test_x,test_y)
    print("Test Score:", test_score)

    print('\n')
    print('-----------------------------')
    print('\n')

def NeighborsReg(train_x,train_y, test_x,test_y):

    print('working in Neighbors Regression')
    knn = KNeighborsRegressor(n_neighbors=3)
    knn.fit(train_x,train_y)

    # score of train
    train_score = knn.score(train_x,train_y)
    print("Train Score:", train_score)

    # score of test
    test_score = knn.score(test_x,test_y)
    print("Test Score:", test_score)

    print('\n')
    print('-----------------------------')
    print('\n')

def RandomForestReg(train_x,train_y, test_x,test_y):

    print('working in Random Forest Regressor')
   
    y_train = np.ravel(train_y)
    y_test = np.ravel(test_y)
    forest = RandomForestRegressor()
    forest.fit(train_x,train_y)

    # score of train
    train_score = forest.score(train_x,train_y)
    print("Train Score:", train_score)

    # score of test
    test_score = forest.score(test_x,test_y)
    print("Test Score:", test_score)

    print('\n')
    print('-----------------------------')
    print('\n')

def xgboostReg(train_x,train_y, test_x,test_y):

    print('working in xgboost')
    xgb_regressor = XGBRegressor()
    xgb_regressor.fit(train_x,train_y)

    Y_pred_train = xgb_regressor.predict(train_x)
    Y_pred_test = xgb_regressor.predict(test_x)

    train_mse = mean_squared_error(train_y, Y_pred_train)
    test_mse = mean_squared_error(test_y, Y_pred_test)

    # score of train
    train_score = xgb_regressor.score(train_x,train_y)
    print("Train Score:", train_score)

    # score of test
    test_score = xgb_regressor.score(test_x,test_y)
    print("Test Score:", test_score)

    print("Train MSE:", train_mse)
    print("Test MSE:", test_mse)
    
    print('\n')
    print('-----------------------------')
    print('\n')

def run_models(x_train,y_train,x_test,y_test):

    # running models from the start
    LinearReg(x_train,y_train,x_test,y_test)
    DecisionTreeReg(x_train,y_train,x_test,y_test)
    NeighborsReg(x_train,y_train,x_test,y_test)
    xgboostReg(x_train,y_train,x_test,y_test)
    RandomForestReg(x_train,y_train,x_test,y_test)