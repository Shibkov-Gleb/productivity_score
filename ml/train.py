import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from preprocessing import load_and_preprocess_data

import joblib

X_train, X_test, y_train, y_test, features = load_and_preprocess_data('data/Time Management and Productivity Insights.csv')

modelDTR = DecisionTreeRegressor()
modelDTR.fit(X_train, y_train)

joblib.dump(modelDTR, 'models/productivity_model.pkl')
joblib.dump(features, 'models/model_features.pkl')

print("Model and features successfully saved to the 'models' folder!")