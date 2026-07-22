from sklearn.tree import DecisionTreeRegressor
from preprocessing import load_and_preprocess_data

import joblib

X_train, X_test, y_train, y_test, features = load_and_preprocess_data('data/Time Management and Productivity Insights.csv')

modelDTR = DecisionTreeRegressor()
modelDTR.fit(X_train, y_train)

joblib.dump(modelDTR, 'models/productivity_model.pkl')
joblib.dump(features, 'models/model_features.pkl')

print("Model and features successfully saved to the 'models' folder")