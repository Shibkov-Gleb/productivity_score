import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

def load_and_preprocess_data(filepath, test_size=0.2, random_state=42):
    df = pd.read_csv(filepath)
    
    X = df.drop(columns=['Productivity Score', 'User ID'])
    Y = df['Productivity Score']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size, random_state)
    
    mmscaler=MinMaxScaler(feature_range=(0,1))

    x_train=mmscaler.fit_transform(x_train)
    x_test=mmscaler.fit_transform(x_test)
    x_train=pd.DataFrame(x_train)
    x_test=pd.DataFrame(x_test)
    
    joblib.dump(mmscaler, '../models/scaler.pkl')
    
    return X_train, X_test, y_train, y_test, list(X.columns)