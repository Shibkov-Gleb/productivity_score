import os
import joblib
import pandas as pd

class ProductivityPredictor:

    def __init__(self, model_dir="../models"):
        model_path = os.path.join(model_dir, "productivity_model.pkl")
        features_path = os.path.join(model_dir, "model_features.pkl")
        scaler_path = os.path.join(model_dir, "scaler.pkl")

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError("Model or scaler files not found! Run train.py first.")

        self.model = joblib.load(model_path)
        self.feature_names = joblib.load(features_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, user_inputs: dict) -> float:
        df_input = pd.DataFrame([user_inputs])

        df_input = df_input[self.feature_names]

        scaled_input = self.scaler.transform(df_input) 

        prediction = self.model.predict(scaled_input)[0]

        return float(max(0, min(100, prediction)))