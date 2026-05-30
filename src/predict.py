"""
predict.py
----------
Prediction pipeline for ShipmentSure.
Used by the Streamlit app to make real-time predictions.
"""

import joblib
import numpy as np
import pandas as pd


def load_artifacts():
    """Load saved model, scaler, and feature names."""
    model         = joblib.load("models/best_model.pkl")
    scaler        = joblib.load("models/scaler.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return model, scaler, feature_names


def predict_shipment(input_dict: dict) -> dict:
    """
    Given a dictionary of input features, return:
    - prediction (0 = Delayed, 1 = On Time)
    - probability of on-time delivery
    - confidence label
    """
    model, scaler, feature_names = load_artifacts()

    # Build dataframe from input
    df = pd.DataFrame([input_dict])

    # Feature engineering (same as training)
    df['cost_per_weight'] = df['Cost_of_the_Product'] / (df['Weight_in_gms'] + 1)
    df['discount_ratio']  = df['Discount_offered'] / (df['Cost_of_the_Product'] + 1)
    df['high_calls']      = (df['Customer_care_calls'] > 4).astype(int)
    df['loyal_customer']  = (df['Prior_purchases'] >= 4).astype(int)

    # Encode categoricals
    mode_map       = {'Ship': 0, 'Flight': 1, 'Road': 2}
    importance_map = {'Low': 0, 'Medium': 1, 'High': 2}
    gender_map     = {'F': 0, 'M': 1}
    warehouse_map  = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'F': 4}

    df['Mode_of_Shipment']   = df['Mode_of_Shipment'].map(mode_map)
    df['Product_importance'] = df['Product_importance'].map(importance_map)
    df['Gender']             = df['Gender'].map(gender_map)
    df['Warehouse_block']    = df['Warehouse_block'].map(warehouse_map)

    # Align feature order
    df = df[feature_names]

    # Scale
    X_scaled = scaler.transform(df)

    # Predict
    prediction   = model.predict(X_scaled)[0]
    probability  = model.predict_proba(X_scaled)[0][1]  # prob of on-time

    # Confidence label
    if probability >= 0.75:
        confidence = "High Confidence ✅"
    elif probability >= 0.50:
        confidence = "Moderate Confidence ⚠️"
    else:
        confidence = "Low Confidence ❌"

    return {
        "prediction":  int(prediction),
        "label":       "On Time 🟢" if prediction == 1 else "Delayed 🔴",
        "probability": round(float(probability) * 100, 2),
        "confidence":  confidence
    }
