"""
preprocess.py
-------------
Data cleaning, feature engineering, and preprocessing pipeline
for ShipmentSure project.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV dataset."""
    df = pd.read_csv(filepath)
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
# 2. Basic Inspection
# ─────────────────────────────────────────────
def inspect_data(df: pd.DataFrame):
    """Print basic info, null counts, and class distribution."""
    print("\n📋 DATASET INFO")
    print("-" * 40)
    print(df.info())
    print("\n📊 NULL VALUES:")
    print(df.isnull().sum())
    print("\n🎯 TARGET DISTRIBUTION:")
    print(df['Reached.on.Time_Y.N'].value_counts())
    print(df['Reached.on.Time_Y.N'].value_counts(normalize=True).mul(100).round(2))


# ─────────────────────────────────────────────
# 3. Feature Engineering
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new meaningful features from existing ones."""
    df = df.copy()

    # Cost-to-weight ratio: higher ratio may indicate special/fragile items
    df['cost_per_weight'] = df['Cost_of_the_Product'] / (df['Weight_in_gms'] + 1)

    # Discount impact: how much discount was given relative to cost
    df['discount_ratio'] = df['Discount_offered'] / (df['Cost_of_the_Product'] + 1)

    # Customer engagement score: more calls = more issues?
    df['high_calls'] = (df['Customer_care_calls'] > df['Customer_care_calls'].median()).astype(int)

    # Prior purchases reliability
    df['loyal_customer'] = (df['Prior_purchases'] >= 4).astype(int)

    print("✅ Feature engineering complete. New features added: cost_per_weight, discount_ratio, high_calls, loyal_customer")
    return df


# ─────────────────────────────────────────────
# 4. Encode Categorical Variables
# ─────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label encode categorical columns."""
    df = df.copy()
    categorical_cols = ['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']
    
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
            print(f"  ✅ Encoded: {col}")
    
    return df


# ─────────────────────────────────────────────
# 5. Scale Numerical Features
# ─────────────────────────────────────────────
def scale_features(X_train, X_test):
    """Normalize numerical features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    
    # Save scaler for deployment
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    print("✅ Scaler saved to models/scaler.pkl")
    
    return X_train_scaled, X_test_scaled, scaler


# ─────────────────────────────────────────────
# 6. Full Pipeline
# ─────────────────────────────────────────────
def run_preprocessing(filepath: str):
    """
    Full preprocessing pipeline.
    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    # Load
    df = load_data(filepath)
    inspect_data(df)

    # Drop ID column if exists
    if 'ID' in df.columns:
        df.drop(columns=['ID'], inplace=True)

    # Feature engineering
    df = engineer_features(df)

    # Encode categoricals
    df = encode_categoricals(df)

    # Handle missing values
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Separate features and target
    TARGET = 'Reached.on.Time_Y.N'
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    feature_names = X.columns.tolist()

    # Train-test split (80-20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n✅ Train size: {X_train.shape}, Test size: {X_test.shape}")

    # Scale
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Save processed feature names
    joblib.dump(feature_names, "models/feature_names.pkl")
    print("✅ Feature names saved.")

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names


if __name__ == "__main__":
    run_preprocessing("data/Train.csv")
