"""
app.py — ShipmentSure Streamlit App (Fixed)
Matches exact feature engineering from training notebook.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="ShipmentSure",
    page_icon="🚚",
    layout="wide"
)

st.markdown("""
<style>
    .main-title { font-size:2.5rem; font-weight:700; color:#1f4e79; text-align:center; }
    .subtitle   { font-size:1.1rem; color:#555; text-align:center; margin-bottom:1.5rem; }
    .result-green { background:#d4edda; border:2px solid #28a745; border-radius:12px;
                    padding:1.5rem; text-align:center; font-size:1.5rem; font-weight:bold; }
    .result-red   { background:#f8d7da; border:2px solid #dc3545; border-radius:12px;
                    padding:1.5rem; text-align:center; font-size:1.5rem; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

# ── Load Artifacts ────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load("model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    with open("models/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    return model, scaler, feature_names

# ── Preprocess Input (matches notebook exactly) ──────────────
def preprocess_input(raw: dict, feature_names: list) -> pd.DataFrame:
    df = pd.DataFrame([raw])

    # Label encode Gender
    df['Gender_Encoded'] = 1 if raw['Gender'] == 'M' else 0
    df.drop(columns=['Gender'], inplace=True)

    # One-hot encode Warehouse_block, Mode_of_Shipment, Product_importance
    df = pd.get_dummies(df, columns=['Warehouse_block', 'Mode_of_Shipment', 'Product_importance'], drop_first=True)

    # Add any missing columns (that weren't generated due to input value)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    # Drop ID if present
    if 'ID' in df.columns:
        df.drop(columns=['ID'], inplace=True)

    # Keep only the features in correct order
    df = df[feature_names]
    return df

# ── UI ────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🚚 ShipmentSure</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Shipment Delay Prediction</p>', unsafe_allow_html=True)
st.divider()

try:
    model, scaler, feature_names = load_artifacts()
    st.success(f"✅ Model loaded | Features: {len(feature_names)}")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.info("Make sure model.pkl, models/scaler.pkl and models/feature_names.pkl are in the project folder.")
    st.stop()

# ── Input Form ────────────────────────────────────────────────
st.subheader("📝 Enter Shipment Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🏭 Warehouse & Shipping**")
    warehouse_block   = st.selectbox("Warehouse Block", ['A', 'B', 'C', 'D', 'F'])
    mode_of_shipment  = st.selectbox("Mode of Shipment", ['Flight', 'Ship', 'Road'])
    customer_care_calls = st.slider("Customer Care Calls", 0, 10, 3)
    customer_rating   = st.slider("Customer Rating (1–5)", 1, 5, 3)

with col2:
    st.markdown("**📦 Product Details**")
    cost_of_product    = st.number_input("Cost of Product ($)", 100, 10000, 300)
    weight_in_gms      = st.number_input("Weight (grams)", 1000, 10000, 4000)
    product_importance = st.selectbox("Product Importance", ['low', 'medium', 'high'])
    discount_offered   = st.slider("Discount Offered (%)", 0, 70, 10)

with col3:
    st.markdown("**👤 Customer Details**")
    gender          = st.selectbox("Gender", ['M', 'F'])
    prior_purchases = st.slider("Prior Purchases", 1, 10, 3)

st.divider()

# ── Predict ───────────────────────────────────────────────────
if st.button("🔮 Predict Delivery Status", use_container_width=True, type="primary"):
    raw_input = {
        'Warehouse_block':     warehouse_block,
        'Mode_of_Shipment':    mode_of_shipment,
        'Customer_care_calls': customer_care_calls,
        'Customer_rating':     customer_rating,
        'Cost_of_the_Product': cost_of_product,
        'Prior_purchases':     prior_purchases,
        'Product_importance':  product_importance,
        'Gender':              gender,
        'Discount_offered':    discount_offered,
        'Weight_in_gms':       weight_in_gms
    }

    try:
        input_df  = preprocess_input(raw_input, feature_names)
        X_scaled  = scaler.transform(input_df)
        prediction  = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]

        st.subheader("🎯 Prediction Result")
        _, mid, _ = st.columns([1, 2, 1])

        with mid:
            if prediction == 0:
                prob_pct = round(probability[0] * 100, 1)
                st.markdown(f'<div class="result-green">🟢 ON TIME<br><small>Confidence: {prob_pct}%</small></div>', unsafe_allow_html=True)
            else:
                prob_pct = round(probability[1] * 100, 1)
                st.markdown(f'<div class="result-red">🔴 DELAYED<br><small>Confidence: {prob_pct}%</small></div>', unsafe_allow_html=True)

        # Show input summary
        st.divider()
        st.subheader("📋 Input Summary")
        st.dataframe(pd.DataFrame([raw_input]), use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write("Feature names expected by model:", feature_names)
