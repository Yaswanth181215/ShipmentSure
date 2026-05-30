"""
app.py — ShipmentSure Streamlit App (Fixed)
Matches exact feature engineering from training notebook.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.graph_objects as go

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="ShipmentSure",
    page_icon="🚚",
    layout="wide"
)

st.markdown("""
<style>

.stApp{
background:linear-gradient(135deg,#f8fafc,#dbeafe);
}

.main-title{
font-size:3rem;
font-weight:800;
text-align:center;
color:#1e3a8a;
}

.subtitle{
font-size:1.2rem;
text-align:center;
color:#64748b;
margin-bottom:20px;
}

.metric-card{
background:white;
padding:15px;
border-radius:15px;
box-shadow:0 4px 15px rgba(0,0,0,0.08);
text-align:center;
}

.result-green{
background:#dcfce7;
border:2px solid #22c55e;
border-radius:15px;
padding:25px;
text-align:center;
font-size:1.5rem;
font-weight:bold;
}

.result-red{
background:#fee2e2;
border:2px solid #ef4444;
border-radius:15px;
padding:25px;
text-align:center;
font-size:1.5rem;
font-weight:bold;
}

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
st.markdown("""
<div style="
padding:30px;
border-radius:20px;
background:linear-gradient(135deg,#2563eb,#4f46e5);
color:white;
text-align:center;
margin-bottom:20px;
">
<h1>🚚 ShipmentSure AI</h1>
<h3>Intelligent Shipment Delay Prediction Platform</h3>
<p>Predict delivery risks before they happen</p>
</div>
""", unsafe_allow_html=True)
st.divider()

try:
    model, scaler, feature_names = load_artifacts()
    st.success(f"✅ Model loaded | Features: {len(feature_names)}")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("📦 Shipments", "10,999")
    c2.metric("🧠 Features", len(feature_names))
    c3.metric("🤖 Model", "Online")
    c4.metric("🚀 Status", "Live")

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

                st.markdown(
                    f'<div class="result-green">🟢 ON TIME<br><small>Confidence: {prob_pct}%</small></div>',
                    unsafe_allow_html=True
                )

            else:
                prob_pct = round(probability[1] * 100, 1)

                st.markdown(
                    f'<div class="result-red">🔴 DELAYED<br><small>Confidence: {prob_pct}%</small></div>',
                    unsafe_allow_html=True
                )

# ── Confidence Gauge ─────────────────────────────

                fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob_pct,
                title={"text": "Prediction Confidence"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#2563eb"},
                    "steps": [
                        {"range": [0, 40], "color": "#ef4444"},
                        {"range": [40, 70], "color": "#f59e0b"},
                        {"range": [70, 100], "color": "#22c55e"}
                    ]
                }
            )
        )

        #fig.update_layout(height=300)

        #st.plotly_chart(fig, use_container_width=True)
        c1, c2 = st.columns(2)

        with c1:
            st.metric(
        label="🟢 On-Time Probability",
        value=f"{round(probability[0] * 100, 1)}%"
    )

        with c2:
            st.metric(
        label="🔴 Delay Probability",
        value=f"{round(probability[1] * 100, 1)}%"
    )
        if prob_pct >= 80:
            st.error("🔴 Risk Level: HIGH")
        elif prob_pct >= 60:
                st.warning("🟠 Risk Level: MEDIUM")
        else:
            st.success("🟢 Risk Level: LOW")
        # 🤖 AI Recommendation
        st.subheader("🤖 AI Recommendation")

        if prediction == 1:
            st.warning("""
High delay risk detected.

Recommended Actions:
• Prioritize shipment processing
• Monitor warehouse operations
• Improve customer communication
• Reduce logistics bottlenecks
""")
        else:
            st.success("""
Shipment is likely to arrive on time.

Recommended Actions:
• Continue current logistics strategy
• Maintain standard monitoring
• No intervention required
""")

# Show input summary
        st.divider()
        st.subheader("📋 Input Summary")
        st.dataframe(pd.DataFrame([raw_input]), use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write("Feature names expected by model:", feature_names)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────

    st.markdown("---")

    st.markdown("""
<div style="
text-align:center;
padding:20px;
margin-top:20px;
color:#64748b;
font-size:14px;
">

<h3>🚚 ShipmentSure AI</h3>

<p>
AI-Powered Shipment Delay Prediction Platform
</p>

<p>
Built with Python • Machine Learning • Streamlit • Plotly
</p>

<p>
Developed by <b>Yaswanth Venkata Pavan</b>
</p>

<p>
<a href="https://github.com/Yaswanth181215/ShipmentSure" target="_blank">
🔗 View GitHub Repository
</a>
</p>

</div>
""", unsafe_allow_html=True)
