import streamlit as st
import pandas as pd
import numpy as np
import joblib, pickle, time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="ShipmentSure AI",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
  --bg:#07090f; --surface:#0d1117; --surface2:#111827; --surface3:#1a2235;
  --border:rgba(255,255,255,0.06); --border2:rgba(255,255,255,0.10);
  --text:#f0f4ff; --muted:#6b7a99;
  --green:#22c55e; --red:#f43f5e; --amber:#f59e0b; --blue:#4f8ef7;
}
*, *::before, *::after { box-sizing:border-box; }
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    background: var(--bg) !important;
    font-family: 'Space Grotesk', sans-serif;
    color: var(--text);
}
[data-testid="block-container"] { padding:0 2.5rem 3rem !important; max-width:1400px !important; }
#MainMenu,footer,.stDeployButton {
    visibility:hidden !important;
    display:none !important;
} { visibility:hidden !important; display:none !important; }
[data-testid="collapsedControl"] {
    display:block !important;
    visibility:visible !important;
    color:white !important;
    background:#2563eb !important;
    border-radius:8px !important;
} { color:var(--text) !important; }
::-webkit-scrollbar { width:3px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--surface3); border-radius:2px; }

[data-testid="stSidebar"] { background:var(--surface) !important; border-right:1px solid var(--border) !important; width:220px !important; }
[data-testid="stSidebar"] > div { padding:0 !important; }
[data-testid="stSidebar"] * { color:var(--muted) !important; }
[data-testid="stRadio"] > label { display:none !important; }
[data-testid="stRadio"] > div { display:flex !important; flex-direction:column !important; gap:1px !important; background:transparent !important; }
[data-testid="stRadio"] > div > label {
    display:flex !important; align-items:center !important; padding:9px 20px !important;
    font-size:13px !important; font-weight:400 !important; color:var(--muted) !important;
    border-radius:0 !important; margin:0 !important; background:transparent !important;
    cursor:pointer !important; border-left:2px solid transparent !important; transition:all 0.15s !important;
}
[data-testid="stRadio"] > div > label:hover { color:var(--text) !important; background:rgba(79,142,247,0.05) !important; border-left-color:rgba(79,142,247,0.3) !important; }
div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child { display:none !important; }

[data-testid="stSelectbox"] > div > div { background:var(--surface2) !important; border:1px solid var(--border2) !important; border-radius:8px !important; color:var(--text) !important; }
[data-testid="stSelectbox"] > div > div:hover { border-color:var(--blue) !important; }
[data-testid="stNumberInput"] > div > div > input { background:var(--surface2) !important; border:1px solid var(--border2) !important; border-radius:8px !important; color:var(--text) !important; }
.stSlider > div > div > div { background:var(--surface3) !important; }
.stSlider > div > div > div > div { background:var(--blue) !important; }
label { color:var(--muted) !important; font-size:12px !important; font-weight:500 !important; }

.stButton > button[kind="primary"] {
    background:linear-gradient(135deg,#2563eb,#4f46e5) !important; color:white !important;
    border:none !important; border-radius:10px !important; padding:14px 0 !important;
    font-family:'Space Grotesk',sans-serif !important; font-size:15px !important; font-weight:600 !important;
    box-shadow:0 4px 24px rgba(79,142,247,0.25) !important; transition:all 0.2s !important;
}
.stButton > button[kind="primary"]:hover { transform:translateY(-1px) !important; box-shadow:0 8px 32px rgba(79,142,247,0.35) !important; }
.stButton > button:not([kind="primary"]) { background:var(--surface2) !important; color:var(--muted) !important; border:1px solid var(--border2) !important; border-radius:8px !important; }
[data-testid="stAlert"] { display:none !important; }

.stTabs [data-baseweb="tab-list"] { background:transparent !important; border-bottom:1px solid var(--border) !important; gap:0 !important; }
.stTabs [data-baseweb="tab"] { background:transparent !important; color:var(--muted) !important; font-size:13px !important; font-weight:500 !important; padding:10px 20px !important; border-radius:0 !important; border-bottom:2px solid transparent !important; font-family:'Space Grotesk',sans-serif !important; }
.stTabs [aria-selected="true"] { color:var(--blue) !important; border-bottom-color:var(--blue) !important; }

.card { background:var(--surface); border:1px solid var(--border); border-radius:14px; padding:20px 22px; }
.card:hover { border-color:var(--border2); }
.card-blue::before   { content:''; display:block; height:2px; background:linear-gradient(90deg,#3b82f6,#6366f1); margin:-20px -22px 16px; border-radius:14px 14px 0 0; }
.card-green::before  { content:''; display:block; height:2px; background:linear-gradient(90deg,#22c55e,#14b8a6); margin:-20px -22px 16px; border-radius:14px 14px 0 0; }
.card-red::before    { content:''; display:block; height:2px; background:linear-gradient(90deg,#f43f5e,#f97316); margin:-20px -22px 16px; border-radius:14px 14px 0 0; }
.card-amber::before  { content:''; display:block; height:2px; background:linear-gradient(90deg,#f59e0b,#ef4444); margin:-20px -22px 16px; border-radius:14px 14px 0 0; }
.card-purple::before { content:''; display:block; height:2px; background:linear-gradient(90deg,#8b5cf6,#ec4899); margin:-20px -22px 16px; border-radius:14px 14px 0 0; }
.card-teal::before   { content:''; display:block; height:2px; background:linear-gradient(90deg,#14b8a6,#3b82f6); margin:-20px -22px 16px; border-radius:14px 14px 0 0; }

.kpi-label { font-size:10px; font-weight:600; letter-spacing:1.4px; text-transform:uppercase; color:var(--muted); margin-bottom:10px; }
.kpi-val   { font-size:30px; font-weight:700; line-height:1; color:var(--text); margin-bottom:6px; }
.kpi-val sub { font-size:13px; font-weight:400; color:var(--muted); }
.delta-up   { font-size:11px; font-weight:500; color:var(--green); }
.delta-down { font-size:11px; font-weight:500; color:var(--red); }
.delta-flat { font-size:11px; font-weight:500; color:var(--muted); }
.eyebrow { font-size:9px; font-weight:700; letter-spacing:2.5px; text-transform:uppercase; color:var(--muted); display:block; margin-bottom:4px; }
.section-title { font-size:18px; font-weight:700; color:var(--text); letter-spacing:-0.4px; margin-bottom:16px; }

.badge-high   { display:inline-flex; align-items:center; gap:5px; padding:5px 11px; border-radius:20px; font-size:11px; font-weight:600; background:rgba(244,63,94,0.12); border:1px solid rgba(244,63,94,0.25); color:#fda4af; }
.badge-medium { display:inline-flex; align-items:center; gap:5px; padding:5px 11px; border-radius:20px; font-size:11px; font-weight:600; background:rgba(245,158,11,0.12); border:1px solid rgba(245,158,11,0.25); color:#fcd34d; }
.badge-low    { display:inline-flex; align-items:center; gap:5px; padding:5px 11px; border-radius:20px; font-size:11px; font-weight:600; background:rgba(34,197,94,0.12);  border:1px solid rgba(34,197,94,0.25);  color:#86efac; }
.badge-info   { display:inline-flex; align-items:center; gap:5px; padding:5px 11px; border-radius:20px; font-size:11px; font-weight:600; background:rgba(79,142,247,0.12); border:1px solid rgba(79,142,247,0.25); color:#93c5fd; }

.res-ontime  { background:linear-gradient(135deg,#011c12,#022c22); border:1px solid rgba(34,197,94,0.25); border-radius:16px; padding:36px 28px; text-align:center; }
.res-delayed { background:linear-gradient(135deg,#1a0510,#2d0a18); border:1px solid rgba(244,63,94,0.25); border-radius:16px; padding:36px 28px; text-align:center; }

@keyframes pg { 0%,100%{box-shadow:0 0 0 0 rgba(34,197,94,0.4)} 50%{box-shadow:0 0 0 8px rgba(34,197,94,0)} }
@keyframes pr { 0%,100%{box-shadow:0 0 0 0 rgba(244,63,94,0.4)} 50%{box-shadow:0 0 0 8px rgba(244,63,94,0)} }
.pg { animation:pg 2s infinite; }
.pr { animation:pr 2s infinite; }
</style>
""", unsafe_allow_html=True)

# ── Plotly base theme ──────────────────────────────────────────
B = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
         font=dict(family='Space Grotesk', color='#6b7a99', size=11),
         margin=dict(l=36,r=16,t=24,b=36))
G = dict(gridcolor='rgba(255,255,255,0.04)', linecolor='rgba(255,255,255,0.06)', zeroline=False)

# ── Load model artifacts ───────────────────────────────────────
@st.cache_resource
def load_artifacts():
    m  = joblib.load("model.pkl")
    sc = joblib.load("models/scaler.pkl")
    with open("models/feature_names.pkl","rb") as f: fn = pickle.load(f)
    with open("models/encoders.pkl","rb") as f:      en = pickle.load(f)
    return m, sc, fn, en

try:
    model, scaler, feature_names, encoders = load_artifacts()
    model_ok = True
except:
    model_ok = False
    model = scaler = feature_names = encoders = None

# ── Preprocessing — matches notebook exactly (19 features) ────
def predict(warehouse, mode, calls, rating, cost, prior, importance, gender, discount, weight):

    gender_enc = encoders['gender'].transform([gender])[0]

    data = {
        "Customer_care_calls": calls,
        "Customer_rating": rating,
        "Cost_of_the_Product": cost,
        "Prior_purchases": prior,
        "Discount_offered": discount,
        "Weight_in_gms": weight,

        "Weight_per_Cost": weight / max(cost, 1),
        "Discount_Value": (discount / 100) * cost,
        "High_Priority": 1 if importance == "high" else 0,
        "High_Rating": 1 if rating >= 4 else 0,

        "Gender_Encoded": gender_enc,

        "Warehouse_block_B": 1 if warehouse == "B" else 0,
        "Warehouse_block_C": 1 if warehouse == "C" else 0,
        "Warehouse_block_D": 1 if warehouse == "D" else 0,
        "Warehouse_block_F": 1 if warehouse == "F" else 0,

        "Mode_of_Shipment_Road": 1 if mode == "Road" else 0,
        "Mode_of_Shipment_Ship": 1 if mode == "Ship" else 0,

        "Product_importance_low": 1 if importance == "low" else 0,
        "Product_importance_medium": 1 if importance == "medium" else 0,
    }

    df = pd.DataFrame([data])

    df = df.reindex(columns=feature_names, fill_value=0)

    xs = scaler.transform(df)

    pred = model.predict(xs)[0]

    try:
        prob = model.predict_proba(xs)[0]

        on_p = round(float(prob[0]) * 100, 1)
        dl_p = round(float(prob[1]) * 100, 1)

    except:
        on_p = 50.0
        dl_p = 50.0

    return int(pred), on_p, dl_p

# ── Sidebar ────────────────────────────────────────────────────
st.sidebar.success("Sidebar Test")
with st.sidebar:
    st.markdown("""
    <div style="padding:24px 20px 18px;border-bottom:1px solid rgba(255,255,255,0.06);">
      <div style="display:flex;align-items:center;gap:10px;">
        <div style="width:32px;height:32px;background:linear-gradient(135deg,#2563eb,#4f46e5);
          border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:16px;">🚚</div>
        <div>
          <div style="font-size:15px;font-weight:700;color:#f0f4ff;">ShipmentSure</div>
          <div style="font-size:9px;color:#334155;text-transform:uppercase;letter-spacing:2px;">AI Platform</div>
        </div>
      </div>
    </div>
    <div style="padding:14px 20px 4px;font-size:9px;font-weight:700;color:#1e293b;text-transform:uppercase;letter-spacing:2px;">Navigation</div>
    """, unsafe_allow_html=True)

    page = st.radio(
    "Navigation",
    ["Dashboard", "Predict", "Analytics", "About"]
)
    st.write("Current Page =", page)

    st.markdown("<br>", unsafe_allow_html=True)
    if model_ok:
        st.markdown("""
        <div style="margin:0 12px;padding:12px;background:rgba(34,197,94,0.05);border:1px solid rgba(34,197,94,0.15);border-radius:10px;">
          <div style="display:flex;align-items:center;gap:6px;margin-bottom:8px;">
            <div style="width:6px;height:6px;border-radius:50%;background:#22c55e;"></div>
            <span style="font-size:10px;font-weight:700;color:#22c55e;letter-spacing:1px;">MODEL ONLINE</span>
          </div>
          <div style="font-size:10px;color:#334155;line-height:2.2;">
            ACCURACY &nbsp;<b style="color:#4f8ef7;">67.1%</b><br>
            F1-SCORE &nbsp;&nbsp;<b style="color:#4f8ef7;">72.7%</b><br>
            ROC-AUC &nbsp;&nbsp;<b style="color:#4f8ef7;">75.3%</b>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="margin:0 12px;padding:12px;background:rgba(244,63,94,0.05);border:1px solid rgba(244,63,94,0.2);border-radius:10px;">
          <div style="font-size:11px;color:#f43f5e;">⚠ model.pkl not found.<br>Need: model.pkl + models/ folder</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════
if page == "Dashboard":
    st.markdown("""
    <div style="padding:40px 0 28px;">
      <div style="display:inline-block;background:rgba(79,142,247,0.1);border:1px solid rgba(79,142,247,0.2);
        border-radius:20px;padding:4px 14px;margin-bottom:16px;">
        <span style="font-size:10px;font-weight:700;color:#4f8ef7;letter-spacing:2px;text-transform:uppercase;">AI Logistics Intelligence</span>
      </div>
      <h1 style="font-size:36px;font-weight:700;color:#f0f4ff;letter-spacing:-1px;line-height:1.15;margin:0 0 10px;">
        Predict Shipment Delays<br>
        <span style="background:linear-gradient(135deg,#4f8ef7,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
          Before They Happen
        </span>
      </h1>
      <p style="font-size:14px;color:#6b7a99;max-width:520px;font-weight:300;line-height:1.6;">
        Machine learning–powered supply chain analysis across 10,999 shipment records.
        Real-time risk assessment for smarter logistics decisions.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # KPI Cards
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    kpi_data = [
        (k1,"card card-blue",  "Total Shipments","10,999","","delta-flat","— Training set"),
        (k2,"card card-green", "On-Time Rate",   "59.7",  "%","delta-up",  "↑ 6,563 orders"),
        (k3,"card card-red",   "Delay Rate",     "40.3",  "%","delta-down","↓ 4,436 orders"),
        (k4,"card card-purple","Model Accuracy", "67.1",  "%","delta-up",  "↑ LightGBM"),
        (k5,"card card-amber", "ROC-AUC Score",  "75.3",  "%","delta-up",  "↑ Best metric"),
        (k6,"card card-teal",  "Features Used",  "19",    "","delta-flat", "— After encoding"),
    ]
    for col,cls,label,val,unit,dcls,delta in kpi_data:
        with col:
            st.markdown(f"""
            <div class="{cls}" style="min-height:108px;">
              <div class="kpi-label">{label}</div>
              <div class="kpi-val">{val}<sub>{unit}</sub></div>
              <div class="{dcls}">{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1
    r1c1, r1c2, r1c3 = st.columns([1.6,1,1])

    with r1c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<span class="eyebrow">Performance</span><div class="section-title" style="font-size:14px;margin-bottom:12px;">Monthly On-Time Rate</div>', unsafe_allow_html=True)
        months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        rates  = [100,100,100,68,43,43,46,43,45,43,42,43]
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Bar(x=months, y=[916]*12, name='Volume',
            marker_color='rgba(79,142,247,0.12)', marker_line_color='rgba(79,142,247,0.25)', marker_line_width=1,
            hovertemplate='%{x}: %{y:,} shipments<extra></extra>'), secondary_y=False)
        fig.add_trace(go.Scatter(x=months, y=rates, name='On-Time %',
            line=dict(color='#4f8ef7',width=2.5,shape='spline'), mode='lines+markers',
            marker=dict(size=6,color='#4f8ef7'), fill='tozeroy', fillcolor='rgba(79,142,247,0.04)',
            hovertemplate='%{x}: %{y}%<extra></extra>'), secondary_y=True)
        fig.update_layout(**B, height=230, showlegend=True,
            legend=dict(orientation='h',y=1.18,x=1,xanchor='right',bgcolor='rgba(0,0,0,0)',font=dict(size=11)))
        fig.update_xaxes(**G)
        fig.update_yaxes(secondary_y=False, showgrid=False, showticklabels=False)
        fig.update_yaxes(secondary_y=True, ticksuffix='%', range=[0,120], **G)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)

    with r1c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<span class="eyebrow">Shipment Mode</span><div class="section-title" style="font-size:14px;margin-bottom:12px;">Volume by Mode</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Pie(
            labels=['Ship','Flight','Road'], values=[7462,1777,1760], hole=0.62,
            textinfo='label+percent', textfont=dict(color='#6b7a99',size=11),
            marker_colors=['#4f8ef7','#a78bfa','#14b8a6'],
            marker_line=dict(color='#07090f',width=2),
            hovertemplate='%{label}: %{value:,}<extra></extra>'))
        fig.update_layout(**B, height=230, showlegend=False,
            annotations=[dict(text='10,999',x=0.5,y=0.5,font=dict(size=15,color='#f0f4ff',family='Space Grotesk'),showarrow=False)])
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)

    with r1c3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<span class="eyebrow">Warehouse</span><div class="section-title" style="font-size:14px;margin-bottom:12px;">On-Time % by Block</div>', unsafe_allow_html=True)
        wh_r = [58.6,60.2,59.7,59.8,59.8]
        fig = go.Figure(go.Bar(
            x=['A','B','C','D','F'], y=wh_r,
            marker_color=['#4f8ef7' if r==max(wh_r) else '#1a2235' for r in wh_r],
            marker_line_width=0, hovertemplate='Block %{x}: %{y:.1f}%<extra></extra>'))
        fig.update_layout(**B, height=230, showlegend=False, bargap=0.35)
        fig.update_xaxes(**G)
        fig.update_yaxes(range=[55,65], ticksuffix='%', **G)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 2
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<span class="eyebrow">Key Risk Driver</span><div class="section-title" style="font-size:14px;margin-bottom:12px;">Discount vs On-Time Delivery</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=['0–10%','11–20%','21–35%','36–50%','51–70%'],
            y=[46.9,100,100,100,100],
            marker_color=['#f43f5e','#22c55e','#22c55e','#22c55e','#22c55e'],
            marker_line_width=0, hovertemplate='Discount %{x}: %{y:.0f}% on-time<extra></extra>'))
        fig.add_hline(y=59.7, line_dash='dot', line_color='rgba(255,255,255,0.15)',
            annotation_text='avg 59.7%', annotation_font=dict(color='#6b7a99',size=10))
        fig.update_layout(**B, height=220, showlegend=False, bargap=0.25)
        fig.update_xaxes(**G)
        fig.update_yaxes(range=[0,115], ticksuffix='%', **G)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)

    with r2c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<span class="eyebrow">Customer Impact</span><div class="section-title" style="font-size:14px;margin-bottom:12px;">Care Calls → On-Time Rate</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Scatter(
            x=[2,3,4,5,6,7], y=[65.2,62.5,59.8,58.4,51.6,51.6],
            mode='lines+markers', line=dict(color='#a78bfa',width=2.5,shape='spline'),
            fill='tozeroy', fillcolor='rgba(167,139,250,0.05)',
            marker=dict(size=8,color='#a78bfa',line=dict(color='#7c3aed',width=2)),
            hovertemplate='%{x} calls: %{y:.1f}% on-time<extra></extra>'))
        fig.update_layout(**B, height=220, showlegend=False)
        fig.update_xaxes(title='Customer Care Calls', dtick=1, **G)
        fig.update_yaxes(range=[45,72], ticksuffix='%', **G)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 3 — extra insights
    r3c1, r3c2, r3c3 = st.columns(3)

    with r3c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<span class="eyebrow">Weight Impact</span><div class="section-title" style="font-size:14px;margin-bottom:12px;">Weight vs On-Time Rate</div>', unsafe_allow_html=True)
        wt_r = [69.1,100.0,63.8,42.2,42.9]
        fig = go.Figure(go.Bar(
            x=['<2.1k','2.1–3k','3–4.4k','4.4–5.6k','>5.6k'], y=wt_r,
            marker_color=['#22c55e' if r>=60 else '#f43f5e' for r in wt_r],
            marker_line_width=0, hovertemplate='%{x}: %{y:.1f}% on-time<extra></extra>'))
        fig.update_layout(**B, height=200, showlegend=False,)
        fig.update_xaxes(**G)
        fig.update_yaxes(range=[0,110], ticksuffix='%', **G)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)

    with r3c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<span class="eyebrow">Product Segment</span><div class="section-title" style="font-size:14px;margin-bottom:12px;">Importance vs On-Time</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Pie(
            labels=['Low','Medium','High'], values=[5297,4754,948], hole=0.6,
            marker_colors=['#1e3a5f','#3b82f6','#22c55e'],
            textinfo='label+percent', textfont=dict(color='#94a3b8',size=11),
            hovertemplate='%{label}: %{value:,}<extra></extra>'))
        fig.update_layout(
             **B,
            height=260,
            showlegend=False,
            bargap=0.3
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)

    with r3c3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<span class="eyebrow">Rating Impact</span><div class="section-title" style="font-size:14px;margin-bottom:12px;">Customer Rating Trend</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Scatter(
            x=[1,2,3,4,5], y=[57.1,58.3,59.9,60.8,62.1],
            mode='lines+markers', line=dict(color='#14b8a6',width=2.5),
            marker=dict(size=8,color='#14b8a6',line=dict(color='#0d9488',width=2)),
            fill='tozeroy', fillcolor='rgba(20,184,166,0.05)',
            hovertemplate='Rating %{x}: %{y:.1f}% on-time<extra></extra>'))
        fig.update_layout(**B, height=200, showlegend=False)
        fig.update_xaxes(title='Rating', dtick=1, **G)
        fig.update_yaxes(range=[50,70], ticksuffix='%', **G)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT
# ══════════════════════════════════════════════════════════════
elif page == "Predict":
    st.markdown("""
    <div style="padding:32px 0 20px;">
      <span class="eyebrow">Prediction Center</span>
      <div style="font-size:26px;font-weight:700;color:#f0f4ff;letter-spacing:-0.6px;">Shipment Risk Assessment</div>
      <p style="font-size:13px;color:#6b7a99;margin-top:6px;font-weight:300;">
        Fill in the order details and get an instant AI-powered delay prediction.
      </p>
    </div>
    """, unsafe_allow_html=True)

    if not model_ok:
        st.error("⚠ Model not loaded. Need: model.pkl + models/scaler.pkl + models/feature_names.pkl + models/encoders.pkl")
        st.stop()

    left, right = st.columns([1.1, 1], gap="large")

    with left:
        # Group 1
        st.markdown("""
        <div class="card" style="margin-bottom:12px;">
          <div style="font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
            color:#6b7a99;margin-bottom:14px;padding-bottom:10px;border-bottom:1px solid rgba(255,255,255,0.06);">
            🏭 &nbsp;Warehouse & Logistics
          </div>
        </div>""", unsafe_allow_html=True)

        with st.container():
            st.markdown('<div style="background:#0d1117;border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:18px 20px;margin-bottom:12px;">', unsafe_allow_html=True)
            st.markdown('<p style="font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#6b7a99;margin-bottom:14px;border-bottom:1px solid rgba(255,255,255,0.06);padding-bottom:10px;">🏭 Warehouse & Logistics</p>', unsafe_allow_html=True)
            wc1, wc2 = st.columns(2)
            with wc1:
                warehouse = st.selectbox("Warehouse Block", ["A","B","C","D","F"])
            with wc2:
                ship_mode = st.selectbox("Mode of Shipment", ["Flight","Ship","Road"])
            calls = st.slider("Customer Care Calls", 0, 10, 3)
            if calls >= 5:
                st.markdown('<div style="padding:8px 12px;background:rgba(244,63,94,0.08);border:1px solid rgba(244,63,94,0.2);border-radius:7px;font-size:11px;color:#fda4af;margin-top:4px;">⚠ High call volume — elevated delay risk</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Group 2
        with st.container():
            st.markdown('<div style="background:#0d1117;border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:18px 20px;margin-bottom:12px;">', unsafe_allow_html=True)
            st.markdown('<p style="font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#6b7a99;margin-bottom:14px;border-bottom:1px solid rgba(255,255,255,0.06);padding-bottom:10px;">📦 Product Details</p>', unsafe_allow_html=True)
            pc1, pc2 = st.columns(2)
            with pc1:
                cost       = st.number_input("Cost of Product ($)", min_value=0, value=300, step=10)
                importance = st.selectbox("Product Importance", ["low","medium","high"])
            with pc2:
                weight   = st.number_input("Weight (grams)", min_value=0, value=4000, step=100)
                discount = st.slider("Discount Offered (%)", 0, 70, 10)
            if discount > 50:
                st.markdown('<div style="padding:8px 12px;background:rgba(244,63,94,0.08);border:1px solid rgba(244,63,94,0.2);border-radius:7px;font-size:11px;color:#fda4af;margin-top:4px;">⚠ Discount >50% is the strongest delay predictor</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Group 3
        with st.container():
            st.markdown('<div style="background:#0d1117;border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:18px 20px;margin-bottom:20px;">', unsafe_allow_html=True)
            st.markdown('<p style="font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#6b7a99;margin-bottom:14px;border-bottom:1px solid rgba(255,255,255,0.06);padding-bottom:10px;">👤 Customer Profile</p>', unsafe_allow_html=True)
            cc1, cc2, cc3 = st.columns(3)
            with cc1: gender = st.selectbox("Gender", ["M","F"])
            with cc2: rating = st.slider("Rating", 1, 5, 3)
            with cc3: prior  = st.number_input("Prior Purchases", min_value=0, value=3, step=1)
            st.markdown('</div>', unsafe_allow_html=True)

        btn = st.button("⚡  Run AI Prediction", type="primary", use_container_width=True)

    # ── Result Panel ─────────────────────────────────────────
    with right:
        st.markdown("<br><br>", unsafe_allow_html=True)

        if btn:
            with st.spinner("Analysing shipment..."):
                time.sleep(0.5)
            pred, on_p, dl_p = predict(warehouse, ship_mode, calls, rating, cost, prior, importance, gender, discount, weight)
            is_on = pred == 1

            # ── Result card ──
            if is_on:
                st.markdown(f"""
                <div class="res-ontime">
                  <div style="display:flex;justify-content:center;margin-bottom:16px;">
                    <div style="width:56px;height:56px;border-radius:50%;background:rgba(34,197,94,0.15);
                      border:2px solid rgba(34,197,94,0.35);display:flex;align-items:center;
                      justify-content:center;font-size:24px;" class="pg">✓</div>
                  </div>
                  <div style="font-size:10px;font-weight:700;letter-spacing:3px;text-transform:uppercase;
                    color:#22c55e;margin-bottom:8px;">Prediction Result</div>
                  <div style="font-size:42px;font-weight:700;color:#86efac;line-height:1;margin-bottom:6px;">On Time</div>
                  <div style="font-size:13px;color:#6b7a99;margin-bottom:24px;">Shipment expected on schedule</div>
                  <div style="font-size:52px;font-weight:700;color:#22c55e;font-family:'JetBrains Mono',monospace;line-height:1;">
                    {on_p}<span style="font-size:20px;font-weight:400;color:#6b7a99;">%</span>
                  </div>
                  <div style="font-size:11px;color:#6b7a99;margin-top:4px;">on-time confidence</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="res-delayed">
                  <div style="display:flex;justify-content:center;margin-bottom:16px;">
                    <div style="width:56px;height:56px;border-radius:50%;background:rgba(244,63,94,0.15);
                      border:2px solid rgba(244,63,94,0.35);display:flex;align-items:center;
                      justify-content:center;font-size:24px;" class="pr">⚠</div>
                  </div>
                  <div style="font-size:10px;font-weight:700;letter-spacing:3px;text-transform:uppercase;
                    color:#f43f5e;margin-bottom:8px;">Prediction Result</div>
                  <div style="font-size:42px;font-weight:700;color:#fda4af;line-height:1;margin-bottom:6px;">Delayed</div>
                  <div style="font-size:13px;color:#6b7a99;margin-bottom:24px;">Shipment is at risk of delay</div>
                  <div style="font-size:52px;font-weight:700;color:#f43f5e;font-family:'JetBrains Mono',monospace;line-height:1;">
                    {dl_p}<span style="font-size:20px;font-weight:400;color:#6b7a99;">%</span>
                  </div>
                  <div style="font-size:11px;color:#6b7a99;margin-top:4px;">delay probability</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Confidence bars ──
            st.markdown(f"""
            <div class="card">
              <div class="kpi-label" style="margin-bottom:14px;">Confidence Breakdown</div>
              <div style="display:flex;justify-content:space-between;font-size:12px;font-weight:600;margin-bottom:5px;">
                <span style="color:#22c55e;">🟢 On Time</span>
                <span style="color:#22c55e;font-family:'JetBrains Mono',monospace;">{on_p}%</span>
              </div>
              <div style="background:rgba(255,255,255,0.05);border-radius:4px;height:8px;margin-bottom:12px;overflow:hidden;">
                <div style="width:{on_p}%;height:100%;background:linear-gradient(90deg,#22c55e,#14b8a6);border-radius:4px;"></div>
              </div>
              <div style="display:flex;justify-content:space-between;font-size:12px;font-weight:600;margin-bottom:5px;">
                <span style="color:#f43f5e;">🔴 Delayed</span>
                <span style="color:#f43f5e;font-family:'JetBrains Mono',monospace;">{dl_p}%</span>
              </div>
              <div style="background:rgba(255,255,255,0.05);border-radius:4px;height:8px;overflow:hidden;">
                <div style="width:{dl_p}%;height:100%;background:linear-gradient(90deg,#f43f5e,#f97316);border-radius:4px;"></div>
              </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Risk Assessment ──
            factors = []
            if discount > 50:      factors.append(("high",  "⚠ Discount >50%",       "Strongest delay predictor in model"))
            if calls >= 5:         factors.append(("high",  "⚠ High Care Calls",     f"{calls} calls — major risk signal"))
            if weight > 4400:      factors.append(("medium","◎ Heavy Shipment",       f"{weight}g — above safe threshold"))
            if rating <= 2:        factors.append(("medium","◎ Low Customer Rating", f"Rating {rating}/5 — concern flag"))
            if prior < 3:          factors.append(("medium","◎ New Customer",         f"Only {prior} prior orders"))
            if ship_mode == "Road":factors.append(("medium","◎ Road Shipment",        "Slower, higher delay variance"))
            if prior >= 5:         factors.append(("low",   "✓ Loyal Customer",      f"{prior} orders — reliable history"))
            if discount <= 10:     factors.append(("low",   "✓ Low Discount",         "Strong on-time signal"))
            if importance == "high":factors.append(("info", "✦ High Importance",      "Priority handling likely"))
            if rating >= 4:        factors.append(("low",   "✓ High Rating",          "Satisfied customer history"))
            if not factors:        factors.append(("low",   "✓ All Clear",            "No major risk factors detected"))

            st.markdown("""
            <div class="card">
              <div class="kpi-label" style="margin-bottom:14px;">⚠ Risk Factor Analysis</div>
            """, unsafe_allow_html=True)

            for level, title, desc in factors[:6]:
                badge = f"badge-{level}"
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:10px;padding:8px 0;
                  border-bottom:1px solid rgba(255,255,255,0.04);">
                  <span class="{badge}" style="flex-shrink:0;">{title}</span>
                  <span style="font-size:11px;color:#475569;">{desc}</span>
                </div>""", unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Order Summary ──
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="kpi-label" style="margin-bottom:12px;">Order Summary</div>', unsafe_allow_html=True)
            s1,s2,s3,s4,s5 = st.columns(5)
            for col, lbl, val in [
                (s1,"Mode",      ship_mode),
                (s2,"Cost",      f"${cost:,}"),
                (s3,"Weight",    f"{weight}g"),
                (s4,"Discount",  f"{discount}%"),
                (s5,"Importance",importance.title()),
            ]:
                with col:
                    st.markdown(f"""
                    <div style="text-align:center;padding:10px 6px;background:#080c14;
                      border-radius:8px;border:1px solid rgba(255,255,255,0.06);">
                      <div style="font-size:9px;color:#334155;text-transform:uppercase;
                        letter-spacing:1px;margin-bottom:4px;">{lbl}</div>
                      <div style="font-size:14px;font-weight:700;color:#e2e8f0;">{val}</div>
                    </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="card" style="text-align:center;padding:52px 28px;border-style:dashed;">
              <div style="font-size:44px;margin-bottom:16px;opacity:0.25;">🔮</div>
              <div style="font-size:15px;font-weight:600;color:#334155;margin-bottom:8px;">Awaiting Input</div>
              <div style="font-size:12px;color:#1e293b;line-height:1.8;">
                Fill in order details on the left<br>
                and click <b style="color:#4f8ef7;">Run AI Prediction</b><br><br>
                You'll see:<br>
                ✦ Prediction result<br>
                ✦ Confidence breakdown<br>
                ✦ Risk factor analysis<br>
                ✦ Order summary
              </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 3 — ANALYTICS
# ══════════════════════════════════════════════════════════════
elif page == "Analytics":
    st.markdown("""
    <div style="padding:32px 0 20px;">
      <span class="eyebrow">Data Intelligence</span>
      <div style="font-size:26px;font-weight:700;color:#f0f4ff;letter-spacing:-0.6px;">Analytics</div>
      <p style="font-size:13px;color:#6b7a99;margin-top:6px;font-weight:300;">
        Feature-level insights from 10,999 shipment records.
      </p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["  Feature Impact  ","  Model Performance  "])

    with tab1:
        ac1, ac2 = st.columns(2)
        with ac1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<span class="eyebrow">ML Insight</span><div class="section-title" style="font-size:14px;margin-bottom:12px;">Top Feature Importances</div>', unsafe_allow_html=True)
            feats = ['Discount_offered','Weight_in_gms','Cost_of_Product','Customer_calls','Prior_purchases','Customer_rating']
            imps  = [0.22,0.18,0.14,0.12,0.08,0.07]
            fig = go.Figure(go.Bar(
                x=imps, y=feats, orientation='h',
                marker_color=['#4f8ef7' if v==max(imps) else '#1a2235' for v in imps],
                marker_line_width=0, hovertemplate='%{y}: %{x:.3f}<extra></extra>'))
            fig.update_layout(**B, height=260, showlegend=False)
            fig.update_xaxes(title='Importance', **G)
            fig.update_yaxes(autorange='reversed', **G)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})
            st.markdown('</div>', unsafe_allow_html=True)

        with ac2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<span class="eyebrow">Weight Bands</span><div class="section-title" style="font-size:14px;margin-bottom:12px;">Weight vs On-Time Rate</div>', unsafe_allow_html=True)
            wt_r = [69.1,100.0,63.8,42.2,42.9]
            fig = go.Figure(go.Bar(
                x=['<2.1k','2.1–3.3k','3.3–4.4k','4.4–5.6k','>5.6k'], y=wt_r,
                marker_color=['#22c55e' if r>=60 else '#f43f5e' for r in wt_r],
                marker_line_width=0, 
                hovertemplate='%{x}: %{y:.1f}% on-time<extra></extra>'))
            fig.update_layout(**B, height=260, showlegend=False)
            fig.update_xaxes(**G)
            fig.update_yaxes(range=[0,110], ticksuffix='%', **G)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<span class="eyebrow">Product Segment</span><div class="section-title" style="font-size:14px;margin-bottom:12px;">On-Time Rate by Product Importance</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=['Low (5,297)','Medium (4,754)','High (948)'], y=[59.3,59.0,65.0],
            marker_color=['#1a2235','#1a2235','#4f8ef7'], marker_line_width=0, 
            hovertemplate='%{x}: %{y:.1f}%<extra></extra>'))
        fig.add_hline(y=59.7, line_dash='dot', line_color='rgba(255,255,255,0.15)')
        fig.update_layout(
    **B,
    height=200,
    showlegend=False,
    bargap=0.4
)      
        fig.update_xaxes(**G)
        fig.update_yaxes(range=[55,70], ticksuffix='%', **G)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<span class="eyebrow">Benchmarks</span><div class="section-title" style="font-size:14px;margin-bottom:12px;">Model Comparison — All Models</div>', unsafe_allow_html=True)
        m_names = ['Logistic Reg.','Decision Tree','Random Forest','KNN','🏆 LightGBM']
        fig = go.Figure()
        for vals, name, color in [
            ([65.7,63.1,66.8,63.5,67.1],'Accuracy','#4f8ef7'),
            ([70.4,71.2,61.0,64.9,72.7],'F1-Score','#22c55e'),
            ([71.9,64.1,73.5,70.1,75.3],'ROC-AUC','#f59e0b'),
        ]:
            fig.add_trace(go.Scatter(x=m_names, y=vals, name=name,
                mode='lines+markers', line=dict(color=color,width=2),
                marker=dict(size=9,color=color,line=dict(width=2,color=color)),
                hovertemplate=f'{name}: %{{y:.1f}}%<extra></extra>'))
        fig.update_layout(**B, height=280,
            legend=dict(orientation='h',y=1.18,bgcolor='rgba(0,0,0,0)',font=dict(color='#6b7a99')))
        fig.update_xaxes(**G)
        fig.update_yaxes(range=[58,80], ticksuffix='%', **G)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)

        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="kpi-label" style="margin-bottom:12px;">ML Pipeline Steps</div>', unsafe_allow_html=True)
            for step, desc in [
                ("📥 Load","10,999 records · Kaggle Supply Chain"),
                ("🔧 Encode","Gender LabelEnc + OHE → 19 features"),
                ("⚖ Scale","StandardScaler normalization"),
                ("🤖 LightGBM","Best model — F1: 72.7%"),
                ("🚀 Deploy","Streamlit web application"),
            ]:
                st.markdown(f'<div style="display:flex;gap:10px;padding:7px 0;border-bottom:1px solid rgba(255,255,255,0.04);"><span style="font-size:12px;min-width:100px;color:#f0f4ff;">{step}</span><span style="font-size:11px;color:#334155;">{desc}</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with mc2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="kpi-label" style="margin-bottom:12px;">Best Model — LightGBM Metrics</div>', unsafe_allow_html=True)
            for k,v,c in [
                ("Accuracy","67.1%","#4f8ef7"),
                ("F1-Score","72.7%","#22c55e"),
                ("ROC-AUC","75.3%","#f59e0b"),
                ("Precision","71.8%","#a78bfa"),
                ("Recall",  "73.6%","#14b8a6"),
            ]:
                st.markdown(f'<div style="display:flex;justify-content:space-between;align-items:center;padding:9px 0;border-bottom:1px solid rgba(255,255,255,0.04);"><span style="font-size:12px;color:#6b7a99;">{k}</span><span style="font-size:16px;font-weight:700;color:{c};">{v}</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ══════════════════════════════════════════════════════════════
elif page == "About":
    st.markdown("""
    <div style="padding:32px 0 20px;">
      <span class="eyebrow">Project Documentation</span>
      <div style="font-size:26px;font-weight:700;color:#f0f4ff;letter-spacing:-0.6px;">About ShipmentSure</div>
    </div>
    """, unsafe_allow_html=True)

    ab1, ab2 = st.columns(2, gap="large")

    with ab1:
        st.markdown("""
        <div class="card card-blue" style="margin-bottom:12px;">
          <div class="kpi-label" style="margin-bottom:10px;">Problem Statement</div>
          <p style="font-size:13px;color:#94a3b8;line-height:1.8;font-weight:300;">
            Late deliveries cost logistics companies millions annually and erode
            customer trust. <b style="color:#f0f4ff;">ShipmentSure</b> uses machine
            learning to predict delay risk before dispatch, enabling proactive
            supply chain intervention.
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="kpi-label" style="margin-bottom:12px;">Tech Stack</div>', unsafe_allow_html=True)
        for tech, detail in [
            ("Python 3.12",  "Core language"),
            ("LightGBM",     "Best model — F1: 72.7%"),
            ("scikit-learn", "Preprocessing pipeline"),
            ("Pandas/NumPy", "Data engineering"),
            ("Streamlit",    "Web deployment"),
            ("Plotly",       "Interactive charts"),
        ]:
            st.markdown(f'<div style="display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid rgba(255,255,255,0.04);"><span style="font-size:12px;font-weight:600;color:#f0f4ff;">{tech}</span><span style="font-size:11px;color:#334155;">{detail}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with ab2:
        st.markdown('<div class="card" style="margin-bottom:12px;">', unsafe_allow_html=True)
        st.markdown('<div class="kpi-label" style="margin-bottom:12px;">Dataset Overview</div>', unsafe_allow_html=True)
        for k,v in [
            ("Source",    "Kaggle Supply Chain Dataset"),
            ("Records",   "10,999"),
            ("Raw Feats", "11"),
            ("After OHE", "19 features"),
            ("Target",    "Reached.on.Time_Y.N"),
            ("Split",     "80% train / 20% test"),
            ("Best Model","LightGBM (tuned)"),
        ]:
            st.markdown(f'<div style="display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid rgba(255,255,255,0.04);"><span style="font-size:11px;color:#6b7a99;">{k}</span><span style="font-size:12px;font-weight:600;color:#f0f4ff;">{v}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="card card-purple">
          <div class="kpi-label" style="margin-bottom:10px;">Key Findings</div>
          <div style="font-size:12px;color:#94a3b8;line-height:2.4;">
            📌 <b style="color:#f0f4ff;">Discount 0–10%</b> has lowest on-time rate (46.9%)<br>
            📌 <b style="color:#f0f4ff;">≥5 Care Calls</b> strongly correlates with delay<br>
            📌 <b style="color:#f0f4ff;">Weight &gt;4.4kg</b> raises delay probability<br>
            📌 <b style="color:#f0f4ff;">LightGBM</b> outperformed 9 other models<br>
            📌 <b style="color:#f0f4ff;">High importance</b> products see better on-time rates
          </div>
        </div>
        """, unsafe_allow_html=True)
