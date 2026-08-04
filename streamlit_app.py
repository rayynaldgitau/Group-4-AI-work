import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Smart Home Energy Predictor", layout="centered", page_icon="⚡")

# ---------- Claude-inspired theme ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:wght@500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
    .stApp { background-color: #F5F4ED; }

    h1 {
        font-family: 'Source Serif 4', Georgia, serif !important;
        color: #1F1E1C !important;
        font-weight: 600 !important;
        font-size: 2.1rem !important;
    }
    h2, h3 { font-family: 'Source Serif 4', Georgia, serif !important; color: #1F1E1C !important; }

    .stCaption, [data-testid="stCaptionContainer"] { color: #85807A !important; }

    .stAlert, div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E8E4DA !important;
        border-radius: 12px !important;
    }
    div[data-testid="stMetric"] {
        padding: 1.2rem 1.4rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetricValue"] {
        color: #D97757 !important;
        font-family: 'Source Serif 4', Georgia, serif !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetricLabel"] { color: #85807A !important; }

    .stSlider [data-baseweb="slider"] > div > div:nth-child(2) { background-color: #D97757 !important; }
    .stSlider [role="slider"] { background-color: #D97757 !important; border-color: #D97757 !important; }
    .stCheckbox [data-baseweb="checkbox"] svg { fill: #D97757 !important; }

    hr { border-color: #E8E4DA !important; }
    p, li, label, .stMarkdown { color: #3D3A35; }

    .block-container { padding-top: 2.5rem; max-width: 780px; }

    .model-badge {
        display: inline-block;
        background-color: #EFE9DD;
        color: #6B5B3E;
        font-size: 0.78rem;
        font-weight: 500;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        margin-bottom: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Only the best-performing model is loaded and used. Across every evaluation run
# in this project (baseline, GridSearchCV-tuned Random Forest, Ridge, Lasso, KNN,
# Extra Trees, Decision Tree), Linear Regression had the lowest test RMSE and the
# only positive R² on the chronological (non-shuffled) test split. No model
# picker is shown, this app always predicts with that model.
lr = joblib.load("lr_model.pkl")
feature_cols = joblib.load("feature_cols.pkl")
defaults = joblib.load("feature_defaults.pkl")

st.title("Energy Usage Predictor")
st.markdown('<span class="model-badge">Model: Linear Regression · best of 7 tested · test R² 0.099</span>', unsafe_allow_html=True)
st.caption("APT3010A Semester Project · trained on the UCI Appliances Energy Prediction dataset")
st.write("Adjust the inputs below to see predicted appliance energy use.")

st.write("")
col1, col2 = st.columns(2)
with col1:
    hour = st.slider("Hour of day", 0, 23, 18)
    is_weekend = st.checkbox("Weekend", value=False)
    lights = st.slider("Lights circuit (Wh)", 0, 70, int(defaults.get("lights", 0)))
    t_out = st.slider("Outdoor temperature (°C)", -5.0, 30.0, float(round(defaults["T_out"], 1)))
    rh_out = st.slider("Outdoor humidity (%)", 20.0, 100.0, float(round(defaults["RH_out"], 1)))
with col2:
    t3 = st.slider("Kitchen area temperature T3 (°C)", 15.0, 30.0, float(round(defaults["T3"], 1)))
    rh3 = st.slider("Kitchen area humidity RH_3 (%)", 20.0, 70.0, float(round(defaults["RH_3"], 1)))
    pressure = st.slider("Pressure (mm Hg)", 720.0, 775.0, float(round(defaults["Press_mm_hg"], 1)))

row = defaults.copy()
row["hour"] = hour
row["is_weekend"] = int(is_weekend)
row["lights"] = lights
row["T_out"] = t_out
row["RH_out"] = rh_out
row["T3"] = t3
row["RH_3"] = rh3
row["Press_mm_hg"] = pressure
for c in feature_cols:
    if c.startswith("season_"):
        row[c] = 0
row["season_winter"] = 1

X = pd.DataFrame([{c: row.get(c, 0) for c in feature_cols}])
pred = lr.predict(X)[0]

st.write("")
st.metric("Predicted Appliance Energy Use", f"{pred:.0f} Wh")

st.divider()
st.subheader("Energy-saving tips")
st.markdown("""
- **Peak-hour shifting** — usage peaks 5-7pm and 11am-1pm, run non-essential appliances off-peak.
- **Kitchen zone** — temp/humidity is the strongest single-room predictor after time of day.
- **Lights circuit** — load tracks closely with occupancy, turning off unused lights measurably
  correlates with lower total appliance draw.
- **Weather sealing** — outdoor humidity and pressure matter, better insulation reduces weather-driven load.
""")
st.caption("Chronological 80/20 split, not shuffled · Linear Regression selected over Random Forest, "
           "Ridge, Lasso, KNN, Extra Trees, and Decision Tree on test-set performance.")
