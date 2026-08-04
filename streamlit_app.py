import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Smart Home Energy Predictor", layout="centered", page_icon="⚡")

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

    div[data-testid="stNumberInput"] input {
        border-color: #E8E4DA !important;
        border-radius: 8px !important;
    }
    div[data-testid="stNumberInput"] input:focus {
        border-color: #D97757 !important;
        box-shadow: 0 0 0 1px #D97757 !important;
    }
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
    .tip-hit {
        background-color: #FBF0DC;
        border-left: 3px solid #C99A2E;
        padding: 0.6rem 0.9rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        font-size: 0.92rem;
    }
    .tip-ok {
        background-color: #F0F5EE;
        border-left: 3px solid #3A7D5C;
        padding: 0.6rem 0.9rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)

# Only the best-performing model is loaded and used. Across every evaluation run
# in this project, Linear Regression had the lowest test RMSE and the only
# positive R² on the chronological (non-shuffled) test split.
lr = joblib.load("lr_model.pkl")
feature_cols = joblib.load("feature_cols.pkl")
defaults = joblib.load("feature_defaults.pkl")

st.title("Energy Usage Predictor")
st.markdown('<span class="model-badge">Model: Linear Regression · best of 7 tested · test R² 0.099</span>', unsafe_allow_html=True)
st.caption("APT3010A Semester Project · trained on the UCI Appliances Energy Prediction dataset")
st.write("Enter exact readings below to see predicted appliance energy use.")

st.write("")
col1, col2 = st.columns(2)
with col1:
    hour = st.number_input("Hour of day (0-23)", min_value=0, max_value=23, value=18, step=1)
    is_weekend = st.checkbox("Weekend", value=False)
    lights = st.number_input("Lights circuit (Wh)", min_value=0, max_value=200, value=int(defaults.get("lights", 0)), step=1)
    t_out = st.number_input("Outdoor temperature (°C)", min_value=-10.0, max_value=40.0, value=float(round(defaults["T_out"], 1)), step=0.1, format="%.1f")
    rh_out = st.number_input("Outdoor humidity (%)", min_value=0.0, max_value=100.0, value=float(round(defaults["RH_out"], 1)), step=0.1, format="%.1f")
with col2:
    t3 = st.number_input("Kitchen area temperature T3 (°C)", min_value=0.0, max_value=40.0, value=float(round(defaults["T3"], 1)), step=0.1, format="%.1f")
    rh3 = st.number_input("Kitchen area humidity RH_3 (%)", min_value=0.0, max_value=100.0, value=float(round(defaults["RH_3"], 1)), step=0.1, format="%.1f")
    pressure = st.number_input("Pressure (mm Hg)", min_value=700.0, max_value=800.0, value=float(round(defaults["Press_mm_hg"], 1)), step=0.1, format="%.1f")

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
st.subheader("Energy-saving tips for these inputs")

# These now respond to the actual values entered above, not static text.
peak_hours = set(list(range(11, 14)) + list(range(17, 20)))
tips = []

if hour in peak_hours:
    tips.append(("hit", f"Hour {hour} falls in a peak-usage window (11am-1pm or 5-7pm). "
                         "If this appliance use is flexible, shifting it to 11pm-5am would target a lower-demand period."))
else:
    tips.append(("ok", f"Hour {hour} is outside the peak-usage window, good timing, no shift needed."))

if lights > 30:
    tips.append(("hit", f"Lights circuit reading ({lights} Wh) is on the high side. Since lights usage tracks "
                         "closely with occupancy in this model, check for lights left on in unoccupied rooms."))
else:
    tips.append(("ok", f"Lights circuit reading ({lights} Wh) is within a typical range."))

if t3 > 22 or rh3 > 50:
    tips.append(("hit", f"Kitchen-zone conditions (T3={t3}°C, RH_3={rh3}%) are elevated, this is the strongest "
                         "single-room predictor after time of day. Worth checking kitchen appliance efficiency."))
else:
    tips.append(("ok", "Kitchen-zone conditions are within a typical range."))

if rh_out > 80 or pressure < 740:
    tips.append(("hit", "Outdoor humidity/pressure conditions suggest weather-linked (likely HVAC) load, "
                         "sealing and insulation improvements would help most under these conditions."))
else:
    tips.append(("ok", "Outdoor conditions are unlikely to be driving weather-linked load right now."))

for kind, text in tips:
    css_class = "tip-hit" if kind == "hit" else "tip-ok"
    st.markdown(f'<div class="{css_class}">{text}</div>', unsafe_allow_html=True)

st.caption("Chronological 80/20 split, not shuffled · Linear Regression selected over Random Forest, "
           "Ridge, Lasso, KNN, Extra Trees, and Decision Tree on test-set performance.")
