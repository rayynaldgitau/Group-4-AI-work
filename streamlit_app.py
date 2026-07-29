import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Smart Home Energy Predictor", layout="centered")

rf = joblib.load("rf_model.pkl")
lr = joblib.load("lr_model.pkl")
feature_cols = joblib.load("feature_cols.pkl")
defaults = joblib.load("feature_defaults.pkl")

st.title("AI-Powered Energy Usage Predictor")
st.caption("APT3010A Semester Project, trained on the UCI Appliances Energy Prediction dataset")
st.markdown("Adjust the inputs below to see predicted appliance energy use (Wh).")

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

model_choice = st.radio("Model", ["Linear Regression", "Random Forest"], horizontal=True)

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
model = rf if model_choice == "Random Forest" else lr
pred = model.predict(X)[0]

st.metric("Predicted Appliance Energy Use", f"{pred:.0f} Wh")

if model_choice == "Random Forest":
    st.warning(
        "Random Forest underperforms Linear Regression on the held-out test period in this "
        "project, it overfits to the Jan-April window. Shown for comparison."
    )

st.divider()
st.subheader("Energy-saving tips")
st.markdown("""
- Peak-hour shifting: usage peaks 5-7pm and 11am-1pm, run non-essential appliances off-peak.
- Kitchen zone temp/humidity is the strongest single-room predictor after time of day.
- Lights circuit load tracks closely with occupancy, turning off unused lights measurably
  correlates with lower total appliance draw in this dataset.
- Outdoor humidity and pressure matter, better sealing/insulation reduces weather-driven load.
""")
st.caption("Chronological 80/20 split, not shuffled. LR test R² = 0.10, RF test R² = -0.89.")
