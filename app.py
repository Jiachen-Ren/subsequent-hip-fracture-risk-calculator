import json
import math
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fine-Gray fracture risk calculator", page_icon="🦴", layout="centered")
st.title("Subsequent fragility fracture risk calculator")
st.caption("Research demonstration · Fine–Gray competing-risk prediction · Not for clinical decision-making")
st.warning("Synthetic teaching-data model. No external validation. Do not use to guide patient care.")

model_path = Path(__file__).with_name("finegray_model.json")
if not model_path.exists():
    st.error("Missing finegray_model.json. Run export_model_to_json.R in RStudio and copy its output into this folder.")
    st.stop()
with model_path.open(encoding="utf-8") as f:
    model = json.load(f)

landmark = st.radio("Assessment time", [1, 3], format_func=lambda v: f"{v}-month landmark", horizontal=True)
st.info(f"Eligible only if still under observation, alive, and without a subsequent fracture at month {landmark}.")
c1, c2 = st.columns(2)
with c1:
    age = st.number_input("Age (years)", min_value=65, max_value=110, value=79, step=1)
    female = st.selectbox("Sex", ["Female", "Male"])
    cci = st.number_input("Charlson Comorbidity Index", min_value=0, max_value=20, value=2, step=1)
    prior = st.selectbox("Previous fragility fracture", ["No", "Yes"])
with c2:
    bi_pre = st.number_input("Prefracture BI (0–100)", min_value=0, max_value=100, value=90, step=5)
    fim_pre = st.number_input("Prefracture FIM-L (2–14)", min_value=2, max_value=14, value=12, step=1)
    bi_now = st.number_input(f"BI at {landmark} month(s) (0–100)", min_value=0, max_value=100, value=70, step=5)
    fim_now = st.number_input(f"FIM-L at {landmark} month(s) (2–14)", min_value=2, max_value=14, value=9, step=1)

features = {
    "age_years": float(age),
    "female_1_yes": float(female == "Female"),
    "CCI": float(cci),
    "prior_fragility_fracture_1_yes": float(prior == "Yes"),
    "barthel_prefracture_0_100": float(bi_pre),
    "fim_l_prefracture_2_14": float(fim_pre),
    "bi_recovery": (float(bi_now) - float(bi_pre)) / 10.0,
    "fim_recovery": float(fim_now) - float(fim_pre),
}
m = model["landmarks"][str(landmark)]
coefficients = m["predictors"]
missing = set(coefficients) - set(features)
if missing:
    st.error(f"Unknown predictors in exported model: {sorted(missing)}")
    st.stop()
eta = sum(float(coefficients[k]) * features[k] for k in coefficients)
h0 = float(m["baseline_cum_subhazard"])
risk = -math.expm1(-h0 * math.exp(min(eta, 500.0)))
risk = min(max(risk, 0.0), 1.0)

st.divider()
st.metric("Predicted cumulative incidence through postoperative month 12", f"{risk*100:.1f}%")
st.progress(min(max(risk, 0.0), 1.0))
st.caption(f"Prediction interval: month {landmark} to month 12. Death before fracture is treated as a competing event.")
st.caption("This is a model-based probability, not an observed individual outcome.")
with st.expander("Calculation details"):
    st.write(f"BI change: {bi_now-bi_pre:+.0f} points; BI recovery covariate: {features['bi_recovery']:+.2f} per 10 points")
    st.write(f"FIM-L change: {fim_now-fim_pre:+.0f} points")
    st.write(f"Linear predictor: {eta:.8f}; baseline cumulative subdistribution hazard: {h0:.8f}")
    st.latex(r"F_1(t\mid x)=1-\exp\{-H_{10}(t)\exp(\beta^\top x)\}")
    st.dataframe(pd.DataFrame([{"Predictor":k, "Value":features[k], "Coefficient":float(coefficients[k])} for k in coefficients]),hide_index=True)
