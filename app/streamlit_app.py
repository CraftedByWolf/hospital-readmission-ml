"""Discharge-desk web app: enter a patient, get a readmission risk.

Run with:  streamlit run app/streamlit_app.py
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from readmission.predict import load_model, predict

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="Readmission Risk", page_icon="🏥", layout="wide")

model, meta = load_model()
threshold = meta["threshold"]

st.title("30-day readmission risk for diabetic patients")
st.caption(f"Model: {meta['selected_model']} (calibrated) · trained {meta['trained_at']} · "
           f"follow-up threshold {threshold:.0%}")

tab_score, tab_batch, tab_model = st.tabs(["Score a patient", "Score a CSV", "How the model performs"])

with tab_score:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Patient")
        age = st.selectbox("Age bracket", [f"[{a}-{a + 10})" for a in range(0, 100, 10)], index=7)
        gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
        race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"])
        st.subheader("History (past year)")
        inpatient = st.number_input("Inpatient stays", 0, 25, 0)
        emergency = st.number_input("Emergency visits", 0, 50, 0)
        outpatient = st.number_input("Outpatient visits", 0, 50, 0)
    with c2:
        st.subheader("This stay")
        los = st.slider("Length of stay (days)", 1, 14, 4)
        n_diag = st.slider("Number of diagnoses", 1, 16, 7)
        n_labs = st.slider("Lab procedures", 0, 130, 43)
        n_procs = st.slider("Procedures", 0, 6, 1)
        n_meds = st.slider("Medications", 1, 80, 16)
        diag_1 = st.selectbox("Primary diagnosis", {
            "428": "Heart failure", "414": "Coronary disease", "786": "Chest pain",
            "410": "Heart attack", "486": "Pneumonia", "491": "COPD", "250.8": "Diabetes",
            "715": "Osteoarthritis", "996": "Device complication", "599": "Urinary tract",
        }.keys(), format_func=lambda c: f"{c} - " + {
            "428": "Heart failure", "414": "Coronary disease", "786": "Chest pain",
            "410": "Heart attack", "486": "Pneumonia", "491": "COPD", "250.8": "Diabetes",
            "715": "Osteoarthritis", "996": "Device complication", "599": "Urinary tract",
        }[c])
    with c3:
        st.subheader("Discharge & treatment")
        discharge = st.selectbox("Discharged to", {1: "Home", 6: "Home with health service",
                                                   3: "Skilled nursing facility", 2: "Another hospital",
                                                   7: "Left against medical advice"}.items(),
                                 format_func=lambda kv: kv[1])[0]
        admission_type = st.selectbox("Admission type", {1: "Emergency", 2: "Urgent", 3: "Elective"}.items(),
                                      format_func=lambda kv: kv[1])[0]
        insulin = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"])
        metformin = st.selectbox("Metformin", ["No", "Steady", "Up", "Down"])
        a1c = st.selectbox("HbA1c result", ["None", "Norm", ">7", ">8"])
        change = st.checkbox("Diabetes medication changed")

    record = {
        "age": age, "gender": gender, "race": race, "number_inpatient": inpatient,
        "number_emergency": emergency, "number_outpatient": outpatient, "time_in_hospital": los,
        "number_diagnoses": n_diag, "num_lab_procedures": n_labs, "num_procedures": n_procs,
        "num_medications": n_meds, "diag_1": diag_1, "discharge_disposition_id": discharge,
        "admission_type_id": admission_type, "insulin": insulin, "metformin": metformin,
        "A1Cresult": a1c, "change": "Ch" if change else "No",
        "diabetesMed": "Yes" if (insulin != "No" or metformin != "No") else "No",
    }
    result = predict([record])[0]
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Readmission risk", f"{result['readmission_risk']:.1%}",
              f"{result['readmission_risk'] - 0.11:+.1%} vs. average", delta_color="inverse")
    m2.metric("Risk band", result["risk_band"].upper())
    m3.metric("Recommend follow-up programme", "YES" if result["flag_for_followup"] else "no")

with tab_batch:
    st.write("Upload a CSV with the same columns as the UCI dataset (any missing column gets a typical default).")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        df = pd.read_csv(up, na_values="?")
        scored = pd.concat([df, pd.DataFrame(predict(df.to_dict(orient="records")))], axis=1)
        st.write(f"{scored['flag_for_followup'].sum()} of {len(scored)} patients flagged for follow-up")
        st.dataframe(scored.sort_values("readmission_risk", ascending=False), use_container_width=True)
        st.download_button("Download scored CSV", scored.to_csv(index=False), "scored.csv")

with tab_model:
    metrics_path = ROOT / "reports" / "metrics.json"
    if metrics_path.exists():
        m = meta["test_metrics"]
        b = meta["business_result_test"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("ROC-AUC (test)", f"{m['roc_auc']:.3f}")
        k2.metric("Top-10% lift", f"{meta['top_decile_lift']:.2f}x")
        k3.metric("Recall at threshold", f"{b['recall']:.0%}")
        k4.metric("Net savings / 1,000 discharges", f"${b['net_savings_per_1000_patients']:,.0f}")
        fig_dir = ROOT / "reports" / "figures"
        a, bcol = st.columns(2)
        a.image(str(fig_dir / "roc_pr.png"))
        bcol.image(str(fig_dir / "savings_curve.png"))
        a.image(str(fig_dir / "calibration.png"))
        bcol.image(str(fig_dir / "feature_importance.png"))
        st.subheader("Fairness check (test set)")
        st.dataframe(pd.read_csv(ROOT / "reports" / "fairness_test.csv"), use_container_width=True)
        with st.expander("Business assumptions"):
            st.json(meta["business_assumptions"])
