"""Feature engineering that turns one raw hospital encounter into model inputs.

Everything lives in a scikit-learn transformer so the exact same logic runs
during training, in the API and in the web app.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MEDICATIONS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]

# Defaults used when a caller (e.g. the API) omits a field.
RAW_DEFAULTS = {
    "race": np.nan, "gender": "Female", "age": "[60-70)",
    "admission_type_id": 1, "discharge_disposition_id": 1, "admission_source_id": 7,
    "time_in_hospital": 4, "payer_code": np.nan, "medical_specialty": np.nan,
    "num_lab_procedures": 43, "num_procedures": 1, "num_medications": 16,
    "number_outpatient": 0, "number_emergency": 0, "number_inpatient": 0,
    "diag_1": np.nan, "diag_2": np.nan, "diag_3": np.nan, "number_diagnoses": 7,
    "max_glu_serum": np.nan, "A1Cresult": np.nan, "change": "No", "diabetesMed": "Yes",
    **{m: "No" for m in MEDICATIONS},
}

NUMERIC = [
    "age_years", "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency", "number_inpatient",
    "number_diagnoses", "total_prior_visits", "n_meds_active", "n_meds_changed",
    "n_meds_up", "n_meds_down", "insulin_active", "on_diabetes_med", "med_change",
    "a1c_tested", "a1c_high", "glu_tested", "glu_high", "procedures_per_day",
    "meds_per_day", "labs_per_day",
]

CATEGORICAL = [
    "race", "gender", "admission_type", "discharge_group", "admission_source",
    "payer_code", "medical_specialty", "diag_1_group", "diag_2_group", "diag_3_group",
    "insulin",
]


def icd9_group(code) -> str:
    """Map an ICD-9 code to the disease groups used by Strack et al. (2014)."""
    if pd.isna(code):
        return "Missing"
    code = str(code)
    if code.startswith(("V", "E")):
        return "Other"
    try:
        v = float(code)
    except ValueError:
        return "Other"
    if 390 <= v <= 459 or v == 785:
        return "Circulatory"
    if 460 <= v <= 519 or v == 786:
        return "Respiratory"
    if 520 <= v <= 579 or v == 787:
        return "Digestive"
    if int(v) == 250:
        return "Diabetes"
    if 800 <= v <= 999:
        return "Injury"
    if 710 <= v <= 739:
        return "Musculoskeletal"
    if 580 <= v <= 629 or v == 788:
        return "Genitourinary"
    if 140 <= v <= 239:
        return "Neoplasms"
    return "Other"


_ADMISSION_TYPE = {1: "Emergency", 2: "Urgent", 3: "Elective", 7: "Trauma"}
_ADMISSION_SOURCE = {1: "Referral", 2: "Referral", 3: "Referral", 7: "Emergency",
                     4: "Transfer", 5: "Transfer", 6: "Transfer", 10: "Transfer",
                     18: "Transfer", 22: "Transfer", 25: "Transfer", 26: "Transfer"}


def _discharge_group(v) -> str:
    if v == 1:
        return "Home"
    if v in (6, 8):
        return "HomeHealth"
    if v in (3, 4, 5, 15, 22, 23, 24, 27, 28, 29, 30):
        return "Facility"
    if v == 2:
        return "OtherHospital"
    if v == 7:
        return "LeftAMA"
    return "Other"


class EncounterFeatures(BaseEstimator, TransformerMixin):
    """Raw encounter rows -> engineered feature frame (stateless)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, default in RAW_DEFAULTS.items():
            if col not in X:
                X[col] = default
        out = pd.DataFrame(index=X.index)

        out["age_years"] = X["age"].astype(str).str.extract(r"\[(\d+)-")[0].astype(float) + 5
        for c in ["time_in_hospital", "num_lab_procedures", "num_procedures",
                  "num_medications", "number_outpatient", "number_emergency",
                  "number_inpatient", "number_diagnoses"]:
            out[c] = pd.to_numeric(X[c], errors="coerce")
        out["total_prior_visits"] = (out["number_outpatient"] + out["number_emergency"]
                                     + out["number_inpatient"])

        meds = X[MEDICATIONS].astype(str)
        out["n_meds_active"] = meds.isin(["Steady", "Up", "Down"]).sum(axis=1)
        out["n_meds_changed"] = meds.isin(["Up", "Down"]).sum(axis=1)
        out["n_meds_up"] = (meds == "Up").sum(axis=1)
        out["n_meds_down"] = (meds == "Down").sum(axis=1)
        out["insulin_active"] = (meds["insulin"] != "No").astype(int)
        out["on_diabetes_med"] = (X["diabetesMed"] == "Yes").astype(int)
        out["med_change"] = (X["change"] == "Ch").astype(int)

        a1c = X["A1Cresult"].fillna("None").astype(str)
        glu = X["max_glu_serum"].fillna("None").astype(str)
        out["a1c_tested"] = (a1c != "None").astype(int)
        out["a1c_high"] = a1c.isin([">7", ">8"]).astype(int)
        out["glu_tested"] = (glu != "None").astype(int)
        out["glu_high"] = glu.isin([">200", ">300"]).astype(int)

        stay = out["time_in_hospital"].clip(lower=1)
        out["procedures_per_day"] = out["num_procedures"] / stay
        out["meds_per_day"] = out["num_medications"] / stay
        out["labs_per_day"] = out["num_lab_procedures"] / stay

        out["race"] = X["race"].astype(object).fillna("Missing").astype(str)
        out["gender"] = X["gender"].astype(str)
        adm_type = pd.to_numeric(X["admission_type_id"], errors="coerce")
        out["admission_type"] = adm_type.map(_ADMISSION_TYPE).fillna("Other")
        out["discharge_group"] = pd.to_numeric(
            X["discharge_disposition_id"], errors="coerce").map(_discharge_group)
        adm_src = pd.to_numeric(X["admission_source_id"], errors="coerce")
        out["admission_source"] = adm_src.map(_ADMISSION_SOURCE).fillna("Other")
        out["payer_code"] = X["payer_code"].astype(object).fillna("Missing").astype(str)
        out["medical_specialty"] = X["medical_specialty"].astype(object).fillna("Missing").astype(str)
        for d in ["diag_1", "diag_2", "diag_3"]:
            out[f"{d}_group"] = X[d].map(icd9_group)
        out["insulin"] = meds["insulin"]
        return out[NUMERIC + CATEGORICAL]


def build_preprocessor(scale_numeric: bool) -> Pipeline:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    columns = ColumnTransformer([
        ("num", Pipeline(num_steps), NUMERIC),
        # Rare categories (e.g. obscure specialties) are pooled to avoid overfitting.
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100,
                              sparse_output=False), CATEGORICAL),
    ], verbose_feature_names_out=False)
    return Pipeline([("features", EncounterFeatures()), ("columns", columns)])
