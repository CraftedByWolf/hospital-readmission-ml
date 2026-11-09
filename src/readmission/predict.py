"""Load the trained model and score encounters."""
import json
from functools import lru_cache

import joblib
import pandas as pd

from .config import load_config


@lru_cache(maxsize=1)
def load_model():
    models_dir = load_config()["paths"]["models_dir"]
    model = joblib.load(models_dir / "model.joblib")
    metadata = json.loads((models_dir / "metadata.json").read_text())
    return model, metadata


def risk_band(p: float, threshold: float) -> str:
    if p >= threshold * 1.5:
        return "high"
    if p >= threshold:
        return "elevated"
    return "low"


def predict(records: list[dict]) -> list[dict]:
    model, meta = load_model()
    threshold = meta["threshold"]
    probs = model.predict_proba(pd.DataFrame(records))[:, 1]
    return [{
        "readmission_risk": round(float(p), 4),
        "flag_for_followup": bool(p >= threshold),
        "risk_band": risk_band(p, threshold),
        "threshold": threshold,
    } for p in probs]
