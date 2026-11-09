import numpy as np
import pandas as pd
import pytest

from readmission.data import clean, split
from readmission.evaluate import best_threshold, net_savings
from readmission.features import CATEGORICAL, NUMERIC, EncounterFeatures, icd9_group


@pytest.mark.parametrize("code,group", [
    ("428", "Circulatory"), ("786", "Respiratory"), ("250.83", "Diabetes"),
    ("V57", "Other"), (np.nan, "Missing"), ("996", "Injury"), ("715", "Musculoskeletal"),
])
def test_icd9_group(code, group):
    assert icd9_group(code) == group


def test_features_handle_missing_columns():
    # A caller that only knows a few fields must still get a full feature row.
    out = EncounterFeatures().transform(pd.DataFrame([{"age": "[50-60)", "insulin": "Up"}]))
    assert list(out.columns) == NUMERIC + CATEGORICAL
    assert out.loc[0, "age_years"] == 55
    assert out.loc[0, "n_meds_up"] == 1
    assert out.loc[0, "insulin_active"] == 1


def test_clean_drops_deceased_and_builds_target():
    df = pd.DataFrame({
        "discharge_disposition_id": [1, 11, 1, 13],
        "gender": ["Male", "Female", "Unknown/Invalid", "Male"],
        "readmitted": ["<30", "NO", ">30", "<30"],
    })
    out = clean(df)
    assert len(out) == 1
    assert out["readmitted_30d"].tolist() == [1]


def test_split_has_no_patient_overlap():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"patient_nbr": rng.integers(0, 300, 2000), "x": rng.random(2000),
                       "readmitted_30d": rng.integers(0, 2, 2000)})
    train, valid, test = split(df, 0.2, 0.2, 0)
    assert not set(train.patient_nbr) & set(test.patient_nbr)
    assert not set(train.patient_nbr) & set(valid.patient_nbr)


def test_business_logic():
    business = {"readmission_cost": 10000, "intervention_cost": 500, "intervention_success": 0.5}
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([.9, .8, .7, .1, .1, .1, .1, .1, .1, .1])
    r = net_savings(y, p, 0.5, business)
    # 2 true positives * 50% prevented * $10k - 3 flagged * $500
    assert r["net_savings"] == pytest.approx(2 * 0.5 * 10000 - 3 * 500)
    # Flagging all 10 nets $5,000; flagging only the top 3 nets $8,500, so the
    # optimum is the lowest grid threshold that excludes the 0.1-risk patients.
    assert best_threshold(y, p, business) == pytest.approx(0.11)
