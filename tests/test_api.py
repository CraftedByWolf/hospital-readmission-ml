import pytest

from readmission.config import load_config

MODEL = load_config()["paths"]["models_dir"] / "model.joblib"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="trained model not present")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from api.main import app
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict_defaults(client):
    r = client.post("/predict", json={})
    assert r.status_code == 200
    assert 0 < r.json()["readmission_risk"] < 1


def test_high_risk_patient_scores_higher(client):
    low = client.post("/predict", json={"number_inpatient": 0, "discharge_disposition_id": 1}).json()
    high = client.post("/predict", json={
        "number_inpatient": 6, "number_emergency": 4, "discharge_disposition_id": 3,
        "diag_1": "428", "time_in_hospital": 9, "number_diagnoses": 9}).json()
    assert high["readmission_risk"] > low["readmission_risk"]
    assert high["flag_for_followup"]


def test_validation_rejects_bad_input(client):
    assert client.post("/predict", json={"time_in_hospital": -3}).status_code == 422


def test_batch(client):
    r = client.post("/predict/batch", json=[{}, {"number_inpatient": 3}])
    assert r.status_code == 200 and len(r.json()) == 2
