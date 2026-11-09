"""REST API for readmission risk scoring.

Run with:  uvicorn api.main:app --reload
Docs at:   http://127.0.0.1:8000/docs
"""
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from readmission.predict import load_model, predict

MedStatus = Literal["No", "Steady", "Up", "Down"]

app = FastAPI(
    title="Hospital Readmission Risk API",
    description="Scores the 30-day readmission risk of a diabetic patient at discharge.",
    version="1.0.0",
)


class Encounter(BaseModel):
    # Allow the remaining raw dataset columns (other medications etc.) to be passed through.
    model_config = ConfigDict(extra="allow")

    race: str | None = Field(None, examples=["Caucasian"])
    gender: Literal["Male", "Female"] = "Female"
    age: str = Field("[70-80)", description="Age bracket, e.g. [70-80)")
    admission_type_id: int = 1
    discharge_disposition_id: int = 1
    admission_source_id: int = 7
    time_in_hospital: int = Field(4, ge=1, le=30, description="Length of stay in days")
    medical_specialty: str | None = Field(None, examples=["InternalMedicine"])
    num_lab_procedures: int = Field(43, ge=0)
    num_procedures: int = Field(1, ge=0)
    num_medications: int = Field(16, ge=0)
    number_outpatient: int = Field(0, ge=0, description="Outpatient visits in prior year")
    number_emergency: int = Field(0, ge=0, description="Emergency visits in prior year")
    number_inpatient: int = Field(0, ge=0, description="Inpatient stays in prior year")
    diag_1: str | None = Field(None, examples=["428"], description="Primary ICD-9 code")
    diag_2: str | None = None
    diag_3: str | None = None
    number_diagnoses: int = Field(7, ge=1)
    max_glu_serum: Literal["None", "Norm", ">200", ">300"] | None = None
    A1Cresult: Literal["None", "Norm", ">7", ">8"] | None = None
    insulin: MedStatus = "No"
    metformin: MedStatus = "No"
    change: Literal["No", "Ch"] = "No"
    diabetesMed: Literal["Yes", "No"] = "Yes"


class Prediction(BaseModel):
    readmission_risk: float
    flag_for_followup: bool
    risk_band: Literal["low", "elevated", "high"]
    threshold: float


@app.get("/health")
def health() -> dict:
    _, meta = load_model()
    return {"status": "ok", "model": meta["selected_model"], "trained_at": meta["trained_at"]}


@app.get("/model-info")
def model_info() -> dict:
    _, meta = load_model()
    return meta


@app.post("/predict", response_model=Prediction)
def predict_one(encounter: Encounter) -> Prediction:
    return predict([encounter.model_dump()])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(encounters: list[Encounter]) -> list[Prediction]:
    return predict([e.model_dump() for e in encounters])
