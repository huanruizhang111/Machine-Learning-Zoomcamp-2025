import pickle

from fastapi import FastAPI
import uvicorn

from typing import Dict, Any

from pydantic import BaseModel, Field
from typing import Literal

app = FastAPI(title="patient-heart-disease-risk-prediction")

class HeartDiseaseInput(BaseModel):
    gender: Literal["female", "male"]
    education_level: Literal[
        "less_than_high_school",
        "high_school",
        "some_college_or_vocational_school",
        "college_or_above"
    ]
    smoker: Literal["non_smoker", "smoker"]
    blood_pressure_medication: Literal["not_on_bp_meds", "on_bp_meds"]
    had_a_stroke: Literal[0, 1]
    hypertensive: Literal[0, 1]
    diabetes: Literal["non_diabetic", "diabetic"]

    age: float = Field(..., ge=0, le=120, description="Age in years")
    cigarettes_per_day: float = Field(..., ge=0)
    total_cholesterol: float = Field(..., ge=0)
    systolic_blood_pressure: float = Field(..., ge=0)
    diasolic_blood_pressure: float = Field(..., ge=0)
    bmi: float = Field(..., ge=00)
    heart_rate: float = Field(..., ge=0)
    glucose_level: float = Field(..., ge=0)

class PredictResponse(BaseModel):
    risk_probability: float
    risk: bool

with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


def predict_heart_disease_risk_single(patient):
    risk = pipeline.predict_proba(patient)[0, 1]
    return float(risk)


@app.post("/predict")
def predict(patient: HeartDiseaseInput) -> PredictResponse:
    prob = predict_heart_disease_risk_single(patient.dict())

    return PredictResponse(
        risk_probability = prob,
        risk = bool(prob >= 0.5)
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)


# Example patient data in JSON format for FastAPI docs testing:
#{
#  "gender": "female",
#  "education_level": "college_or_above",
#  "smoker": "smoker",
#  "blood_pressure_medication": "not_on_bp_meds",
#  "had_a_stroke": 1,
#  "hypertensive": 1,
#  "diabetes": "non_diabetic",
#  "age": 91,
#  "cigarettes_per_day": 20.0,
#  "total_cholesterol": 243.0,
#  "systolic_blood_pressure": 97.0,
#  "diasolic_blood_pressure": 63.0,
#  "bmi": 22.53,
#  "heart_rate": 76.0,
#  "glucose_level": 64.0
#}

# Use 127.0.0.1 for curl requests from local machine instead of 0.0.0.0
# because in GitHub Codespaces, services (like Uvicorn on port 9696) are not public by default.
# they’re private to the container. The 127.0.0.1 can be found under ports tab in Codespaces UI.