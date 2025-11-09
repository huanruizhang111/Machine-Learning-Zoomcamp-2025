import pickle

from fastapi import FastAPI
import uvicorn

from typing import Dict, Any

app = FastAPI(title="patient-heart-disease-risk-prediction")

with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


def predict_heart_disease_risk_single(patient):
    risk = pipeline.predict_proba(patient)[0, 1]
    return float(risk)


@app.post("/predict_heart_disease_risk")
def predict_heart_disease_risk(patient: Dict[str, Any]):
    prob = predict_heart_disease_risk_single(patient)

    return {
        "risk_probability": prob,
        "risk": bool(prob >= 0.5)
    }

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