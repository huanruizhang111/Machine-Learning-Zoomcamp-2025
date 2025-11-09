import requests

url = 'http://127.0.0.1:9696/predict_heart_disease_risk'

patient = {
  "gender": "female",
  "education_level": "college_or_above",
  "smoker": "smoker",
  "blood_pressure_medication": "not_on_bp_meds",
  "had_a_stroke": 1,
  "hypertensive": 1,
  "diabetes": "non_diabetic",
  "age": 91,
  "cigarettes_per_day": 20.0,
  "total_cholesterol": 243.0,
  "systolic_blood_pressure": 97.0,
  "diasolic_blood_pressure": 63.0,
  "bmi": 22.53,
  "heart_rate": 76.0,
  "glucose_level": 64.0
}

predictions = requests.post(url, json=patient).json()

if predictions["risk_probability"] > 0.5:
    print(predictions["risk_probability"])
    print('Patient is likely to have heart disease in 10 years')
else:
    print('Patient is not likely to have heart disease in 10 years')
