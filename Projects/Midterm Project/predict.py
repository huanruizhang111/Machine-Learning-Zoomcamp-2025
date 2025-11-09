import pickle

with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

patient = {
    'gender': 'female',
    'education_level': 'college_or_above',
    'smoker': 'smoker',
    'blood_pressure_medication': 'not_on_bp_meds',
    'had_a_stroke': 1,
    'hypertensive': 1,
    'diabetes': 'non_diabetic',
    'age': 91,
    'cigarettes_per_day': 20.0,
    'total_cholesterol': 243.0,
    'systolic_blood_pressure': 97.0,
    'diasolic_blood_pressure': 63.0,
    'bmi': 22.53,
    'heart_rate': 76.0,
    'glucose_level': 64.0
}

risk = pipeline.predict_proba(patient)[0, 1]
print(f'The predicted prob of 10-year risk of coronary heart disease is {risk:.3f}')

if risk > 0.5:
    print('The patient is at high risk of coronary heart disease')
else:
    print('The patient is at low risk of coronary heart disease')