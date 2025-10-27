import requests

url = 'http://127.0.0.1:9696/predict'
# url = 'https://redesigned-funicular-pjgr9g7wpgw9c7xq9-9696.app.github.dev/05-deployment/practice/predict'
# url = 'https://mlzoomcamp-flask-uv.fly.dev/predict'

customer = {
    "gender": "male",
    "seniorcitizen": 0,
    "partner": "no",
    "dependents": "yes",
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 1,
    "monthlycharges": 29.85,
    "totalcharges": 129.85
}

response = requests.post(url, json=customer)

#print("Status code:", response.status_code)
#print("Response text:", response.text)  # <-- Add this line

# Only try to parse JSON if it looks valid
#try:
#    predictions = response.json()
#    if predictions['churn']:
#        print('customer is likely to churn, send promo')
#    else:
#        print('customer is not likely to churn')
#except Exception as e:
#    print("Failed to parse JSON:", e)


predictions = response.json()

if predictions['churn']:
    print('customer is likely to churn, send promo')
else:
    print('customer is not likely to churn')