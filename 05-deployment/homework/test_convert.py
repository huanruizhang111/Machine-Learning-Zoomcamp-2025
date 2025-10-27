import requests

url = 'http://127.0.0.1:9696/predict_convert'

client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}

predictions = requests.post(url, json=client).json()

if predictions["convert_probability"] > 0.5:
    print(predictions["convert_probability"])
    print('Client is likely to convert')
else:
    print('Client is not likely to convert')