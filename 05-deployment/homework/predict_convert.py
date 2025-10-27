import pickle
from typing import Literal
from pydantic import BaseModel, Field

from fastapi import FastAPI
import uvicorn

app = FastAPI(title="customer-convert-prediction")

with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


def predict_single(customer):
    result = pipeline.predict_proba(customer)[0, 1]
    return float(result)


from typing import Dict, Any

@app.post("/predict_convert")
def predict_convert(customer: Dict[str, Any]):
    prob = predict_single(customer)

    return {
        "convert_probability": prob,
        "convert": bool(prob >= 0.5)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)

# https://redesigned-funicular-pjgr9g7wpgw9c7xq9-9696.app.github.dev/docs#/default/predict_convert_predict_convert_post