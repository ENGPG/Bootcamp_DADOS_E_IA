from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

class DataRequest(BaseModel):
    features: list

# Carregar modelo e encoder
model = joblib.load('modelo_otimizado.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.post('/predict/')
def predict(data: DataRequest):
    X = np.array(data.features).reshape(1, -1)
    prediction = model.predict(X)
    predicted_class = label_encoder.inverse_transform(prediction)[0]
    return {'predicted_class': predicted_class}