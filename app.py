import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = os.getenv("MODEL_PATH", "model/titanic_model.joblib")
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Titanic Survival Predictor")


class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float | None = None
    SibSp: int = 0
    Parch: int = 0
    Fare: float | None = None
    Embarked: str | None = "S"


@app.get("/")
def health_check():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
def predict_survival(passenger: Passenger):
    df = pd.DataFrame([passenger.model_dump()])
    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][1])
    return {
        "survived": pred,
        "survival_probability": round(prob, 4),
    }
