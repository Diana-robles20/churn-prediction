from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Cargar modelo entrenado
model = joblib.load("models/model.pkl")


@app.get("/")
def home():

    return {
        "message": "API de predicción de churn funcionando"
    }


@app.get("/health")
def health():

    return {
        "status": "ok"
    }


@app.get("/predict")
def predict():

    # Crear fila con ceros
    example_data = pd.DataFrame(
        [[0] * len(model.feature_names_in_)],
        columns=model.feature_names_in_
    )

    # Predicción
    prediction = model.predict(example_data)

    result = int(prediction[0])

    return {
        "prediction": result
    }