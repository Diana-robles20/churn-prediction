'''
QA & Production Engineer
Predicción usando datos reales
'''

import os
import joblib
import pandas as pd


def predict():

    model_path = 'models/model.pkl'
    data_path = 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'

    # Verificar modelo
    if not os.path.exists(model_path):
        print("ERROR: El modelo no existe. Ejecuta src.main primero.")
        return

    # Cargar modelo
    model = joblib.load(model_path)

    print("Modelo cargado correctamente.")

    # Cargar dataset
    df = pd.read_csv(data_path)

    # Mismo preprocesamiento
    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'],
        errors='coerce'
    )

    df['TotalCharges'].fillna(
        df['TotalCharges'].median(),
        inplace=True
    )

    df.drop('customerID', axis=1, inplace=True)

    binary_cols = ['gender', 'Partner', 'Churn']

    for col in binary_cols:

        df[col] = df[col].map({
            'Male': 1,
            'Female': 0,
            'Yes': 1,
            'No': 0
        })

    df = pd.get_dummies(df, drop_first=True)

    # Tomar un cliente de ejemplo
    sample = df.drop('Churn', axis=1).iloc[[0]]

    # Predicción
    prediction = model.predict(sample)

    print("\nResultado de predicción:")

    if prediction[0] == 1:
        print("El cliente abandonará la empresa (Churn).")

    else:
        print("El cliente permanecerá en la empresa.")


if __name__ == "__main__":
    predict()