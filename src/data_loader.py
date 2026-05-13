
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# parámetro 'config' de la función : diccionario de
# config/params.yaml

def load_and_preprocess_data(config):

    # Cargar datos
    data_path = config['paths']['raw_data']

    df = pd.read_csv(data_path)

    # Limpiar TotalCharges  
    # df.info() muestra que ninguna variable presenta valores nulos
    # , lo cual se debe a que las celdas vacías tipo '' o ' ' cuentan 
    # como texto 

    df['TotalCharges'] = df['TotalCharges'].replace(' ', pd.NA)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

    median_value = df['TotalCharges'].median()

    df['TotalCharges'] = df['TotalCharges'].fillna(median_value)

    # Eliminar customerID
    df = df.drop(columns=['customerID'])

    # VARIABLES

    # Variables binarias
    binarias = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
        "Churn"
    ]

    # Variables con múltiples categorías
    categoricas = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod"
    ]

    # Variables numéricas
    numericas = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
]

    # CODIFICAR BINARIAS

    mapeo_binario = {
        "Yes": 1,
        "No": 0,
        "Male": 1,
        "Female": 0
    }

    for col in binarias:
        df[col] = df[col].map(mapeo_binario)


    # ONE HOT ENCODING

    df = pd.get_dummies(
        df,
        columns=categoricas,
        drop_first=True
    )

    df = df.replace({True: 1, False: 0})

    # NORMALIZAR NUMÉRICAS

    scaler = StandardScaler()

    df[numericas] = scaler.fit_transform(df[numericas])

    # Separar X e y
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )

    return X_train, X_test, y_train, y_test


import os

def save_processed_data(X_train, X_test, y_train, y_test, config):

    processed_path = config['paths']['processed_data']

    # Crear carpeta si no existe
    os.makedirs(processed_path, exist_ok=True)

    # Guardar archivos
    X_train.to_csv(f"{processed_path}/X_train.csv", index=False)
    X_test.to_csv(f"{processed_path}/X_test.csv", index=False)

    y_train.to_csv(f"{processed_path}/y_train.csv", index=False)
    y_test.to_csv(f"{processed_path}/y_test.csv", index=False)
