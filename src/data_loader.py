'''
DATA ENGINEER
Carga y preprocesamiento de datos
'''

import pandas as pd

from sklearn.model_selection import train_test_split


def load_data(config):

    # Cargar dataset
    path = config['paths']['data_path']

    df = pd.read_csv(path)

    # Limpiar TotalCharges
    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'],
        errors='coerce'
    )

    df['TotalCharges'].fillna(
        df['TotalCharges'].median(),
        inplace=True
    )

    # Eliminar customerID
    df.drop('customerID', axis=1, inplace=True)

    # Variables binarias
    binary_cols = ['gender', 'Partner', 'Churn']

    for col in binary_cols:

        df[col] = df[col].map({
            'Male': 1,
            'Female': 0,
            'Yes': 1,
            'No': 0
        })

    # One Hot Encoding
    df = pd.get_dummies(df, drop_first=True)

    # Separar variables
    X = df.drop('Churn', axis=1)

    y = df['Churn']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )

    return X_train, X_test, y_train, y_test