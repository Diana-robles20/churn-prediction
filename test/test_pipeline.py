'''
QA TESTS
Pruebas unitarias del pipeline
'''

import yaml

from src.data_loader import load_data
from src.trainer_model import train_and_save_model


# Cargar configuración
with open('config/params.yaml', 'r') as file:
    config = yaml.safe_load(file)


def test_load_data():

    X_train, X_test, y_train, y_test = load_data(config)

    # Verificar que no estén vacíos
    assert X_train.empty == False
    assert X_test.empty == False
    assert y_train.empty == False
    assert y_test.empty == False


def test_train_model():

    X_train, X_test, y_train, y_test = load_data(config)

    metrics = train_and_save_model(
        X_train,
        y_train,
        X_test,
        y_test,
        config
    )

    # Verificar que sea diccionario
    assert isinstance(metrics, dict)

    # Verificar métricas
    assert 'Accuracy Score' in metrics
    assert 'Recall Score' in metrics
    assert 'F1 Score' in metrics