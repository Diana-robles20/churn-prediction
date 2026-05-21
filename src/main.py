'''
MLOps ENGINEER
Orquestación del pipeline
'''

import yaml

from src.data_loader import (
    load_data,
    save_processed_data
)

from src.trainer_model import train_and_save_model


def load_config(path='config/params.yaml'):

    with open(path, 'r') as file:
        config = yaml.safe_load(file)

    return config


def main():

    print("\nCargando configuración...")
    config = load_config()

    print("\nCargando y procesando datos...")

    X_train, X_test, y_train, y_test = load_data(config)

    # Guardar datos procesados
    save_processed_data(
        X_train,
        X_test,
        y_train,
        y_test,
        config
    )

    print("\nIniciando competencia entre modelos y fase de entrenamiento...")

    metrics = train_and_save_model(
        X_train,
        y_train,
        X_test,
        y_test,
        config
    )

    print("\nEstos son los resultados del modelo ganador:")

    for metric, value in metrics.items():

        if metric == 'Modelo Ganador':
            print(f"- {metric}: {value}")

        else:
            print(f"- {metric}: {value:.4f}")


if __name__ == "__main__":
    main()