import yaml

from data_loader import (
    load_and_preprocess_data,
    save_processed_data
)

if __name__ == "__main__":

    # Leer configuración
    with open("config/params.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Preprocesar datos
    X_train, X_test, y_train, y_test = load_and_preprocess_data(config)

    # Guardar datos procesados
    save_processed_data(
        X_train,
        X_test,
        y_train,
        y_test,
        config
    )
