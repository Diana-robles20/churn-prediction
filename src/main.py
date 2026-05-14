from trainer_model import train_and_save_model

if __name__ == "__main__":
    ''' Entrenamiento del Modelo '''
    print("Iniciando fase de entrenamiento...")

    metrics = train_and_save_model(X_train, y_train, X_test, y_test, config)

    print("\n Estos son los resultados del modelo:")
    for metric, value in metrics.items():
        print(f"{metric}: {value: .4f}")