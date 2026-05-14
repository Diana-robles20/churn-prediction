from trainer_model import train_and_save_model

if __name__ == "__main__":
    ''' Entrenamiento del Modelo '''
    print("\nIniciando competencia entre modelos y fase de entrenamiento...")

    metrics = train_and_save_model(X_train, y_train, X_test, y_test, config)

    print("\nEstos son los resultados del modelo ganador:")
    for metric, value in metrics.items():
        if metric == 'Modelo Ganador':
            print(f"- {metric}: {value}")

        else:
            print(f"- {metric}: {value: .4f}")