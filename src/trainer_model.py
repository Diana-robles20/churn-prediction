'''
ML ENGINEER: Entrenamiento del modelo
'''


''' Librerias '''
import os
import joblib # -> Almacenamiento del modelo

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, recall_score, f1_score


''' Entrenamiento del Modelo '''
#Función de entrenamiento del modelo
def train_and_save_model(X_train, y_train, X_test, y_test, config): 
	select_model = config['select_model']

	#Implementar modelos: RandomForest y LogisticRegression (fabrica de modelos)
	if select_model == 'RandomForest':
		model = RandomForestClassifier(n_estimators = config['rf_params']['n_estimators'], random_state = config['data']['random_state'])

	elif select_model == 'LogisticRegression':
		model = LogisticRegression(max_iter = config['lr_params']['max_iter'], random_state = config['data']['random_state'])

	else:
		raise ValueError(f"El modelo {select_model} no es soportado.")


	#Entrenar el modelo.
	model.fit(X_train, y_train)


	#Calcular métricas: accuracy, recall y f1 score
	y_pred = model.predict(X_test)

	metrics = {
		'Accuracy Score': accuracy_score(y_test, y_pred),
		'Recall Score': recall_score(y_test, y_pred),
		'F1 Score': f1_score(y_test, y_pred)
	}


	#Guardar el modelo entrenado
	save_model_path = config['paths']['model_path']
	os.makedirs(os.path.dirname(save_model_path), exist_ok = True)

	joblib.dump(model, save_model_path)
	print(f"El modelo fue almacenado exitosamente en: {save_model_path}")

	return metrics