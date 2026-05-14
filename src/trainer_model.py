'''
ML ENGINEER: Entrenamiento del modelo
'''


''' Librerias '''
import os
import joblib # -> Almacenamiento del modelo

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, recall_score, f1_score


''' Entrenamiento del Modelo '''
#Función de entrenamiento del modelo
def train_and_save_model(X_train, y_train, X_test, y_test, config): 

	#Implementar modelos (fabrica de modelos): RandomForest y LogisticRegression/SVM (el usuario puede elegir entre estos dos)
	model_rf = RandomForestClassifier(n_estimators = config['rf_params']['n_estimators'], random_state = config['data']['random_state'])

	name_model = config['select_model']

	if name_model == 'LogisticRegression':
		model_sec = LogisticRegression(max_iter = config['lr_params']['max_iter'], random_state = config['data']['random_state'])

	elif name_model == 'SVM':
		model_sec = SVC(kernel = config['svm_params']['kernel'], random_state = config['data']['random_state'])
	
	else:
		raise ValueError(f"El modelo {name_model} no es soportado.")
	
	print(f"Entrenando modelos: Random Forest y {name_model}...")


	#Entrenar modelos.
	model_rf.fit(X_train, y_train)
	model_sec.fit(X_train, y_train)


	#Calcular métricas: accuracy, recall y f1 score (y mostrar solo para el mejor modelo)
	rf_ypred = model_rf.predict(X_test)
	sec_ypred = model_sec.predict(X_test)

	rf_acc = accuracy_score(y_test, rf_ypred)
	sec_acc = accuracy_score(y_test, sec_ypred)

	if rf_acc >= sec_acc:
		best_model = model_rf
		best_pred = rf_ypred
		best_name = 'Random Forest'
	
	else:
		best_model = model_sec
		best_pred = sec_ypred
		best_name = name_model

	metrics = {
		'Modelo Ganador': best_name,
		'Accuracy Score': accuracy_score(y_test, best_pred),
		'Recall Score': recall_score(y_test, best_pred),
		'F1 Score': f1_score(y_test, best_pred)
	}


	#Guardar el mejor modelo
	save_model_path = config['paths']['model_path']
	os.makedirs(os.path.dirname(save_model_path), exist_ok = True)

	joblib.dump(best_model, save_model_path)
	print(f"\n El mejor modelo fue {best_name}.")
	print(f"El modelo fue almacenado exitosamente en: {save_model_path}")

	return metrics