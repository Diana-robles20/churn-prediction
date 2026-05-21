# 📡 Proyecto Colaborativo MLOps: Predicción de Churn (Abandono de Clientes)

## 🎯 Objetivo del Proyecto
Construir un pipeline de Machine Learning modular, reproducible y colaborativo para predecir si un cliente de telecomunicaciones abandonará el servicio (**Churn**).

El proyecto simula un entorno laboral real donde diferentes roles especializados integran su código en un solo repositorio utilizando Git y buenas prácticas de MLOps.

---

## 📂 Dataset

Todos los equipos trabajaron con el dataset **Telco Customer Churn**.

- **Fuente:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Archivo:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Problema:** Clasificación binaria (`Churn = Yes/No`)

### 📥 Instrucciones para usar el dataset

1. Descargar el archivo CSV desde Kaggle.
2. Guardarlo en:

```text
data/raw/
```

3. El dataset NO debe subirse al repositorio Git.

---

## 👥 Roles y Responsabilidades

### 👷 Data Engineer (`src/data_loader.py`)

Responsable de:

- Cargar el dataset.
- Limpiar y transformar los datos.
- Convertir `TotalCharges` a numérico.
- Imputar valores faltantes.
- Eliminar `customerID`.
- Codificar variables binarias.
- Dividir los datos en entrenamiento y prueba.

---

### 🧠 ML Engineer (`src/trainer_model.py`)

Responsable de:

- Implementar múltiples modelos de Machine Learning.
- Entrenar los modelos.
- Comparar métricas.
- Seleccionar el mejor modelo.
- Guardar el modelo entrenado usando `joblib`.

Modelos implementados:

- Random Forest
- Logistic Regression
- SVM

---

### ⚙️ MLOps Engineer (`src/main.py` y `config/params.yaml`)

Responsable de:

- Orquestar el pipeline completo.
- Gestionar la configuración mediante YAML.
- Integrar todos los módulos.
- Garantizar la ejecución correcta del proyecto.

El pipeline se ejecuta con:

```bash
python -m src.main
```

---

### 🛡️ QA & Production Engineer (`src/predict.py` y `tests/`)

Responsable de:

- Crear el script de predicción.
- Cargar el modelo entrenado.
- Realizar predicciones de ejemplo.
- Implementar pruebas unitarias.
- Manejar errores de ejecución.

Las pruebas se ejecutan con:

```bash
pytest
```

---

## ⚙️ Configuración del Proyecto

El archivo `config/params.yaml` contiene los hiperparámetros y rutas principales del proyecto.

Ejemplo:

```yaml
select_model: 'LogisticRegression'

paths:
  model_path: 'models/model.pkl'
  data_path: 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'

data:
  test_size: 0.2
  random_state: 42

rf_params:
  n_estimators: 100

lr_params:
  max_iter: 3000

svm_params:
  kernel: 'linear'
```

---

## 🚀 Instalación y Ejecución

### 1. Crear entorno virtual

```bash
python -m venv venv
```

### 2. Activar entorno virtual

En Windows:

```bash
venv\Scripts\Activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar pipeline

```bash
python -m src.main
```

### 5. Ejecutar predicciones

```bash
python -m src.predict
```

### 6. Ejecutar pruebas unitarias

```bash
pytest
```

---

## 📂 Estructura del Proyecto

```text
churn-prediction/
├── config/
│   └── params.yaml
│
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   └── model.pkl
│
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── data_loader.py
│   ├── trainer_model.py
│   ├── main.py
│   └── predict.py
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Resultados del Mejor Modelo

Después de ejecutar el pipeline y comparar los modelos, el mejor modelo obtenido fue:

### 🏆 Modelo Ganador

`LogisticRegression`

### 📈 Métricas Obtenidas

| Métrica | Valor |
|---|---|
| Accuracy | 0.8211 |
| Recall | 0.6059 |
| F1-Score | 0.6420 |

El modelo entrenado fue almacenado exitosamente en:

```text
models/model.pkl
```

---

## 🧪 Pruebas Unitarias

Se implementaron pruebas unitarias básicas para validar:

- Que el preprocesamiento de datos no devuelva DataFrames vacíos.
- Que el entrenamiento del modelo retorne las métricas esperadas.
- Que el pipeline funcione correctamente.

Resultado:

```text
2 passed
```

---

## ✅ Checklist de Entrega

- [x] El comando `python -m src.main` ejecuta todo el pipeline sin errores.
- [x] El archivo `config/params.yaml` existe y controla hiperparámetros.
- [x] Hay al menos 2 modelos implementados.
- [x] `predict.py` carga el modelo y realiza una predicción.
- [x] Existen pruebas unitarias funcionales.
- [x] El README incluye métricas del mejor modelo.
- [x] El proyecto utiliza configuración modular y reproducible.

---

## 🤖 LLM Contribution

Durante el desarrollo del proyecto se utilizaron herramientas de inteligencia artificial como ChatGPT y Gemini para:

- Ayudar en la estructuración del pipeline de MLOps.
- Resolver errores de integración entre módulos.
- Generar y corregir pruebas unitarias.
- Apoyar en la configuración del entorno virtual y dependencias.
- Mejorar la documentación y organización del proyecto.

Todo el código generado fue revisado, comprendido y adaptado por los integrantes del equipo.

---

---

## 🌐 API REST con FastAPI

Como funcionalidad adicional, el proyecto incluye una API REST desarrollada con FastAPI para realizar inferencias utilizando el modelo entrenado.

### ▶️ Ejecutar API

```bash
uvicorn src.api:app --reload
```

La API estará disponible en:

```text
http://127.0.0.1:8000
```

### 📄 Documentación automática

FastAPI genera automáticamente documentación interactiva Swagger UI:

```text
http://127.0.0.1:8000/docs
```

### 🔍 Endpoints disponibles

| Endpoint | Método | Descripción |
|---|---|---|
| `/` | GET | Verifica que la API esté funcionando |
| `/health` | GET | Endpoint de estado |
| `/predict` | GET | Realiza una predicción usando el modelo entrenado |

### ✅ Ejemplo de respuesta

```json
{
  "prediction": 0
}
```

---

## 🚀 Conclusión

Este proyecto permitió aplicar conceptos fundamentales de MLOps, integración de pipelines, pruebas unitarias, control de versiones y trabajo colaborativo utilizando Machine Learning en un entorno similar al laboral.