# Análisis de Ética, Equidad y Transparencia - Modelo de Predicción de Churn

Este documento presenta un análisis ético detallado del modelo de predicción de abandono de clientes (churn) desarrollado en este repositorio, centrándose en la identificación de sesgos, la evaluación de la equidad y la promoción de la transparencia en el proceso de Machine Learning.

## 1. Análisis de Sesgos en los Datos

El modelo utiliza el dataset **Telco Customer Churn**, el cual contiene atributos demográficos y de servicio. Se han identificado los siguientes puntos críticos:

- **Atributos Sensibles:** El dataset incluye variables como `gender` (género), `SeniorCitizen` (ciudadano mayor), `Partner` (pareja) y `Dependents` (dependientes). 

El uso de estas variables puede introducir sesgos si el modelo aprende patrones históricos de discriminación en lugar de comportamientos de servicio genuinos.

- **Codificación de Género:** En el módulo `src/data_loader.py`, se observa una codificación binaria manual (`Male: 1, Female: 0`). Es fundamental monitorear si esta representación numérica influye desproporcionadamente en la importancia de las características del modelo.

- **Representación de Grupos:** Es necesario realizar un análisis de balanceo para asegurar que los diferentes subgrupos (por ejemplo, personas mayores vs. jóvenes) estén representados equitativamente y que el modelo no presente una tasa de error significativamente mayor para un grupo específico.

## 2. Evaluación de Equidad (Fairness)

Aunque el pipeline actual se centra en métricas de rendimiento técnico (Accuracy, Recall, F1-Score), la equidad debe ser una métrica de primer nivel:

| Dimensión de Equidad | Estado Actual | Recomendación |
| --- | --- | --- |
| **Paridad Demográfica** | No evaluada explícitamente. | Medir si la tasa de predicción de "Churn" es similar entre géneros. |
| **Igualdad de Oportunidades** | No evaluada explícitamente. | Asegurar que el modelo identifique correctamente el abandono (True Positive Rate) de igual manera para todos los grupos demográficos. |
| **Sesgo de Selección** | Riesgo moderado. | Evaluar si la eliminación de `customerID` y el tratamiento de valores nulos en `TotalCharges` afecta de manera desigual a ciertos perfiles de clientes. |

## 3. Transparencia y Explicabilidad

La transparencia es un pilar fundamental de este proyecto MLOps:

- **Reproducibilidad:** El uso de `config/params.yaml` permite que cualquier investigador replique exactamente los experimentos, lo cual es una práctica de transparencia técnica excelente.

- **Selección de Modelos:** El proyecto implementa **Logistic Regression**, **Random Forest** y **SVM**. Mientras que la Regresión Logística es altamente interpretable, modelos como Random Forest requieren técnicas adicionales (como SHAP o Feature Importance) para explicar por qué se tomó una decisión específica para un cliente.

- **Documentación de Decisiones:** La estructura modular separa claramente las responsabilidades (Data Engineer, ML Engineer, MLOps, QA), permitiendo una trazabilidad clara de quién y cómo se transforman los datos y se entrenan los modelos.

## 4. Mitigación y Buenas Prácticas

Para mejorar la postura ética del modelo, se proponen las siguientes acciones:

1. **Auditoría de Sesgos:** Implementar herramientas como *AI Fairness 360* o *Fairlearn* en la etapa de validación.

1. **Anonimización:** Asegurar que no se re-identifiquen individuos tras la eliminación de `customerID`.

1. **Monitoreo de Deriva (Drift):** Implementar alertas para detectar si el modelo comienza a actuar de forma sesgada ante cambios en el comportamiento de los nuevos clientes en producción.

---
