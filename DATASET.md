# Documentación del Dataset - Telco Customer Churn

## Nombre : Telco Customer Churn Dataset

---

## Descripción General

Este dataset contiene información de clientes de una empresa de telecomunicaciones y se utiliza para predecir si un cliente abandonará el servicio.

El problema corresponde a una tarea de clasificación binaria donde la variable objetivo es:

* `Churn = Yes` → El cliente abandonó el servicio.
* `Churn = No` → El cliente permaneció en la empresa.

---

## Objetivo del Dataset

El objetivo principal es analizar el comportamiento de los clientes para identificar patrones asociados al abandono del servicio y construir modelos de Machine Learning capaces de predecir el churn.

Este tipo de análisis es importante para reducir pérdida de clientes, mejorar estrategias de retención, optimizar campañas de marketing e incrementar ingresos de la empresa.

---

## 📂 Fuente del Dataset

Dataset disponible públicamente en Kaggle:

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Archivo utilizado:

```text
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

## Variables Principales

El dataset contiene información como:

* Género del cliente.
* Antigüedad del servicio (`tenure`).
* Tipo de contrato.
* Servicios contratados.
* Facturación mensual (`MonthlyCharges`).
* Facturación total (`TotalCharges`).
* Método de pago.
* Estado de abandono (`Churn`).

---

## Aplicaciones del Dataset

Este dataset puede utilizarse para:

* Modelos de clasificación binaria.
* Sistemas de recomendación y retención.
* Análisis de comportamiento de clientes.
* Proyectos de MLOps.
* Prácticas de limpieza y preprocesamiento de datos.
* Evaluación de métricas de Machine Learning.

---

## ⚠️ Posibles Sesgos y Limitaciones

Aunque el dataset es ampliamente utilizado con fines educativos, presenta algunas limitaciones importantes:

* Puede no representar a todas las empresas de telecomunicaciones.
* Los datos corresponden a un contexto específico y pueden no generalizarse a otros países o mercados.
* Algunas variables pueden introducir sesgos relacionados con tipo de contrato o perfil económico del cliente.
* El dataset no contiene información demográfica profunda ni contexto temporal completo.
* Existe desbalance parcial entre clientes que abandonan y clientes que permanecen.

---

## 🔍 Consideraciones Éticas

El uso de modelos predictivos sobre abandono de clientes debe realizarse de manera responsable para evitar:

* Discriminación indirecta.
* Decisiones automatizadas injustas.
* Uso indebido de información personal.
* Falta de transparencia en las predicciones.

Por ello, es importante evaluar continuamente la equidad, interpretabilidad y transparencia de los modelos desarrollados.

---

## Uso en nuestro Proyecto

En este proyecto el dataset fue utilizado para la limpieza y preprocesamiento de datos, entrenamiento de modelos de clasificación, evaluación de métricas como Accuracy, Recall y F1-Score y la implementación de un pipeline básico de MLOps.
