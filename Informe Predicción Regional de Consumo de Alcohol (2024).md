Informe: Predicción Regional de Consumo de Alcohol (2024)

## 1. Introducción

El aprendizaje automático es una disciplina que permite a las máquinas aprender de los datos para mejorar su desempeño en tareas específicas sin programación explícita

Para este proyecto, se aplicará el aprendizaje profundo (Deep Learning), un modelo inspirado en el cerebro humano que utiliza capas de neuronas interconectadas para aprender niveles de abstracción

## 2. Contextualización y Formulación del Problema

El consumo de alcohol varía significativamente entre las regiones rusas y los tipos de bebida, lo que requiere un modelo capaz de modelar relaciones muy complejas

El problema se define como una tarea de regresión, ya que el objetivo es predecir una variable numérica continua (litros de consumo) en una escala real

El reto consiste en generalizar comportamientos inteligentes para el año 2024 a partir de los datos históricos de 2017 a 2023

## 3. Objetivos

Principal: Predecir el consumo de alcohol para el año 2024 desglosado por tipo de bebida y región.

Técnico: Minimizar el error cuadrático medio (MSE) o el error absoluto medio (MAE) para asegurar que las predicciones sean precisas

## 4. Datos y Variables

Siguiendo las restricciones, se utilizará únicamente la información del dataset:

Variables de Entrada (Features): Año (2017-2023), Región y Tipo de Bebida (Vino, Cerveza, etc.).

Variable Objetivo (Target): Volumen de consumo (valor numérico continuo)


Paradigma: Aprendizaje supervisado, donde el modelo aprende de pares de entrada y salida correctos (datos históricos etiquetados)

- 5. Propuesta de Solución: Red Neuronal Profunda (Feedforward)

Se propone una arquitectura de red neuronal de alimentación hacia adelante con las siguientes características:

Jerarquía de conceptos: Las capas iniciales detectarán patrones simples en las tendencias regionales y las capas posteriores los combinarán para reconocer estructuras de consumo complejas

Capas Ocultas: Múltiples capas interconectadas para procesar la información de las capas anteriores

Capa de Salida: Utilizará una función de activación lineal, necesaria para estimar valores numéricos en problemas de regresión

- 6. Preprocesamiento de Datos

Codificación: Las etiquetas de las regiones y tipos de bebida deben convertirse en formatos procesables, como vectores numéricos (one-hot encoding)

Normalización: Las cifras de consumo deben normalizarse o estandarizarse para facilitar el entrenamiento, especialmente si tienen rangos muy amplios entre diferentes bebidas

- 7. Entrenamiento y Regularización

Algoritmo: Se empleará backpropagation (propagación hacia atrás) y gradiente descendente para ajustar los pesos y minimizar el error

Prevención de Fallos: Para evitar el overfitting (sobreajuste), donde el modelo memoriza los datos pasados en lugar de aprender la tendencia, se aplicará dropout como técnica de regularización
