# AGENTS.md: Predicción de Consumo de Alcohol en Rusia (2024)

## 1. Descripción del Proyecto
Red neuronal profunda para predecir el consumo de alcohol puro per cápita en regiones de Rusia para 2024. Problema de **regresión supervisada** usando datos históricos 2017-2023.

**Variable objetivo**: `Consumption of alcoholic beverages (in liters of pure alcohol per capita)`

## 2. Dataset
- **Formato original**: CSV con columnas `Region, Year, Type of alcoholic beverages, Consumption (thousands decaliters), Consumption (liters per capita), Consumption (liters of pure alcohol per capita)`
- **Tipos de bebida**: Wine, Beer, Vodka, Sparkling wine, Brandy, Cider
- **Regiones**: ~85 regiones de Rusia
- **Años**: 2017-2023
- **Encoding issue**: `�ider` debe tratarse como `Cider`

## 3. Preprocesamiento
- **Pivot a formato ancho (features por año)**: Cada fila = una combinación Región + Tipo de bebida.
  - Features: `alcohol_puro_2017, alcohol_puro_2018, ..., alcohol_puro_2022` (6 variables numéricas)
  - Target: `alcohol_puro_2023`
  - Para predecir 2024: incluir `alcohol_puro_2023` como feature adicional
- **Codificación categórica**: Región y Tipo de bebida → One-hot encoding
- **Normalización**: Estandarización (Z-score) de las variables numéricas

## 4. Split de Datos
- **Train/Validation/Test estático**: 70/15/15
  - Train: ~357 muestras
  - Validation: ~77 muestras
  - Test: ~77 muestras (usando 2023 como referencia real)
- Sin shuffle temporal (serie temporal)

## 5. Stack Tecnológico
- **Lenguaje**: Python 3.10+
- **Framework**: PyTorch
- **Manejo de datos**: pandas, numpy, scikit-learn
- **Métricas**: MSE, MAE, R²
- **Hardware**: CPU (GPU opcional si está disponible)

## 6. Arquitectura del Modelo
- **Input**: ~97 features (85 regiones one-hot + 6 bebidas one-hot + 6 años)
- **Capas ocultas**: 2-3 capas fully-connected con 64-128 neuronas cada una
- **Activación**: ReLU
- **Dropout**: 0.2-0.3 después de cada capa oculta
- **Salida**: 1 neurona con activación lineal
- **Loss**: MSELoss

## 7. Entrenamiento
- **Optimizador**: Adam (lr inicial ~0.001)
- **Batch size**: 16-32
- **Early stopping**: Paciencia de 10-15 épocas en validation loss
- **Regularización adicional**: Weight decay (L2) opcional

## 8. Evaluación
- **Test final**: Sobre el split de test (2023)
- **Predicción 2024**: Usar modelo entrenado con features 2017-2023 para predecir 2024

## 9. Estructura del Proyecto
```
src/
  data/        → carga, limpieza, pivot, splits
  model/       → definición de la red neuronal
  train/       → loop de entrenamiento, early stopping
  predict/     → generación de predicciones 2024
  evaluate/    → cálculo de métricas
notebooks/     → exploración y visualización
data/          → dataset original
```

## 10. Convenciones
- No agregar librerías sin consultar
- Separar preprocesamiento de la arquitectura del modelo
- Verificar funcionamiento antes de cada commit
- No commitear sin autorización explícita

## 11. Consideraciones Éticas
- El modelo es una caja negra con baja explicabilidad
- Evaluar posibles sesgos hacia regiones con datos atípicos (Cáucaso Norte vs. regiones del norte)
- Las predicciones son estimaciones, no verdades absolutas
