# AGENTS.md: Predicción de Consumo de Alcohol en Rusia (2024)

## 1. Descripción del Proyecto
Red neuronal profunda para predecir el consumo de alcohol puro per cápita en regiones de Rusia para 2024. Problema de **regresión supervisada** usando datos históricos 2017-2023.

**Variable objetivo**: `Consumption of alcoholic beverages (in liters of pure alcohol per capita)`

## 2. Dataset
- **Formato original**: CSV con columnas `Region, Year, Type of alcoholic beverages, Consumption (thousands decaliters), Consumption (liters per capita), Consumption (liters of pure alcohol per capita)`
- **Tipos de bebida**: Wine, Beer, Vodka, Sparkling wine, Brandy, Cider, Liqueurs (7 tipos)
- **Regiones**: 85 regiones de Rusia
- **Años**: 2017-2023
- **Total filas**: 4,165
- **Encoding**: cp1252; `[^A-Za-z]ider` se normaliza a `Cider`

## 3. Preprocesamiento
- **Pivot a formato ancho (features por año)**: Cada fila = una combinación Región + Tipo de bebida.
  - 3 métricas (pure alcohol, liters per capita, thousands decaliters) × 6 años = 18 columnas numéricas
  - Target: `alc_2023` (pure alcohol 2023)
  - Para predecir 2024: incluir `alc_2023` como feature adicional (20 numéricas total)
- **Codificación categórica**: Región y Tipo de bebida → One-hot encoding (~96 dummies)
- **Normalización**: Estandarización (Z-score) de las variables numéricas con `StandardScaler`
- **Muestras totales**: 595 (85 regiones × 7 bebidas)

## 4. Split de Datos
- **Train/Validation/Test estático**: 70/15/15
  - Train: 415 muestras
  - Validation: 90 muestras
  - Test: 90 muestras
- Sin shuffle temporal (serie temporal)

## 5. Stack Tecnológico
- **Lenguaje**: Python 3.10+
- **Framework**: PyTorch 2.0+
- **Manejo de datos**: pandas, numpy, scikit-learn
- **Visualización**: matplotlib, seaborn
- **Entorno**: Docker con hot-reload (volumen montado)
- **Jupyter**: Servicio separado en docker-compose, puerto 8888, token automático

## 6. Arquitectura del Modelo
- **Input**: ~110 features (depende de regiones en train)
- **Capas ocultas**: 2 capas fully-connected [128, 64]
- **Activación**: ReLU
- **Dropout**: 0.3 después de cada capa oculta
- **Salida**: 1 neurona con activación lineal
- **Loss**: MSELoss
- **Parámetros**: ~22,500

## 7. Entrenamiento
- **Optimizador**: Adam (lr=0.001, weight_decay=1e-5)
- **Batch size**: 32
- **Max épocas**: 200
- **Early stopping**: Paciencia 15 épocas en validation loss (restaura mejores pesos)
- **Checkpoint**: `output/best_model.pth`

## 8. Evaluación
- **Test final**: Sobre el split de test (2023) → R² > 0.98
- **Baseline**: LinearRegression (R² similar)
- **Métricas**: MSE, MAE, R²
- **Predicción 2024**: Usar modelo entrenado con features 2017-2023
- **Outputs CSVs**: predicciones_2024.csv, ranking_regiones.csv, ranking_bebidas.csv

## 9. Cómo ejecutar
```bash
# Pipeline completo
docker compose run --remove-orphans app

# Jupyter Notebook (http://localhost:8888, token en logs)
docker compose up jupyter
```

## 10. Estructura del Proyecto
```
├── .dockerignore
├── .gitignore
├── AGENTS.md
├── Dockerfile
├── LICENSE
├── README.md
├── docker-compose.yml
├── requirements.txt
├── notebooks/
│   └── exploracion.ipynb     → pipeline interactivo + análisis libre
├── src/
│   ├── __init__.py
│   ├── config.py              → Config dataclass
│   ├── main.py                → orquestación end-to-end
│   ├── visualize.py           → 5 tipos de gráficas
│   ├── data/
│   │   ├── dataset.py         → AlcoholDataset (PyTorch)
│   │   ├── loader.py          → CSVLoader con detección de encoding
│   │   ├── preprocessor.py    → pivot, one-hot, scaler
│   │   └── splitter.py        → train/val/test split
│   ├── model/
│   │   └── architecture.py    → AlcoholPredictor (110→128→64→1)
│   ├── training/
│   │   ├── early_stopping.py  → EarlyStopping con restauración
│   │   └── trainer.py         → loop de entrenamiento
│   ├── predict/
│   │   └── predictor.py       → predicción 2024 + rankings
│   └── evaluate/
│       └── metrics.py         → MSE, MAE, R² + baseline sklearn
├── data/ (no usado; CSV está en raíz)
├── output/ (generado en runtime, ignorado por git)
└── Consumpt... (dataset original)
```

## 11. Convenciones
- No agregar librerías sin consultar
- Separar preprocesamiento de la arquitectura del modelo
- Verificar funcionamiento antes de cada commit
- No commitear sin autorización explícita

## 12. Consideraciones Éticas
- El modelo es una caja negra con baja explicabilidad
- Evaluar posibles sesgos hacia regiones con datos atípicos (Cáucaso Norte vs. regiones del norte)
- Las predicciones son estimaciones, no verdades absolutas
